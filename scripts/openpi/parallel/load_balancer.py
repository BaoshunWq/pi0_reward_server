#!/usr/bin/env python
"""
简单的负载均衡器，将请求分配到多个reward server
使用最少连接数策略(Least Connections)
"""
import argparse
import asyncio
import aiohttp
from aiohttp import web
import logging
from typing import List, Dict
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServerPool:
    """管理多个后端服务器的连接池"""
    
    def __init__(self, base_port: int, num_servers: int):
        self.servers = [
            {
                'url': f'http://localhost:{base_port + i}',
                'active_requests': 0,
                'total_requests': 0,
                'errors': 0,
                'last_error_time': 0
            }
            for i in range(num_servers)
        ]
        self.lock = asyncio.Lock()
    
    async def get_best_server(self) -> Dict:
        """选择当前负载最小的服务器"""
        async with self.lock:
            # 过滤掉最近有错误的服务器（30秒内）
            current_time = time.time()
            available_servers = [
                s for s in self.servers
                if current_time - s['last_error_time'] > 30
            ]
            
            if not available_servers:
                # 如果所有服务器都有错误，使用全部服务器
                available_servers = self.servers
            
            # 选择活跃请求数最少的服务器
            best_server = min(available_servers, key=lambda s: s['active_requests'])
            best_server['active_requests'] += 1
            best_server['total_requests'] += 1
            return best_server
    
    async def release_server(self, server: Dict, had_error: bool = False):
        """释放服务器"""
        async with self.lock:
            server['active_requests'] = max(0, server['active_requests'] - 1)
            if had_error:
                server['errors'] += 1
                server['last_error_time'] = time.time()
    
    def get_stats(self) -> List[Dict]:
        """获取所有服务器的统计信息"""
        return [
            {
                'url': s['url'],
                'active': s['active_requests'],
                'total': s['total_requests'],
                'errors': s['errors']
            }
            for s in self.servers
        ]


class LoadBalancer:
    def __init__(self, base_port: int, num_servers: int):
        self.pool = ServerPool(base_port, num_servers)
        self.session = None
    
    async def start(self):
        """启动负载均衡器"""
        timeout = aiohttp.ClientTimeout(total=600)  # 10分钟超时
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info(f"Load balancer started with {len(self.pool.servers)} backend servers")
    
    async def stop(self):
        """停止负载均衡器"""
        if self.session:
            await self.session.close()
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """健康检查端点"""
        stats = self.pool.get_stats()
        return web.json_response({
            'status': 'healthy',
            'servers': stats
        })
    
    async def handle_score(self, request: web.Request) -> web.Response:
        """
        按样本级别分发batch请求到多个后端服务器
        1. 拆分batch为单个样本
        2. 并发发送到不同服务器
        3. 收集结果并合并返回
        
        支持两种返回格式：
        - 普通模式：done_result = [0.5, 0.3, ...] (每个元素是 success_rate)
        - danger模式：done_result = [[0.5, 2], [0.3, 1], ...] (每个元素是 [success_rate, collision_count])
        """
        try:
            # 读取并解析请求体
            body = await request.json()
            
            responses = body.get('responses', [])
            metas = body.get('metas', [])
            reward_kwargs = body.get('reward_function_kwargs', {})
            
            # 检查是否启用danger模式（用于日志显示）
            use_danger = reward_kwargs.get('danger', False) or os.environ.get('USE_DANGER', '0') == '1'
            mode_str = "danger" if use_danger else "normal"
            
            num_samples = len(responses)
            logger.info(f"Received batch with {num_samples} samples ({mode_str} mode), splitting and distributing...")
            
            if num_samples == 0:
                return web.json_response({'error': 'Empty batch'}, status=400)
            
            # 创建单样本请求任务
            tasks = []
            for i in range(num_samples):
                # 构造单样本请求
                single_request = {
                    'responses': [responses[i]],
                    'metas': [metas[i]] if i < len(metas) else [{}],
                    'reward_function_kwargs': reward_kwargs
                }
                
                # 创建异步任务
                task = self._forward_single_sample(i, single_request)
                tasks.append(task)
            
            # 并发执行所有任务，若有异常直接抛出
            results = await asyncio.gather(*tasks)
            
            # 合并结果（支持普通模式和danger模式）
            # 普通模式：done_list[0] = 0.5 (float)
            # danger模式：done_list[0] = [0.5, 2] (list)
            merged_done_result = []
            merged_collision_result = []
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    raise TypeError(f"Sample {i}: Unexpected result type {type(result)}")
                if 'error' in result:
                    raise RuntimeError(f"Sample {i}: {result['error']}")
                if 'done_result' not in result:
                    raise KeyError(f"Sample {i}: missing done_result")

                done_list = result['done_result']
                if not isinstance(done_list, list) or len(done_list) == 0:
                    raise ValueError(f"Sample {i}: invalid done_result {done_list}")

                # 提取单个样本的结果（支持两种格式）
                sample_result = done_list[0]
                if use_danger:
                    # danger模式：sample_result = [success_rate, collision_count]
                    if isinstance(sample_result, (list, tuple)) and len(sample_result) >= 2:
                        merged_done_result.append(sample_result[0])
                        merged_collision_result.append(sample_result[1])
                    else:
                        # 兼容处理：如果不是列表格式，假设只有success_rate
                        merged_done_result.append(float(sample_result) if not isinstance(sample_result, (list, tuple)) else float(sample_result[0]))
                        merged_collision_result.append(0)
                    response_data = {
                        'done_result': merged_done_result,  # 兼容测试脚本
                        'collision_result': merged_collision_result,  # 兼容测试脚本
                        'num_samples': num_samples,
                        'num_errors': 0,
                    }
                else:
                    # 普通模式：sample_result = success_rate (float)
                    merged_done_result.append(float(sample_result))
                    response_data = {
                        'done_result': merged_done_result,  # 兼容测试脚本
                        'num_samples': num_samples,
                        'num_errors': 0,
                    }
        
            
            logger.info(f"Batch completed successfully: {num_samples} samples ({mode_str} mode)")
            # 只记录前3个结果，避免日志过长
            preview = merged_done_result[:3] if len(merged_done_result) > 3 else merged_done_result
            logger.info(f"Response preview (first {len(preview)} samples): {preview}")

            return web.json_response(response_data)
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response(
                {'error': f'Batch processing error: {str(e)}'},
                status=500
            )
    
    async def _forward_single_sample(self, sample_idx: int, request_data: dict) -> dict:
        """
        转发单个样本到后端服务器
        
        Args:
            sample_idx: 样本索引
            request_data: 单样本请求数据
            
        Returns:
            服务器响应的JSON数据
        """
        server = await self.pool.get_best_server()
        had_error = False
        
        try:
            target_url = f"{server['url']}/score"
            logger.info(f"Sample {sample_idx} -> {target_url} (active: {server['active_requests']})")
            
            async with self.session.post(
                target_url,
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Sample {sample_idx} completed on {server['url']}")
                    return result
                else:
                    error_text = await resp.text()
                    logger.error(f"Sample {sample_idx} failed: HTTP {resp.status}")
                    had_error = True
                    return {'error': f'HTTP {resp.status}: {error_text}'}
        
        except asyncio.TimeoutError:
            logger.error(f"Sample {sample_idx} timeout on {server['url']}")
            had_error = True
            return {'error': 'Timeout'}
        
        except aiohttp.ClientError as e:
            logger.error(f"Sample {sample_idx} client error: {e}")
            had_error = True
            return {'error': f'Client error: {str(e)}'}
        
        except Exception as e:
            logger.error(f"Sample {sample_idx} unexpected error: {e}")
            had_error = True
            return {'error': f'Unexpected error: {str(e)}'}
        
        finally:
            await self.pool.release_server(server, had_error)


def parse_args():
    parser = argparse.ArgumentParser(description="Load balancer for reward servers")
    parser.add_argument(
        '--listen_port',
        type=int,
        default=7000,
        help='Port to listen on for incoming requests'
    )
    parser.add_argument(
        '--base_port',
        type=int,
        default=6006,
        help='Base port of backend reward servers'
    )
    parser.add_argument(
        '--num_servers',
        type=int,
        default=1,
        help='Number of backend servers'
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # 创建负载均衡器
    lb = LoadBalancer(args.base_port, args.num_servers)
    await lb.start()
    
    # 创建web应用
    app = web.Application()
    app.router.add_get('/health', lb.handle_health)
    app.router.add_post('/score', lb.handle_score)
    
    # 启动服务器
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', args.listen_port)
    
    logger.info(f"Load balancer listening on port {args.listen_port}")
    logger.info(f"Backend servers: {args.base_port} to {args.base_port + args.num_servers - 1}")
    
    await site.start()
    
    # 保持运行
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await lb.stop()
        await runner.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
