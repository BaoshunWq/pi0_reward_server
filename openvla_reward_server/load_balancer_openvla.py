#!/usr/bin/env python
"""
OpenVLA-OFT 专用的负载均衡器。

将 `/score` 请求按样本拆分后并发转发到多台 OpenVLA 奖励服务器，
使用最少连接数策略，遇到异常/错误直接抛出，避免用默认值静默返回。
"""
import argparse
import asyncio
import logging
import time
from typing import Dict, List

import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServerPool:
    """管理多个后端服务器的连接池（最少连接数策略）"""

    def __init__(self, base_port: int, num_servers: int):
        self.servers = [
            {
                "url": f"http://localhost:{base_port + i}",
                "active_requests": 0,
                "total_requests": 0,
                "errors": 0,
                "last_error_time": 0,
            }
            for i in range(num_servers)
        ]
        self.lock = asyncio.Lock()

    async def get_best_server(self) -> Dict:
        """选择当前负载最小的服务器；近期报错的节点会被暂时跳过。"""
        async with self.lock:
            current_time = time.time()
            available = [
                s for s in self.servers if current_time - s["last_error_time"] > 30
            ]
            if not available:
                available = self.servers

            best = min(available, key=lambda s: s["active_requests"])
            best["active_requests"] += 1
            best["total_requests"] += 1
            return best

    async def release_server(self, server: Dict, had_error: bool = False):
        """释放服务器并记录错误。"""
        async with self.lock:
            server["active_requests"] = max(0, server["active_requests"] - 1)
            if had_error:
                server["errors"] += 1
                server["last_error_time"] = time.time()

    def get_stats(self) -> List[Dict]:
        """获取所有服务器的统计信息。"""
        return [
            {
                "url": s["url"],
                "active": s["active_requests"],
                "total": s["total_requests"],
                "errors": s["errors"],
            }
            for s in self.servers
        ]


class LoadBalancer:
    def __init__(self, base_port: int, num_servers: int):
        self.pool = ServerPool(base_port, num_servers)
        self.session: aiohttp.ClientSession | None = None

    async def start(self):
        """启动负载均衡器"""
        timeout = aiohttp.ClientTimeout(total=600)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info(
            "[OpenVLA-OFT LB] started with %s backend servers", len(self.pool.servers)
        )

    async def stop(self):
        """停止负载均衡器"""
        if self.session:
            await self.session.close()

    async def handle_health(self, request: web.Request) -> web.Response:
        """健康检查端点"""
        stats = self.pool.get_stats()
        return web.json_response({"status": "healthy", "servers": stats})

    async def handle_score(self, request: web.Request) -> web.Response:
        """
        按样本级别分发 batch 请求到后端服务器：
        1) 拆分 batch 为单样本
        2) 并发发送
        3) 收集结果并合并返回（异常直接抛出，不使用默认值兜底）
        """
        try:
            body = await request.json()

            responses = body.get("responses", [])
            metas = body.get("metas", [])
            reward_kwargs = body.get("reward_function_kwargs", {})

            num_samples = len(responses)
            logger.info(
                "[OpenVLA-OFT LB] Received batch with %s samples, splitting...",
                num_samples,
            )

            if num_samples == 0:
                return web.json_response({"error": "Empty batch"}, status=400)

            tasks = []
            for i in range(num_samples):
                if responses[i]["outputs"][0]["text"].endswith("."):
                    responses[i]["outputs"][0]["text"] = responses[i]["outputs"][0]["text"][:-1]
                single_request = {

                    "responses": [responses[i]],
                    "metas": [metas[i]] if i < len(metas) else [{}],
                    "reward_function_kwargs": reward_kwargs,
                }
                # import pdb
                # pdb.set_trace()
                logger.info(f"Single request: {single_request}")
                tasks.append(self._forward_single_sample(i, single_request))

            # 并发执行；若有异常将直接向上传播
            results = await asyncio.gather(*tasks)

            merged_done_result = []
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    raise TypeError(
                        f"[OpenVLA-OFT LB] Sample {i}: unexpected result type {type(result)}"
                    )
                if "error" in result:
                    raise RuntimeError(f"[OpenVLA-OFT LB] Sample {i}: {result['error']}")
                if "done_result" not in result:
                    raise KeyError(f"[OpenVLA-OFT LB] Sample {i}: missing done_result")

                done_list = result["done_result"]
                if not isinstance(done_list, list) or len(done_list) == 0:
                    raise ValueError(
                        f"[OpenVLA-OFT LB] Sample {i}: invalid done_result {done_list}"
                    )

                merged_done_result.append(done_list[0])

            response_data = {
                "done_result": merged_done_result,
                "num_samples": num_samples,
                "num_errors": 0,
            }

            logger.info("[OpenVLA-OFT LB] Batch completed: %s samples", num_samples)
            logger.info("[OpenVLA-OFT LB] Response data: %s", response_data)
            return web.json_response(response_data)

        except Exception as e:  # noqa: BLE001
            logger.error("[OpenVLA-OFT LB] Error processing batch: %s", e)
            import traceback

            logger.error(traceback.format_exc())
            return web.json_response(
                {"error": f"Batch processing error: {str(e)}"}, status=500
            )

    async def _forward_single_sample(
        self, sample_idx: int, request_data: dict
    ) -> Dict:
        """
        将单个样本转发到后端奖励服务器。
        """
        server = await self.pool.get_best_server()
        had_error = False

        try:
            target_url = f"{server['url']}/score"
            logger.info(
                "[OpenVLA-OFT LB] Sample %s -> %s (active: %s)",
                sample_idx,
                target_url,
                server["active_requests"],
            )

            async with self.session.post(
                target_url,
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(
                        "[OpenVLA-OFT LB] Sample %s completed on %s",
                        sample_idx,
                        server["url"],
                    )
                    return result

                error_text = await resp.text()
                logger.error(
                    "[OpenVLA-OFT LB] Sample %s failed: HTTP %s", sample_idx, resp.status
                )
                had_error = True
                return {"error": f"HTTP {resp.status}: {error_text}"}

        except asyncio.TimeoutError:
            logger.error("[OpenVLA-OFT LB] Sample %s timeout on %s", sample_idx, server["url"])
            had_error = True
            return {"error": "Timeout"}

        except aiohttp.ClientError as e:  # noqa: BLE001
            logger.error("[OpenVLA-OFT LB] Sample %s client error: %s", sample_idx, e)
            had_error = True
            return {"error": f"Client error: {str(e)}"}

        except Exception as e:  # noqa: BLE001
            logger.error("[OpenVLA-OFT LB] Sample %s unexpected error: %s", sample_idx, e)
            had_error = True
            return {"error": f"Unexpected error: {str(e)}"}

        finally:
            await self.pool.release_server(server, had_error)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load balancer for OpenVLA-OFT reward servers (Least Connections)"
    )
    parser.add_argument(
        "--listen_port",
        type=int,
        default=6100,  # 与 openvla-oft 启动脚本保持一致
        help="Port to listen on for incoming requests",
    )
    parser.add_argument(
        "--base_port",
        type=int,
        default=6101,  # 第一个后端 reward 端口
        help="Base port of backend reward servers",
    )
    parser.add_argument(
        "--num_servers",
        type=int,
        default=1,  # 与启动脚本默认 GPU 数一致
        help="Number of backend servers",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    lb = LoadBalancer(args.base_port, args.num_servers)
    await lb.start()

    app = web.Application()
    app.router.add_get("/health", lb.handle_health)
    app.router.add_post("/score", lb.handle_score)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.listen_port)

    logger.info(
        "[OpenVLA-OFT LB] listening on %s; backends %s-%s",
        args.listen_port,
        args.base_port,
        args.base_port + args.num_servers - 1,
    )

    await site.start()

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("[OpenVLA-OFT LB] Shutting down...")
    finally:
        await lb.stop()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

