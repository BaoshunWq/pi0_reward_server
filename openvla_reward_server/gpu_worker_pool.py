"""
GPU工作池：管理多个GPU worker并行处理任务
使用进程池 + 队列实现跨GPU的并行计算
适配 OpenVLA-OFT 模型
"""
import os
import multiprocessing as mp
from multiprocessing import Queue, Process, Semaphore
from typing import List, Dict, Any, Optional
import traceback
import time

# 注意：EGL上下文警告是AttributeError异常，不是Warning
# 无法通过warnings模块过滤，这些警告可以安全忽略

# 全局信号量：限制同时创建LIBERO环境的worker数量
# 这可以避免EGL上下文资源耗尽
MAX_CONCURRENT_ENVS = 1  # 一次只允许一个worker创建环境


def gpu_worker(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    worker_id: int,
    policy_port: int = None
):
    """
    GPU工作进程：从任务队列取任务，在指定GPU上执行，结果放入结果队列
    
    Args:
        gpu_id: GPU设备ID
        task_queue: 任务队列 (task_id, response, meta, kwargs)
        result_queue: 结果队列 (task_id, result, error)
        worker_id: 工作进程ID
        policy_port: Policy服务器端口（可选）
    """
    # 设置当前进程只使用指定的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 设置EGL设备（用于robosuite离屏渲染）
    # 这确保每个worker只在指定的GPU上创建EGL上下文
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    
    # 延迟导入，确保在设置环境变量后导入
    # 注意：这里导入 openvla_reward_server 的 compute_score
    from openvla_reward_server.reward_core import compute_score
    
    print(f"[Worker-{worker_id}] Started on GPU {gpu_id}, policy_port={policy_port}")
    print(f"[Worker-{worker_id}] EGL_DEVICE_ID={os.environ.get('EGL_DEVICE_ID')}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    
    # 环境缓存：避免重复创建EGL上下文
    # 注意：这是一个简单的缓存，实际使用中可能需要更复杂的管理
    env_cache = {}
    
    while True:
        try:
            # 从队列获取任务，超时5秒
            task = task_queue.get(timeout=5)
            
            # 毒丸信号：退出
            if task is None:
                print(f"[Worker-{worker_id}] Received stop signal, exiting...")
                break
            
            task_id, response, meta, kwargs = task
            print(f"[Worker-{worker_id}] Processing task {task_id} on GPU {gpu_id}")
            
            start_time = time.time()
            
            # 执行计算
            try:
                # 如果指定了policy_port，覆盖kwargs中的port
                if policy_port is not None:
                    kwargs = dict(kwargs)
                    if "libero_cfg" not in kwargs:
                        kwargs["libero_cfg"] = {}
                    kwargs["libero_cfg"]["port"] = policy_port
                
                # compute_score期望列表输入
                result = compute_score(
                    responses=[response],
                    metas=[meta] if meta else None,
                    **kwargs
                )
                # 取第一个结果（因为我们只传了一个样本）
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                elapsed = time.time() - start_time
                print(f"[Worker-{worker_id}] Task {task_id} completed in {elapsed:.2f}s")
                
                result_queue.put((task_id, result, None))
                
            except Exception as e:
                error_msg = f"Error in task {task_id}: {str(e)}\n{traceback.format_exc()}"
                print(f"[Worker-{worker_id}] {error_msg}")
                result_queue.put((task_id, None, error_msg))
        
        except mp.queues.Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            print(f"[Worker-{worker_id}] Unexpected error: {e}\n{traceback.format_exc()}")
            continue
    
    print(f"[Worker-{worker_id}] Stopped")


class GPUWorkerPool:
    """GPU工作池：管理多个GPU worker"""
    
    def __init__(self, gpu_ids: List[int], base_policy_port: int = None):
        """
        Args:
            gpu_ids: 可用的GPU ID列表，例如 [0, 1, 2, 3]
            base_policy_port: Policy服务器基础端口，每个worker使用 base_port + worker_index
                            如果为None，从环境变量或kwargs中获取
        """
        self.gpu_ids = gpu_ids
        self.num_workers = len(gpu_ids)
        self.base_policy_port = base_policy_port
        self.task_queue = Queue(maxsize=1000)  # 任务队列
        self.result_queue = Queue(maxsize=1000)  # 结果队列
        self.workers: List[Process] = []
        self.is_started = False
    
    def start(self):
        """启动所有worker进程"""
        if self.is_started:
            return
        
        print(f"[GPUWorkerPool] Starting {self.num_workers} workers on GPUs {self.gpu_ids}")
        if self.base_policy_port:
            print(f"[GPUWorkerPool] Policy servers: ports {self.base_policy_port} to {self.base_policy_port + self.num_workers - 1}")
        
        for i, gpu_id in enumerate(self.gpu_ids):
            # 计算该worker对应的policy端口
            policy_port = self.base_policy_port + i if self.base_policy_port else None
            
            worker = Process(
                target=gpu_worker,
                args=(gpu_id, self.task_queue, self.result_queue, i, policy_port),
                daemon=False  # 不使用daemon，确保优雅退出
            )
            worker.start()
            self.workers.append(worker)
        
        self.is_started = True
        print(f"[GPUWorkerPool] All workers started")
    
    def process_batch(
        self,
        responses: List[Dict],
        metas: Optional[List[Dict]],
        **kwargs
    ) -> List[Any]:
        """
        并行处理一个batch的样本
        
        Args:
            responses: 响应列表
            metas: 元数据列表（可选）
            **kwargs: 传递给compute_score的其他参数
        
        Returns:
            结果列表，顺序与输入一致
        """
        if not self.is_started:
            raise RuntimeError("Worker pool not started. Call start() first.")
        
        num_samples = len(responses)
        print(f"[GPUWorkerPool] Processing batch of {num_samples} samples")
        
        # 1. 将所有任务放入队列
        for i, response in enumerate(responses):
            meta = metas[i] if metas and i < len(metas) else None
            self.task_queue.put((i, response, meta, kwargs))
        
        # 2. 收集结果
        results = [None] * num_samples
        errors = []
        
        for _ in range(num_samples):
            try:
                task_id, result, error = self.result_queue.get(timeout=600)  # 10分钟超时
                
                if error:
                    errors.append(f"Task {task_id}: {error}")
                    results[task_id] = 0.0  # 错误时返回默认值
                else:
                    results[task_id] = result
            
            except mp.queues.Empty:
                raise TimeoutError(f"Timeout waiting for results. Processed {len([r for r in results if r is not None])}/{num_samples}")
        
        if errors:
            print(f"[GPUWorkerPool] Completed with {len(errors)} errors:")
            for err in errors[:5]:  # 只打印前5个错误
                print(f"  {err}")
        
        print(f"[GPUWorkerPool] Batch processing completed")
        return results
    
    def stop(self):
        """停止所有worker"""
        if not self.is_started:
            return
        
        print(f"[GPUWorkerPool] Stopping workers...")
        
        # 发送停止信号
        for _ in self.workers:
            self.task_queue.put(None)
        
        # 等待所有worker退出
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                print(f"[GPUWorkerPool] Force terminating worker {worker.pid}")
                worker.terminate()
        
        self.workers.clear()
        self.is_started = False
        print(f"[GPUWorkerPool] All workers stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# 全局worker pool实例（单例模式）
_global_pool: Optional[GPUWorkerPool] = None


def get_global_pool(gpu_ids: Optional[List[int]] = None, base_policy_port: Optional[int] = None) -> GPUWorkerPool:
    """
    获取全局GPU工作池（单例）
    
    Args:
        gpu_ids: GPU ID列表，仅在首次调用时有效
        base_policy_port: Policy服务器基础端口，仅在首次调用时有效
    """
    global _global_pool
    
    if _global_pool is None:
        if gpu_ids is None:
            # 从环境变量读取
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        
        if base_policy_port is None:
            # 从环境变量读取
            base_policy_port_str = os.environ.get("BASE_POLICY_PORT")
            if base_policy_port_str:
                base_policy_port = int(base_policy_port_str)
        
        _global_pool = GPUWorkerPool(gpu_ids, base_policy_port)
        _global_pool.start()
    
    return _global_pool


def shutdown_global_pool():
    """关闭全局worker pool"""
    global _global_pool
    if _global_pool:
        _global_pool.stop()
        _global_pool = None

