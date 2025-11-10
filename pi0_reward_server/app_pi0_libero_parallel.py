"""
并行版本的Flask应用 - 支持多GPU并发处理请求
"""
import sys
import os
import logging
from flask import Flask, request, jsonify
import traceback
from reward_core_parallel import compute_score, DEFAULT_LIBERO_CFG
from client_pool import initialize_global_pool, cleanup_global_pool
from config import config
import atexit


def create_app(gpu_ports=None):
    """
    创建Flask应用
    
    Args:
        gpu_ports: GPU端口列表，例如 [8000, 8001] 表示两个GPU上的模型服务
                  如果为None，从配置文件读取
    """
    app = Flask(__name__)
    
    # 解析GPU端口配置
    if gpu_ports is None:
        hosts_ports = config.get_hosts_ports()
    else:
        host = config.GPU_HOST
        hosts_ports = [(host, port) for port in gpu_ports]
    
    # 初始化客户端连接池
    logging.info(f"Initializing client pool with hosts_ports: {hosts_ports}")
    initialize_global_pool(hosts_ports)
    
    # 注册清理函数
    atexit.register(cleanup_global_pool)
    
    @app.get("/health")
    def health():
        """健康检查端点"""
        return "ok", 200
    
    @app.get("/pool_status")
    def pool_status():
        """查看连接池状态"""
        from client_pool import get_global_pool
        pool = get_global_pool()
        if pool:
            return jsonify({
                "total_clients": pool.total_clients,
                "available_clients": pool.pool.qsize(),
                "hosts_ports": [(info['host'], info['port']) for info in [pool.pool.queue[i] for i in range(pool.pool.qsize())]] if pool.pool.qsize() > 0 else []
            }), 200
        else:
            return jsonify({"error": "Pool not initialized"}), 500

    @app.post("/score")
    def score():
        """
        计算评分端点
        
        Request JSON:
        {
          "responses": [ <vLLM对象1>, <vLLM对象2>, ... ],
          "metas": [
            {
              "original_instruction": "put the red bowl on the left shelf",
              "suite": "libero_object",
              "task_id": 3,
              "seed": 0,
              "init_state_id": 0
            },
            ...
          ],
          "reward_function_kwargs": {
            "alpha": 1.0, "beta": 0.1, "gamma": 0.5,
            "num_trials_per_task": 1, "center_crop": 1,
            "libero_cfg": {
              "model_family": "openvla",
              "task_suite_name": "libero_spatial",
              "task_id": 0,
              "checkpoint": "/path/to/openvla.ckpt"
            },
            "max_workers": 2  # 可选：最大并行worker数
          }
        }
        """
        try:
            body = request.get_json(force=True) or {}

            responses = body.get("responses", [])
            metas = body.get("metas", None)
            kwargs = body.get("reward_function_kwargs", {}) or {}

            if not isinstance(responses, list):
                return jsonify({"error": "`responses` must be a list"}), 400
            if len(responses) == 0:
                return jsonify({"error": "`responses` cannot be empty"}), 400

            # 填默认 libero_cfg
            kwargs = dict(kwargs)
            lib_cfg = dict(DEFAULT_LIBERO_CFG)
            lib_cfg.update(kwargs.get("libero_cfg", {}) or {})
            kwargs["libero_cfg"] = lib_cfg
            
            # 提取max_workers参数（如果有）
            max_workers = kwargs.pop("max_workers", None)
            if max_workers is not None:
                max_workers = int(max_workers)

            # 并行计算评分
            logging.info(f"Processing {len(responses)} requests with max_workers={max_workers}")
            done_result = compute_score(responses, metas, max_workers=max_workers, **kwargs)

            logging.info(f"[Results: {done_result}]")

            return jsonify({"done_result": done_result}), 200

        except Exception as e:
            logging.error(f"Error in /score endpoint: {e}")
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc(limit=5)
            }), 500

    return app


if __name__ == "__main__":
    # 配置日志
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 打印配置信息
    config.print_config()
    
    # 验证配置
    errors = config.validate()
    if errors:
        logging.error("配置验证失败:")
        for error in errors:
            logging.error(f"  - {error}")
        sys.exit(1)
    
    app = create_app()
    
    # 开发模式：使用Flask内置服务器（不推荐生产环境）
    # 注意：Flask的debug模式会导致代码重载，可能会创建多个客户端池
    app.run(host="0.0.0.0", port=config.PORT, debug=config.FLASK_DEBUG, threaded=True)

