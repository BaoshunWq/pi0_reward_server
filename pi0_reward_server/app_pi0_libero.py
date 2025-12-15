import sys
import os
import logging
from flask import Flask, request, jsonify
import traceback
from .gpu_worker_pool import get_global_pool

# 配置日志 - 确保输出到stderr（会被重定向到日志文件）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,  # 只输出到stderr
    force=True  # 强制重新配置
)

logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)
    
    # 从环境变量读取policy端口配置（默认使用普通版本的配置）
    policy_port = os.environ.get("POLICY_PORT")
    if policy_port:
        logger.info(f"[Config] Using policy port from env: {policy_port}")
        print(f"[Config] Using policy port from env: {policy_port}", flush=True)

    @app.get("/health")
    def health():
        return "ok", 200

    @app.post("/score")
    def score():
        """
        Request JSON（只有 responses，可选 metas）:
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
            }
          }
        }
        """
        try:
            body = request.get_json(force=True) or {}

            # import pdb
            # pdb.set_trace()
            responses = body.get("responses", [])

            metas     = body.get("metas", None)
            kwargs    = body.get("reward_function_kwargs", {}) or {}
            
            logger.info(f"Received /score request with {len(responses)} samples")

            if not isinstance(responses, list):
                return jsonify({"error": "`responses` must be a list"}), 400
            if len(responses) == 0:
                return jsonify({"error": "`responses` must be a list"}), 400

            # 检查是否使用 danger 版本
            use_danger = kwargs.get("danger", False) or os.environ.get("USE_DANGER", "0") == "1"
            if use_danger:
                logger.info("[Mode] Using DANGER version (with collision detection)")
                os.environ["LIBERO_CONFIG_PATH"] = "vla_simulator_env/liberoDanger/libero/configs"
                from . import reward_core_danger as _rc
                
                DEFAULT_LIBERO_CFG = _rc.DEFAULT_LIBERO_CFG
                if policy_port:
                    DEFAULT_LIBERO_CFG["port"] = int(policy_port)
                compute_score = _rc.compute_score
            else:
                logger.info("[Mode] Using NORMAL version")
                os.environ["LIBERO_CONFIG_PATH"] = "openpi/third_party/libero/libero/configs"
                from . import reward_core as _rc
                
                DEFAULT_LIBERO_CFG = _rc.DEFAULT_LIBERO_CFG
                if policy_port:
                    DEFAULT_LIBERO_CFG["port"] = int(policy_port)
                compute_score = _rc.compute_score

            # 填默认 libero_cfg
            kwargs = dict(kwargs)
            lib_cfg = dict(DEFAULT_LIBERO_CFG)
            lib_cfg.update(kwargs.get("libero_cfg", {}) or {})
            kwargs["libero_cfg"] = lib_cfg

            # 检查是否启用并行处理
            use_parallel = os.environ.get("USE_GPU_POOL", "0") == "1"
            
            if use_parallel and len(responses) > 1:
                # 使用GPU工作池并行处理batch
                logger.info(f"[Parallel Mode] Processing {len(responses)} samples across multiple GPUs")
                # 将 danger 标志传递给 worker pool
                kwargs["_use_danger"] = use_danger
                pool = get_global_pool()
                done_result = pool.process_batch(responses, metas, **kwargs)
            else:
                # 单GPU顺序处理（原有逻辑）
                logger.info(f"[Sequential Mode] Processing {len(responses)} samples on single GPU")
                result = compute_score(responses, metas, **kwargs)
                
                # 处理返回值差异：danger版本返回 (success_list, collision_list)，普通版本只返回 success_list
                if use_danger and isinstance(result, tuple) and len(result) == 2:
                    success_list, collision_list = result
                    # 返回格式：每个元素是 [success_rate, collision_count]
                    done_result = [[sr, cc] for sr, cc in zip(success_list, collision_list)]
                else:
                    done_result = result

            logger.info(f"Completed {len(responses)} samples, results: {done_result}")

            return jsonify({"done_result": done_result}), 200

        except Exception as e:
            return jsonify({
                "error": str(e),
                "traceback": traceback.format_exc(limit=2)
            }), 500

    return app

if __name__ == "__main__":
    app = create_app()
    # 从环境变量读取端口，默认8000
    port = int(os.environ.get("PORT", 8000))
    # 使用threaded=True允许并发处理多个请求
    # 注意：对于CPU密集型任务，线程并发受GIL限制，建议使用waitress或gunicorn
    print(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
