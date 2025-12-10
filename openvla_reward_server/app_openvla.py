import logging
import os
import sys
import traceback
from flask import Flask, jsonify, request

from .reward_core import DEFAULT_LIBERO_CFG, compute_score
from .gpu_worker_pool import get_global_pool

# Configure logging to stderr so callers can redirect to files.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
    force=True,
)

logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)

    # Allow overriding the downstream policy port via env.
    policy_port = os.environ.get("POLICY_PORT")
    if policy_port:
        DEFAULT_LIBERO_CFG["port"] = int(policy_port)
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

            responses = body.get("responses", [])
            metas = body.get("metas", None)
            kwargs = body.get("reward_function_kwargs", {}) or {}

            logger.info(f"[OpenVLA-OFT] Received /score request with {len(responses)} samples")

            if not isinstance(responses, list):
                return jsonify({"error": "`responses` must be a list"}), 400
            if len(responses) == 0:
                return jsonify({"error": "`responses` must be a list"}), 400

            # 填默认 libero_cfg
            kwargs = dict(kwargs)
            lib_cfg = dict(DEFAULT_LIBERO_CFG)
            lib_cfg.update(kwargs.get("libero_cfg", {}) or {})
            kwargs["libero_cfg"] = lib_cfg

            # 检查是否启用并行处理
            use_parallel = os.environ.get("USE_GPU_POOL", "0") == "1"
            
            if use_parallel and len(responses) > 1:
                # 使用GPU工作池并行处理batch
                logger.info(f"[OpenVLA-OFT] [Parallel Mode] Processing {len(responses)} samples across multiple GPUs")
                pool = get_global_pool()
                done_result = pool.process_batch(responses, metas, **kwargs)
            else:
                # 单GPU顺序处理（原有逻辑）
                logger.info(f"[OpenVLA-OFT] [Sequential Mode] Processing {len(responses)} samples on single GPU")
                done_result = compute_score(responses, metas, **kwargs)

            logger.info(f"[OpenVLA-OFT] Completed {len(responses)} samples, results: {done_result}")

            return jsonify({"done_result": done_result}), 200

        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e), "traceback": traceback.format_exc(limit=2)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 6100))
    print(f"Starting OpenVLA-OFT reward server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


