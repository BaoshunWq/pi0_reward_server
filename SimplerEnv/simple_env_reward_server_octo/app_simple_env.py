import logging
import os
import sys
import traceback
from flask import Flask, jsonify, request

from .reward_core import DEFAULT_SIMPLE_ENV_CFG, compute_score


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
    force=True,
)

logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)
    policy_port = os.environ.get("POLICY_PORT")
    if policy_port:
        DEFAULT_SIMPLE_ENV_CFG["port"] = int(policy_port)
    @app.get("/health")
    def health():
        return "ok", 200
    @app.post("/score")
    def score():
        try:
            body = request.get_json(force=True) or {}
            responses = body.get("responses", [])
            metas = body.get("metas", None)
            kwargs = body.get("reward_function_kwargs", {}) or {}
            kwargs = dict(kwargs)
            cfg = dict(DEFAULT_SIMPLE_ENV_CFG)
            cfg.update(kwargs.get("simple_env_cfg", {}) or {})
            kwargs["simple_env_cfg"] = cfg
            if not isinstance(responses, list) or len(responses) == 0:
                return jsonify({"error": "`responses` must be a non-empty list"}), 400
            done_result = compute_score(responses, metas, **kwargs)
            return jsonify({"done_result": done_result}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc(limit=2)}), 500
    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 6101))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
