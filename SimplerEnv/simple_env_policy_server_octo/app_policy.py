import os
import sys
import time
import base64
import uuid
import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, "/root/autodl-tmp/code/attackVLA/pi0_reward_server/SimplerEnv")


def _decode_image_nd(payload: dict) -> np.ndarray:
    b = base64.b64decode(payload["__numpy__"])
    dt = np.dtype(payload["dtype"])
    shape = tuple(payload["shape"])
    arr = np.frombuffer(b, dt)
    arr = arr.reshape(shape)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def create_app():
    app = Flask(__name__)
    sessions = {}

    @app.get("/health")
    def health():
        return "ok", 200

    @app.post("/reset")
    def reset():
        body = request.get_json(force=True) or {}
        session_id = body.get("session_id") or str(uuid.uuid4())
        task_description = body.get("task_description") or ""
        policy_setup = body.get("policy_setup") or "google_robot"
        action_scale = float(body.get("action_scale") or 1.0)
        model_type = body.get("model_type") or "octo-base"
        model = sessions.get(session_id)
        if model is None:
            from simpler_env.policies.octo.octo_model import OctoInference
            model = OctoInference(model_type=model_type, policy_setup=policy_setup, action_scale=action_scale)
            sessions[session_id] = model
        model.reset(task_description)
        return jsonify({"session_id": session_id}), 200

    @app.post("/infer")
    def infer():
        body = request.get_json(force=True) or {}
        session_id = body.get("session_id")
        if not session_id:
            return jsonify({"error": "missing session_id"}), 400
        model = sessions.get(session_id)
        if model is None:
            return jsonify({"error": "unknown session_id"}), 400
        img_nd = body.get("image_nd")
        if img_nd is None:
            return jsonify({"error": "missing image_nd"}), 400
        task_description = body.get("task_description")
        img = _decode_image_nd(img_nd)
        raw_action, action = model.step(img, task_description)
        return jsonify({"raw_action": {
            "world_vector": raw_action["world_vector"].tolist(),
            "rotation_delta": raw_action["rotation_delta"].tolist(),
            "open_gripper": raw_action["open_gripper"].tolist(),
        }, "action": {
            "world_vector": action["world_vector"].tolist(),
            "rot_axangle": action["rot_axangle"].tolist(),
            "gripper": np.asarray(action["gripper"]).tolist(),
            "terminate_episode": np.asarray(action["terminate_episode"]).tolist(),
        }}), 200

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8001))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
