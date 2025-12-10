# OpenVLA-OFT Reward Server

Self-contained reward server built from the existing `pi0_reward_server` logic and the parallel launch scripts. It speaks the same `/score` API but targets the OpenVLA-OFT policy server (default websocket port `23451`).

## Layout
- `app_openvla.py`: Flask app exposing `/health` and `/score`.
- `reward_core.py`: merges defaults and drives the LIBERO eval loop.
- `launch_waitress.sh`: convenience launcher (Waitress).

## Quickstart
1) Ensure the OpenVLA-OFT policy server is running, e.g.:
```bash
cd /root/autodl-tmp/code/attackVLA/pi0_reward_server/openvla-oft
python -m experiments.robot.libero.policy_server serve_policy --policy-server-port 23451
```

2) Start the reward server:
```bash
cd /root/autodl-tmp/code/attackVLA/pi0_reward_server
PORT=6100 POLICY_PORT=23451 bash openvla_reward_server/launch_waitress.sh
```

3) Send a request (same schema as `pi0_reward_server`):
```bash
curl -X POST http://localhost:6100/score \
  -H "Content-Type: application/json" \
  -d '{"responses":[{"generated_text":"pick up the bowl"}],"metas":[{"suite":"libero_spatial","task_id":0}],"reward_function_kwargs":{"num_trials_per_task":1}}'
```

## Notes
- Defaults mirror the LIBERO settings from `openvla-oft` (`NUM_ACTIONS_CHUNK=8`, policy port `23451`).
- `POLICY_PORT` env overrides the downstream websocket target; `PORT` controls the reward server port.
- The server reuses `pi0_reward_server.env_pertask` for the actual rollout, so no extra dependencies are added. All new files live in this folder as requested.


