# 启动指南（中文）

本文帮助你在本地或远程环境中快速运行 OpenVLA-OFT。若需更多细节，请结合 `README.md`、`SETUP.md` 等英文文档阅读。

## 1. 环境准备

1. **硬件需求**
   - 推理：至少 1 块 16 GB VRAM（LIBERO），或 18 GB VRAM（ALOHA）。
   - 训练：1–8 块 27–80 GB VRAM 的 GPU（默认 bfloat16）。更多说明见官网 FAQ。
2. **软件依赖**
   - 建议使用 Conda，并按 `SETUP.md` 完成环境创建与依赖安装。
   - 安装好 `pi0_reward_server` 所需的系统依赖（如 CUDA 驱动、FFmpeg 等）。
3. **代码获取**
   - 克隆本仓库并切换到 `openvla-oft` 目录：  
     ```bash
     git clone <repo_url>
     cd pi0_reward_server/openvla-oft
     ```

## 2. 快速验证推理流程

1. 激活 Conda 环境（假设名为 `openvla-oft`）：  
   ```bash
   conda activate openvla-oft
   ```
2. 运行示例脚本下载模型并生成动作片段：
   ```bash
   python - <<'PY'
   import pickle
   from experiments.robot.libero.run_libero_eval import GenerateConfig
   from experiments.robot.openvla_utils import (
       get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
   )
   from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

   cfg = GenerateConfig(
       pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
       use_l1_regression=True,
       use_diffusion=False,
       use_film=False,
       num_images_in_input=2,
       use_proprio=True,
       load_in_8bit=False,
       load_in_4bit=False,
       center_crop=True,
       num_open_loop_steps=NUM_ACTIONS_CHUNK,
       unnorm_key="libero_spatial_no_noops",
   )

   vla = get_vla(cfg)
   processor = get_processor(cfg)
   action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
   proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

   with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as f:
       observation = pickle.load(f)

   actions = get_vla_action(
       cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector
   )
   for act in actions:
       print(act)
   PY
   ```
3. 若终端打印出动作序列，即表示模型加载与前向推理成功。

## 3. 远程服务工作流（OpenPI 风格）

当需要由多个评估节点共享同一策略时，可参考以下流程：

1. **启动策略服务器（加载一次模型）**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python openvla-oft/scripts/serve_policy.py \
     --pretrained-checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
     --policy-server-port 23451
   ```
   - 该脚本使用 `experiments/robot/libero/policy_server.py` 中的 websocket 服务。
2. **在评估端连接远程策略**
   ```bash
   python openvla-oft/experiments/robot/libero/run_libero_eval.py \
     --policy-mode remote \
     --policy-server-host 0.0.0.0 \
     --policy-server-port 23451 \
     --task-suite-name libero_spatial
   ```
   - 设置 `policy_mode=remote` 后，评估脚本会跳过本地 checkpoint 加载，直接通过 websocket 请求动作。
3. **对接奖励服务器或多实例评估**
   - 只需保证每个评估进程指向同一 host/port，即可实现多路复用。

## 4. 常用配置项速览

- `GenerateConfig`（位于 `experiments/robot/libero/run_libero_eval.py`）是统一的运行配置，核心字段包括：
  - `pretrained_checkpoint`：Hugging Face 或本地 checkpoint 标识。
  - `num_open_loop_steps`：每次推理生成的动作序列长度。
  - `policy_mode`：`local`（默认，加载本地模型）或 `remote`（走 websocket）。
  - `policy_server_host` / `policy_server_port`：远程模式下必填。
  - `task_suite_name`：LIBERO 任务集，如 `libero_spatial`。
  - 训练/推理混合精度、输入模态等开关（`use_l1_regression`、`use_diffusion`、`use_proprio` 等）。

在命令行中可直接覆盖这些参数（例如 `--task-suite-name libero_spatial`）。

## 5. 常见排查提示

1. **显存不足**：可尝试 `load_in_8bit=True` 或裁剪输入图像分辨率。
2. **Websocket 连接失败**：确认策略服务器端口开放、防火墙放行、以及 `policy_server_host` 是否填写服务器公网/内网地址。
3. **任务初始状态缺失**：若使用自定义 `initial_states_path`，需确保 JSON 中的 `success` 标志为真，否则评估脚本会跳过该 episode。
4. **日志与监控**：`run_libero_eval.py` 会在本地生成日志文件，并可通过 `--use-wandb` 将成功率等指标同步到 Weights & Biases。

## 6. 更多资料

- `README.md`：系统需求、远程流程、引用信息。
- `LIBERO.md` / `ALOHA.md`：分别涵盖仿真与真实机器人任务。
- `SETUP.md`：详细的依赖安装与环境配置步骤。

建议中文团队在初次集成时以本指南为骨架，并在验证成功后，根据自身集群/机器人平台需求对配置文件与脚本做进一步定制。

