# VERL Qwen 指令重写使用指南

本文档介绍如何使用 VERL 训练的 Qwen 语言模型来重写和改进机器人操作指令。

## 新增内容

### 1. 新类: `EmbodiedRedTeamModelWithVERLQwen`

位于 `generate_intruction.py` 中,这是一个专门用于 VERL 训练的 Qwen 模型的包装类。

**特点:**
- ✅ 纯文本输入,不需要图像
- ✅ 专注于指令重写和改进
- ✅ 支持 Qwen 的 chat 模板
- ✅ 自动生成多样化的指令变体
- ✅ 集成语义相似度筛选

### 2. 使用方法

#### 方法 1: 通过 `build_red_team_generator` 工厂函数

```python
from generate_intruction import build_red_team_generator, CLIPEmbeddingModel

# 1. 创建语义相似度模型
clip_embedder = CLIPEmbeddingModel(device="cuda")

# 2. 创建 VERL Qwen 生成器
verl_generator = build_red_team_generator(
    backend="verl_qwen",
    embedding_model=clip_embedder,
    model_path="/path/to/your/verl_trained_qwen_model",  # VERL 训练的模型路径
    device="cuda"
)

# 3. 生成指令变体
task = "pick up the red bowl and place it on the shelf"
selected_instructions, similarities = verl_generator(
    task=task,
    semantic_type="clip",
    num_instructions=10,
    select_topk=3
)

print("Selected instructions:")
for inst, sim in zip(selected_instructions, similarities):
    print(f"  - {inst} (similarity: {sim})")
```

#### 方法 2: 通过配置文件使用 (集成到现有流程)

在你的配置文件中添加:

```python
class Config:
    BACKEND = "verl_qwen"  # 选择 VERL Qwen 后端
    verl_model_path = "/path/to/your/verl_trained_qwen_model"
    semantic_type = "clip"
    # ... 其他配置
```

然后在代码中:

```python
from utils import load_relate_model

# 加载模型
red_team = load_relate_model(cfg)

# 生成指令
selected_instructions, similarities = red_team(
    task="pick up the red bowl",
    semantic_type=cfg.semantic_type,
    num_instructions=10,
    select_topk=3
)
```

#### 方法 3: 直接使用类

```python
from generate_intruction import EmbodiedRedTeamModelWithVERLQwen, CLIPEmbeddingModel

# 1. 创建语义模型
clip_embedder = CLIPEmbeddingModel(device="cuda")

# 2. 直接实例化
verl_model = EmbodiedRedTeamModelWithVERLQwen(
    embedding_model=clip_embedder,
    model_path="/path/to/your/verl_trained_qwen_model",
    device="cuda"
)

# 3. 使用
task = "put the blue cube in the drawer"
selected, sims = verl_model(
    task=task,
    semantic_type="clip",
    num_instructions=10,
    select_topk=5
)
```

### 3. 完整示例

```python
import torch
from generate_intruction import build_red_team_generator, CLIPEmbeddingModel

# 检查 CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. 加载 CLIP 嵌入模型
clip_embedder = CLIPEmbeddingModel(device=device)
print("[CLIP] Embedding model loaded.")

# 2. 创建 VERL Qwen 生成器
verl_qwen = build_red_team_generator(
    backend="verl_qwen",
    embedding_model=clip_embedder,
    model_path="/path/to/verl_qwen_model",
    device=device,
)

# 3. 原始任务
original_task = "pick up the red bowl and place it on the left shelf"
print(f"\nOriginal task: {original_task}")

# 4. 生成多个指令变体
print("\n--- Generating instruction variations ---")
selected_instructions, similarities = verl_qwen(
    task=original_task,
    semantic_type="clip",
    num_instructions=10,
    select_topk=5
)

print(f"\nTop {len(selected_instructions)} selected instructions:")
for i, (inst, sim) in enumerate(zip(selected_instructions, similarities), 1):
    print(f"{i}. [{sim:.4f}] {inst}")

# 5. 获取所有候选指令
print("\n--- Getting all candidates ---")
best_one, best_sim, all_candidates = verl_qwen(
    task=original_task,
    semantic_type="clip",
    num_instructions=10,
    return_all_annotations=True
)

print(f"\nBest instruction: {best_one}")
print(f"Similarity: {best_sim}")
print(f"\nAll {len(all_candidates)} candidates:")
for i, cand in enumerate(all_candidates, 1):
    print(f"  {i}. {cand}")
```

## 参数说明

### `EmbodiedRedTeamModelWithVERLQwen.__call__` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `task` | str | 必需 | 原始任务描述 |
| `semantic_type` | str | "clip" | 语义相似度类型 ("clip" 或其他) |
| `image_url` | Optional[str] | None | 图像URL (VERL模型不使用,仅保持接口兼容) |
| `num_instructions` | int | 10 | 生成的指令数量 |
| `return_all_annotations` | bool | False | 是否返回所有候选指令 |
| `select_topk` | int | 3 | 选择 top-k 个最相似的指令 |

### 返回值

- **如果 `return_all_annotations=False`:**
  - 返回: `(selected_instructions: List[str], similarities: List[float])`
  - `selected_instructions`: top-k 个选中的指令列表
  - `similarities`: 对应的相似度分数列表

- **如果 `return_all_annotations=True`:**
  - 返回: `(best_instruction: str, best_similarity: float, all_candidates: List[str])`
  - `best_instruction`: 相似度最高的指令
  - `best_similarity`: 最高相似度分数
  - `all_candidates`: 所有候选指令列表

## 与其他后端的比较

| 后端 | 输入 | 优势 | 使用场景 |
|------|------|------|----------|
| `smolvlm` | 文本 + 图像 | 轻量级,快速 | 需要视觉上下文的指令生成 |
| `qwenvl` | 文本 + 图像 | 强大的多模态理解 | 复杂的视觉-语言任务 |
| `verl_qwen` | 纯文本 | 专门训练,无需图像 | 纯指令重写和改进 |

## 配置示例

在 `vlmRewrite_main.py` 或类似脚本中使用:

```python
class Config:
    # 后端选择
    BACKEND = "verl_qwen"  # 'smolvlm', 'qwenvl', 'verl_qwen'
    
    # VERL Qwen 配置
    verl_model_path = "/path/to/verl_trained_qwen"
    
    # 语义相似度配置
    semantic_type = "clip"  # 'clip' 或其他
    
    # 生成参数
    num_instructions = 10
    select_topk = 3
```

## 注意事项

1. **模型路径**: 确保 `verl_model_path` 指向正确的 VERL 训练模型目录
2. **设备**: 如果有多个 GPU,可以通过 `device` 参数指定
3. **内存**: VERL Qwen 模型会占用显存,确保有足够的 GPU 内存
4. **图像参数**: 虽然 VERL Qwen 不使用图像,但保留了 `image_url` 参数以保持接口一致性

## 故障排查

### 问题 1: 找不到模型文件
```
FileNotFoundError: /path/to/verl_model not found
```
**解决**: 检查 `model_path` 是否正确,确保模型文件存在

### 问题 2: CUDA 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 
- 减少 `num_instructions` 参数
- 使用更小的模型
- 使用 CPU: `device="cpu"`

### 问题 3: 生成的指令质量不佳
**解决**:
- 调整 `temperature` 参数 (在 `_generate_instructions` 中)
- 增加 `num_instructions` 以获得更多候选
- 调整 `select_topk` 以选择更多/更少的结果

## 相关文件

- `generate_intruction.py`: 包含所有后端实现
- `utils.py`: 工具函数和模型加载
- `vlmRewrite_main.py`: 主执行脚本

