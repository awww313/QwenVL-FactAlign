# QwenVL-FactAlign

> 基于 DPO 偏好对齐缓解 Qwen2.5-VL 多模态大模型幻觉

通过 Direct Preference Optimization (DPO) 对 Qwen2.5-VL-3B-Instruct 进行偏好对齐微调，针对性抑制多模态大模型在图文理解与知识问答场景中的**虚假生成、无依据脑补、图文错配**等幻觉问题。

---

## 📋 项目概览

| 项目 | 说明 |
|------|------|
| **基座模型** | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| **微调框架** | [Llama-Factory](https://github.com/hiyouga/LlamaFactory) (DPO stage) |
| **微调算法** | Direct Preference Optimization (DPO) |
| **微调方式** | QLoRA 4-bit NF4 + LoRA (rank=16, alpha=32) |
| **冻结策略** | 冻结视觉塔 + 多模态投影仪，仅训练 LLM 部分 |
| **训练数据** | 图文配对偏好数据集（chosen/rejected 对比样本） |
| **核心目标** | 抑制 MLLM 幻觉，提升图文理解的忠实度 |

## 🏗️ 项目结构

```
├── train_dpo_qwen2_5_vl.yaml    # DPO 训练配置
├── data/
│   ├── dataset_info.json        # 数据集注册信息
│   ├── qwen2.5-vl-3b.json       # 主训练数据集（偏好对比）
│   ├── qwen2.5-vl-3b_2.json     # 扩展训练数据集
│   └── qwen2.5-vl-3b_debug.json # 调试用数据集（小批量）
├── saves/qwen2.5-vl-3b-lora-dpo/ # 训练产出
│   ├── adapter_model.safetensors # 最终 LoRA 权重 ⬅️ LFS
│   ├── adapter_config.json       # LoRA 配置
│   ├── checkpoint-100/           # 第 100 步检查点
│   ├── checkpoint-200/           # 第 200 步检查点
│   ├── all_results.json          # 训练结果汇总
│   ├── training_loss.png         # Loss 曲线
│   ├── training_rewards_accuracies.png # Reward/Accuracy 曲线
│   ├── trainer_log.jsonl         # 训练日志
│   └── ...
├── requirements.txt              # 环境依赖
├── check_versions.py             # 版本兼容性检查
└── run_check.bat                 # 一键检查脚本
```

## ⚙️ 训练配置

### 核心参数

| 参数 | 值 | 说明 |
|------|------|------|
| `stage` | `dpo` | DPO 偏好对齐 |
| `finetuning_type` | `lora` | LoRA 微调 |
| `quantization_bit` | `4` | QLoRA 4-bit 量化 |
| `quantization_method` | `bitsandbytes` | bitsandbytes NF4 |
| `lora_rank` / `lora_alpha` | `16` / `32` | LoRA 秩与缩放 |
| `lora_target` | `all` | 全模块 LoRA（含 q/k/v/o/mlp） |
| `pref_beta` | `0.2` | DPO β 参数（控制对偏好对的依赖程度） |
| `pref_loss` | `sigmoid` | DPO 原版 sigmoid 损失 |
| `freeze_vision_tower` | `true` | 冻结视觉编码器 |
| `freeze_multi_modal_projector` | `true` | 冻结多模态投影仪 |
| `learning_rate` | `4e-6` | 学习率 |
| `num_train_epochs` | `2.0` | 训练轮数 |
| `per_device_batch_size` | `1` | 每设备 batch size |
| `gradient_accumulation_steps` | `16` | 梯度累积步数 |
| `image_max_pixels` | `50176` | 最大图像像素（224×224） |
| `flash_attn` | `auto` | Flash Attention 加速 |
| `bf16` | `true` | BF16 混合精度训练 |

### 训练结果

| 指标 | 数值 |
|------|------|
| 训练 Loss | 0.321 |
| 训练时长 | ~25 分钟（1500 秒） |
| 训练步数 | 200 步（2 epoch） |
| 有效 Batch Size | 16 |
| 吞吐量 | 2.13 samples/s |

## 🚀 快速开始

### 环境准备

```bash
# 克隆此仓库
git clone https://github.com/awww313/QwenVL-FactAlign.git
cd QwenVL-FactAlign

# 拉取 LFS 权重文件
git lfs pull

# 安装依赖
pip install -r requirements.txt
```

### 检查版本兼容性

```bash
python check_versions.py
```

### 开始训练

```bash
llamafactory-cli train train_dpo_qwen2_5_vl.yaml
```

### 推理测试

加载微调后的 LoRA 权重进行推理：

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

# 加载基座模型
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(
    base_model,
    "saves/qwen2.5-vl-3b-lora-dpo",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
```

## 🎯 幻觉抑制策略

本项目的关键在于 **偏好数据集的构造** 与 **DPO 超参调优**：

### 偏好数据构造
- **chosen 样本**：严格基于图像事实的描述，只描述可观察到的物体、空间关系、颜色、纹理和场景上下文
- **rejected 样本**：包含常见幻觉类型 — 错误属性推断、虚假对象声称、过度解读、位置关系错乱

### DPO 超参选择
- **`pref_beta=0.2`**：在保持生成多样性的同时强化对幻觉样本的压制。较大 β 会使模型更严格地遵循偏好信号
- **`pref_loss=sigmoid`**：DPO 原版 sigmoid 损失，不需要 reference model，训练更轻量

### 训练策略
- **冻结视觉塔 + 投影仪**：保留预训练的视觉编码能力，仅通过 LoRA 调整语言头，避免灾难性遗忘
- **LoRA rank=16**：适中的低秩适配，足够捕获偏好差异而不引入过多参数量
- **统一图像尺寸**：`image_max_pixels=50176`，保证所有图像以 224×224 输入，减少视觉编码不一致

## 📊 训练产出说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `adapter_model.safetensors` | ~115MB | **最终 LoRA 权重**，可直接加载推理 |
| `checkpoint-100/adapter_model.safetensors` | ~115MB | 第 100 步检查点权重 |
| `checkpoint-200/adapter_model.safetensors` | ~115MB | 第 200 步检查点权重 |
| `training_loss.png` | — | 训练 Loss 曲线 |
| `training_rewards_accuracies.png` | — | DPO reward 和 accuracy 曲线 |
| `trainer_log.jsonl` | — | 详细训练日志（loss / rewards / accuracies 逐步记录） |
| `all_results.json` | — | 训练结果汇总 |

> ⚠️ 权重文件使用 **Git LFS** 管理。克隆仓库后需要执行 `git lfs pull` 才能获取完整权重。

## 🧪 评估建议

微调后推荐使用以下方式评估幻觉抑制效果：

| 评估方法 | 说明 |
|----------|------|
| **对比测试** | 用同一张图片 + 同一问题，对比微调前后 Qwen2.5-VL 的回答差异 |
| **反事实提问** | 提问图片中不存在的内容，检查模型是否会编造 |
| **否定测试** | 故意提与图片矛盾的问题，看模型能否正确拒绝/纠正 |
| **CHAIR 指标** | Caption Hallucination Assessment — 评估描述中的物体幻觉率 |
| **POPE 指标** | Polling-based Object Probing Evaluation — 检测模型对物体存在的判断准确性 |
| **MMHal-Bench** | 多模态幻觉评估 benchmark，覆盖多种幻觉类型 |

## 📜 环境依赖

- Python 3.10+
- PyTorch ≥ 2.0
- transformers ≥ 4.49
- Llama-Factory（当前版本）
- peft / bitsandbytes / accelerate / trl
- flash-attn（可选，推荐）

## 🤝 致谢

- 微调框架：[Llama-Factory](https://github.com/hiyouga/LlamaFactory)
- 基座模型：[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) by 阿里通义千问团队
- DPO 算法：[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) by Stanford NLP
