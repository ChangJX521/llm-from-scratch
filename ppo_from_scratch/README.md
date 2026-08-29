# PPO from Scratch

这是一个面向学习与实验的 PPO-RLHF 实现。它不使用 TRL 的现成 PPO Trainer，而是直接用 PyTorch 和 Transformers 实现从文本生成、奖励计算、优势估计到 PPO 更新和独立测试的完整流程。

最终实验已经在独立测试集上得到正向结果：Reward 平均提升 `+0.3787`，胜率 `60.9%`。参见 [实验报告](EXPERIMENT_REPORT.md)。

## 1. 训练目标

给定用户问题 `x`，Actor 策略 `πθ` 生成回答 `y`，Reward Model 给出偏好分数 `r(x, y)`。PPO 的目标是在提高奖励的同时，通过冻结的 Reference Model `πref` 限制 Actor 偏离原模型的程度。

本项目中的单 token shaped reward 为：

```text
r_t = -β [log πθ(a_t|s_t) - log πref(a_t|s_t)]
```

Reward Model 的序列分数只添加到回答最后一个有效 token。随后使用 GAE 计算 advantage：

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ A_{t+1}
```

Actor 使用 PPO clipped objective：

```text
ratio_t = exp(log πθ - log πold)
Lpolicy = -mean(min(ratio_t A_t, clip(ratio_t, 1-ε, 1+ε) A_t))
```

Value Head 使用 clipped value loss。所有 loss 与统计量都通过 action mask 只作用于回答 token，不包含 prompt 和 padding。

## 2. 模型组成

```text
WebGPT prompt
     │
     ▼
Actor (Qwen3-0.6B) ───────► response
     │                         │
     │                         ▼
     │                  Reward Model
     │                         │
     ├──── log-prob ───────────┤
     │                         ▼
Reference ─ log-prob ─► KL-shaped reward
     │                         │
     └── hidden states ─► Value Head
                               │
                               ▼
                        GAE + PPO update
```

- **Actor**：`Qwen/Qwen3-0.6B`，参与梯度更新。
- **Reference**：Actor 的冻结初始副本，用于 KL 约束。
- **Reward Model**：`OpenAssistant/reward-model-deberta-v3-large-v2`，冻结并为问答对打分。
- **Value Head**：线性层，为每个 token 预测 state value。

## 3. 代码结构

| 文件 | 职责 |
| --- | --- |
| `config.py` | 模型、数据、PPO、验证、输出参数 |
| `models.py` | 模型加载、混合精度、Value Head、checkpoint |
| `rollout.py` | 数据集、生成、log-prob、Reward、KL、GAE |
| `trainer.py` | PPO 更新、KL 控制、验证选模、早停、最终测试 |
| `reporting.py` | CSV、TensorBoard、训练曲线与前后对比图 |
| `ppo_train.py` | 随机种子、组件装配和训练入口 |

## 4. 为什么当前版本能够稳定工作

早期实验暴露了两个关键问题：

1. 一次性为全部 rollout 序列计算完整词表 logits 会超过 24GB 显存。
2. 只限制每次 PPO update 的局部 KL，不能阻止 Actor 相对初始 Reference 的累计漂移。

当前版本采取以下措施：

- Actor 保存 FP32 主权重和 FP32 Adam 状态，前向/反向使用 BF16 autocast。
- Actor、Reference 的 rollout log-prob 每次只计算 2 条序列。
- PPO epoch 为 1，Actor 学习率为 `1e-6`。
- 每轮记录 Actor 相对 Reference 的 sampled KL。
- KL 超过目标时提高惩罚系数，超过安全阈值时停止更新。
- 384 条数据训练、64 条验证选模、最后 64 条只做测试。
- 每 4 轮验证一次，连续 4 次无提升就早停。
- 训练结束后恢复验证集表现最好的 Actor，而不是保存最后一轮。

## 5. 环境安装

成功实验环境：Python 3.11、PyTorch 2.12.1、Transformers 4.57.6、Datasets 5.0.1，单张 RTX 4090 24GB。

建议先创建独立环境，再安装依赖：

```bash
conda create -n llm python=3.11 -y
conda activate llm
pip install -r requirements.txt
```

如果需要特定 CUDA 版本，请先按照 PyTorch 对应平台的安装方式安装 `torch`，再安装其余依赖。

## 6. 数据集

默认从 Hugging Face 的 WebGPT Comparisons Parquet 文件加载数据，避免新版 `datasets` 不再支持 dataset script 的问题。

无法联网或已经下载数据时，可以覆盖成本地文件：

```bash
export WEBGPT_DATASET_FILE=/absolute/path/to/0000.parquet
```

注意：变量内容必须是纯路径或纯 URL，不能写成 Markdown 的 `[URL](URL)` 形式。

## 7. 启动训练

在仓库根目录运行：

```bash
python ppo_from_scratch/ppo_train.py
```

服务器断开连接后继续训练：

```bash
tmux new-session -d -s ppo_training \
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   python -u ppo_from_scratch/ppo_train.py 2>&1 | \
   tee ppo_from_scratch/outputs/training.log"
```

查看训练：

```bash
tmux attach -t ppo_training
```

从 tmux 脱离但不结束训练：按 `Ctrl+B`，然后按 `D`。

## 8. 训练输出

```text
ppo_from_scratch/
├── checkpoints/ppo_actor/       # 最佳 Actor、tokenizer、Value Head
├── outputs/
│   ├── metrics.csv              # 每轮 PPO 指标
│   ├── validation.csv           # 验证集选模历史
│   ├── evaluation.csv           # 测试集逐样本对比
│   ├── response_examples.md     # 原模型与 PPO 回答
│   ├── training_curves.png
│   └── before_after_evaluation.png
└── runs/                        # TensorBoard events
```

实时查看 TensorBoard：

```bash
tensorboard --logdir ppo_from_scratch/runs --port 6006
```

## 9. 关键指标怎么看

- `reward`：当前 rollout 的 Reward Model 原始分数。不同 episode 的 prompt 不同，不能只看相邻点涨跌。
- `sampled_kl`：Actor 相对冻结 Reference 的累计漂移，是长期稳定性的关键指标。
- `approx_kl`：当前 PPO update 相对采样时旧策略的局部变化。
- `clip_fraction`：触发 PPO ratio 裁剪的回答 token 比例。
- `value_loss`：Value Head 的拟合误差。
- `validation.csv`：选择 checkpoint 的依据。
- `evaluation.csv`：最终测试集结果，不能参与调参或选模。

## 10. 当前限制

- Reward Model 只是人类偏好的代理，Reward 提升不等价于所有维度的回答质量提升。
- 最终测试集只有 64 条样本，需要更大规模评估和人工盲评才能支持更强结论。
- 当前是单 GPU、全参数 Actor 更新，没有使用 LoRA、ZeRO 或分布式训练。
- 回答最大长度为 128 tokens，部分回答仍可能被截断。
- checkpoint 只保存模型，不保存 optimizer，因此当前不支持无损断点续训。
