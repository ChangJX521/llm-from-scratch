# LLM From Scratch

这个仓库用于从核心公式出发实现和理解大语言模型训练算法。目前已经完成的是一个可运行、可评估、能够在独立测试集上获得正向提升的 PPO（Proximal Policy Optimization）RLHF 训练流程。

这里的 “from scratch” 指的是不依赖 TRL 等现成 PPO Trainer，自己实现 rollout、token log-probability、KL 惩罚、GAE、PPO clipped loss、Value Head、验证选模和评估报告；Actor、Reference 与 Reward Model 仍然使用预训练模型。

## 当前实现

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| PPO rollout | 完成 | 多样本生成、奖励打分、token 级 KL |
| Advantage | 完成 | GAE 与有效 token mask |
| PPO update | 完成 | clipped policy/value loss、梯度裁剪 |
| RLHF models | 完成 | Actor、Reference、Reward Model、Value Head |
| 显存优化 | 完成 | FP32 主权重、BF16 autocast、分批 log-prob 计算 |
| 稳定训练 | 完成 | 累计 Reference KL 保护、验证集选模、早停 |
| 实验报告 | 完成 | CSV、TensorBoard、PNG 图表、回答对比 |

## 最终结果

2026-08-29 的最终实验使用 384/64/64 的训练、验证、测试划分。训练在第 28 轮早停并恢复第 12 轮最佳模型。

| 指标 | 原模型 | PPO 模型 |
| --- | ---: | ---: |
| 测试集平均 Reward | -0.8938 | **-0.5150** |
| 平均 Reward 变化 | — | **+0.3787** |
| 中位数 Reward 变化 | — | **+0.3181** |
| 测试集胜率 | — | **60.9%** |

64 条独立测试样本中，PPO 模型 39 胜、1 平、24 负；平均提升的近似 95% 置信区间为 `[+0.103, +0.655]`。

详细实现与运行方法见 [PPO README](ppo_from_scratch/README.md)，完整实验分析见 [实验报告](ppo_from_scratch/EXPERIMENT_REPORT.md)。

## 仓库结构

```text
.
├── README.md
├── requirements.txt
└── ppo_from_scratch/
    ├── README.md
    ├── EXPERIMENT_REPORT.md
    ├── config.py
    ├── models.py
    ├── rollout.py
    ├── trainer.py
    ├── reporting.py
    ├── ppo_train.py
    └── assets/
```

训练生成的 `outputs/`、`runs/` 和 `checkpoints/` 默认不会提交到 Git；最终实验的关键图表已单独保存在 `ppo_from_scratch/assets/`。
