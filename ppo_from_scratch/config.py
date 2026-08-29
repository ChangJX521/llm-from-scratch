import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PPOConfig:
    # Model configuration
    actor_model_name: str = "Qwen/Qwen3-0.6B"
    reward_model_name: str = (
        "OpenAssistant/reward-model-deberta-v3-large-v2"
    )

    # Dataset configuration
    dataset_file: str = os.environ.get(
        "WEBGPT_DATASET_FILE",
        "https://huggingface.co/datasets/openai/"
        "webgpt_comparisons/resolve/refs%2Fconvert%2Fparquet/"
        "default/train/0000.parquet",
    )
    max_train_samples: int | None = 512

    # Training configuration
    # 48 * 8 = 384 training prompts. The next 64 are used only for
    # checkpoint selection, and the final 64 remain untouched for testing.
    num_episodes: int = 48
    num_ppo_epochs: int = 1
    validation_interval: int = 4
    early_stopping_patience: int = 4

    # Batch configuration
    rollout_batch_size: int = 8
    generation_batch_size: int = 2
    train_batch_size: int = 2
    num_samples_per_prompt: int = 2

    # Sequence configuration
    max_prompt_length: int = 192
    max_new_tokens: int = 128
    reward_max_length: int = 512

    # Generation configuration
    temperature: float = 0.9
    top_p: float = 0.95

    # Optimizer configuration
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    # PPO configuration
    gamma: float = 1.0
    gae_lambda: float = 0.95
    policy_clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    target_kl: float = 0.01

    # Reward configuration
    kl_coef: float = 0.1
    kl_coef_multiplier: float = 1.5
    target_reference_kl: float = 0.03
    max_reference_kl: float = 0.08
    reward_clip: float = 5.0

    # Output and reproducibility
    output_dir: str = str(
        Path(__file__).resolve().parent
        / "checkpoints"
        / "ppo_actor"
    )
    report_dir: str = str(
        Path(__file__).resolve().parent / "outputs"
    )
    tensorboard_dir: str = str(
        Path(__file__).resolve().parent / "runs"
    )
    num_validation_samples: int = 64
    num_eval_samples: int = 64
    min_validation_improvement: float = 0.01
    seed: int = 42
