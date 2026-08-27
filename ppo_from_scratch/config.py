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
    dataset_file: str = str(
        Path.home()
        / ".cache/huggingface/datasets/downloads"
        / "5b1e3bd48462132bbeb4467d6607701937a71503a8a190793ed22c060d79dc2c"
    )
    max_train_samples: int | None = 512

    # Training configuration
    num_episodes: int = 3
    num_ppo_epochs: int = 4

    # Batch configuration
    rollout_batch_size: int = 4
    generation_batch_size: int = 2
    train_batch_size: int = 2
    num_samples_per_prompt: int = 2

    # Sequence configuration
    max_prompt_length: int = 128
    max_new_tokens: int = 64
    reward_max_length: int = 512

    # Generation configuration
    temperature: float = 1.0
    top_p: float = 1.0

    # Optimizer configuration
    actor_learning_rate: float = 5e-6
    critic_learning_rate: float = 5e-5
    max_grad_norm: float = 1.0

    # PPO configuration
    gamma: float = 1.0
    gae_lambda: float = 0.95
    policy_clip_eps: float = 0.2
    value_clip_eps: float = 0.2

    # Reward configuration
    kl_coef: float = 0.05
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
    num_eval_samples: int = 16
    seed: int = 42
