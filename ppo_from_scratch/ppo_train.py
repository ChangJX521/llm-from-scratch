from pathlib import Path
import random
import sys

import torch
from transformers import AutoTokenizer


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppo_from_scratch.config import PPOConfig
from ppo_from_scratch.models import (
    configure_actor_tokenizer,
    load_models,
)
from ppo_from_scratch.rollout import build_prompt_dataset
from ppo_from_scratch.trainer import PPOTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    config = PPOConfig()
    set_seed(config.seed)

    actor_tokenizer = AutoTokenizer.from_pretrained(
        config.actor_model_name
    )
    configure_actor_tokenizer(actor_tokenizer)

    dataset = build_prompt_dataset(
        tokenizer=actor_tokenizer,
        config=config,
    )
    models = load_models(config)

    trainer = PPOTrainer(
        config=config,
        models=models,
        actor_tokenizer=actor_tokenizer,
    )
    trainer.train(dataset)


if __name__ == "__main__":
    main()
