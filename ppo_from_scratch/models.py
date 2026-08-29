from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import PPOConfig


@dataclass
class PPOModels:
    actor: PreTrainedModel
    reference: PreTrainedModel
    reward_model: PreTrainedModel
    value_head: nn.Module
    reward_tokenizer: PreTrainedTokenizerBase
    device: torch.device


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1)
        nn.init.normal_(
            self.projection.weight,
            mean=0.0,
            std=1.0 / hidden_size**0.5,
        )
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states.float()).squeeze(-1)


def configure_actor_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def model_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=model_dtype(device),
    )


def load_models(config: PPOConfig) -> PPOModels:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    inference_dtype = model_dtype(device)

    actor = AutoModelForCausalLM.from_pretrained(
        config.actor_model_name,
        dtype=torch.float32,
    ).to(device)

    reference = AutoModelForCausalLM.from_pretrained(
        config.actor_model_name,
        dtype=inference_dtype,
    ).to(device)

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_name,
        dtype=inference_dtype,
    ).to(device)

    reward_tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_name
    )

    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)

    reference.eval()
    reward_model.eval()

    value_head = ValueHead(actor.config.hidden_size).to(device)

    return PPOModels(
        actor=actor,
        reference=reference,
        reward_model=reward_model,
        value_head=value_head,
        reward_tokenizer=reward_tokenizer,
        device=device,
    )


def save_checkpoint(
    models: PPOModels,
    actor_tokenizer: PreTrainedTokenizerBase,
    config: PPOConfig,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models.actor.save_pretrained(output_dir)
    actor_tokenizer.save_pretrained(output_dir)
    torch.save(
        models.value_head.state_dict(),
        output_dir / "value_head.pt",
    )
