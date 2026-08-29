from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import PPOConfig
from .models import PPOModels, autocast_context


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    reward_scores: torch.Tensor
    sampled_kl: torch.Tensor
    raw_prompts: list[str]
    responses: list[str]

    def __len__(self) -> int:
        return self.input_ids.shape[0]


def build_prompt_dataset(
    tokenizer: PreTrainedTokenizerBase,
    config: PPOConfig,
) -> Dataset:
    dataset = load_dataset(
        "parquet",
        data_files={"train": config.dataset_file},
        split="train",
    )
    dataset = dataset.shuffle(seed=config.seed)

    if config.max_train_samples is not None:
        sample_count = min(config.max_train_samples, len(dataset))
        dataset = dataset.select(range(sample_count))

    def format_example(example: dict) -> dict[str, str]:
        raw_prompt = example["question"]["full_text"]
        messages = [{"role": "user", "content": raw_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return {
            "raw_prompt": raw_prompt,
            "formatted_prompt": formatted_prompt,
        }

    return dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting WebGPT prompts",
    )


def token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    prediction_logits = logits[:, :-1].transpose(1, 2)
    target_ids = input_ids[:, 1:]
    return -F.cross_entropy(
        prediction_logits,
        target_ids,
        reduction="none",
    )


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask_float = mask.to(values.dtype)
    return (values * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def response_length(
    response_ids: torch.Tensor,
    eos_token_id: int | None,
) -> int:
    if eos_token_id is None:
        return response_ids.shape[0]

    eos_positions = torch.nonzero(
        response_ids == eos_token_id,
        as_tuple=False,
    )
    if len(eos_positions) == 0:
        return response_ids.shape[0]

    return int(eos_positions[0].item()) + 1


def generate_sequences(
    actor: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    formatted_prompts: list[str],
    raw_prompts: list[str],
    config: PPOConfig,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
]:
    sequence_list = []
    attention_list = []
    response_mask_list = []
    repeated_raw_prompts = []
    response_texts = []

    actor.eval()

    for start in range(
        0,
        len(formatted_prompts),
        config.generation_batch_size,
    ):
        prompt_batch = formatted_prompts[
            start : start + config.generation_batch_size
        ]
        raw_prompt_batch = raw_prompts[
            start : start + config.generation_batch_size
        ]
        model_inputs = tokenizer(
            prompt_batch,
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
            return_tensors="pt",
        ).to(device)
        padded_prompt_length = model_inputs["input_ids"].shape[1]

        with torch.inference_mode(), autocast_context(device):
            generated_ids = actor.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                num_return_sequences=config.num_samples_per_prompt,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for prompt_index, raw_prompt in enumerate(raw_prompt_batch):
            prompt_mask = model_inputs["attention_mask"][
                prompt_index
            ].bool()
            prompt_ids = model_inputs["input_ids"][prompt_index][
                prompt_mask
            ]

            for sample_index in range(config.num_samples_per_prompt):
                generated_index = (
                    prompt_index * config.num_samples_per_prompt
                    + sample_index
                )
                completion_ids = generated_ids[
                    generated_index,
                    padded_prompt_length:,
                ]
                completion_length = response_length(
                    completion_ids,
                    tokenizer.eos_token_id,
                )
                completion_ids = completion_ids[:completion_length]

                sequence = torch.cat([prompt_ids, completion_ids])
                response_mask = torch.cat(
                    [
                        torch.zeros_like(prompt_ids, dtype=torch.bool),
                        torch.ones_like(
                            completion_ids,
                            dtype=torch.bool,
                        ),
                    ]
                )

                sequence_list.append(sequence)
                attention_list.append(
                    torch.ones_like(sequence, dtype=torch.bool)
                )
                response_mask_list.append(response_mask)
                repeated_raw_prompts.append(raw_prompt)
                response_texts.append(
                    tokenizer.decode(
                        completion_ids,
                        skip_special_tokens=True,
                    )
                )

    input_ids = pad_sequence(
        sequence_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(device)
    attention_mask = pad_sequence(
        attention_list,
        batch_first=True,
        padding_value=False,
    ).to(device)
    response_token_mask = pad_sequence(
        response_mask_list,
        batch_first=True,
        padding_value=False,
    ).to(device)

    return (
        input_ids,
        attention_mask,
        response_token_mask,
        repeated_raw_prompts,
        response_texts,
    )


def generate_evaluation_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    formatted_prompts: list[str],
    config: PPOConfig,
    device: torch.device,
) -> list[str]:
    responses = []
    model.eval()

    for start in range(
        0,
        len(formatted_prompts),
        config.generation_batch_size,
    ):
        prompt_batch = formatted_prompts[
            start : start + config.generation_batch_size
        ]
        model_inputs = tokenizer(
            prompt_batch,
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
            return_tensors="pt",
        ).to(device)
        prompt_length = model_inputs["input_ids"].shape[1]

        with torch.inference_mode(), autocast_context(device):
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        responses.extend(
            tokenizer.batch_decode(
                generated_ids[:, prompt_length:],
                skip_special_tokens=True,
            )
        )

    return responses


def score_responses(
    models: PPOModels,
    raw_prompts: list[str],
    responses: list[str],
    config: PPOConfig,
) -> torch.Tensor:
    scores = []

    for start in range(
        0,
        len(responses),
        config.generation_batch_size,
    ):
        prompt_batch = raw_prompts[
            start : start + config.generation_batch_size
        ]
        response_batch = responses[
            start : start + config.generation_batch_size
        ]
        reward_inputs = models.reward_tokenizer(
            prompt_batch,
            response_batch,
            padding=True,
            truncation=True,
            max_length=config.reward_max_length,
            return_tensors="pt",
        ).to(models.device)

        with torch.inference_mode(), autocast_context(models.device):
            logits = models.reward_model(**reward_inputs).logits

        batch_scores = logits.reshape(logits.shape[0], -1)[:, 0]
        scores.append(batch_scores.float())

    reward_scores = torch.cat(scores)
    return reward_scores.clamp(
        min=-config.reward_clip,
        max=config.reward_clip,
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(
        rewards.shape[0],
        device=rewards.device,
        dtype=rewards.dtype,
    )

    for token_index in reversed(range(rewards.shape[1])):
        current_mask = action_mask[:, token_index].to(rewards.dtype)

        if token_index + 1 < rewards.shape[1]:
            next_mask = action_mask[:, token_index + 1].to(
                rewards.dtype
            )
            next_value = values[:, token_index + 1]
        else:
            next_mask = torch.zeros_like(current_mask)
            next_value = torch.zeros_like(last_advantage)

        delta = (
            rewards[:, token_index]
            + gamma * next_value * next_mask
            - values[:, token_index]
        )
        last_advantage = (
            delta
            + gamma * gae_lambda * next_mask * last_advantage
        )
        last_advantage = last_advantage * current_mask
        advantages[:, token_index] = last_advantage

    returns = advantages + values
    valid_advantages = advantages[action_mask]
    advantage_mean = valid_advantages.mean()
    advantage_std = valid_advantages.std(unbiased=False).clamp_min(1e-8)
    advantages = torch.where(
        action_mask,
        (advantages - advantage_mean) / advantage_std,
        torch.zeros_like(advantages),
    )

    return advantages, returns


def collect_rollout(
    models: PPOModels,
    actor_tokenizer: PreTrainedTokenizerBase,
    formatted_prompts: list[str],
    raw_prompts: list[str],
    config: PPOConfig,
) -> RolloutBatch:
    (
        input_ids,
        attention_mask,
        response_token_mask,
        repeated_raw_prompts,
        responses,
    ) = generate_sequences(
        actor=models.actor,
        tokenizer=actor_tokenizer,
        formatted_prompts=formatted_prompts,
        raw_prompts=raw_prompts,
        config=config,
        device=models.device,
    )
    action_mask = response_token_mask[:, 1:]

    models.actor.eval()
    models.value_head.eval()

    old_logprob_batches = []
    old_value_batches = []
    reference_logprob_batches = []

    for start in range(
        0,
        input_ids.shape[0],
        config.generation_batch_size,
    ):
        end = start + config.generation_batch_size
        batch_input_ids = input_ids[start:end]
        batch_attention_mask = attention_mask[start:end]

        with torch.inference_mode(), autocast_context(models.device):
            actor_outputs = models.actor(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            old_logprob_batches.append(
                token_logprobs(
                    actor_outputs.logits,
                    batch_input_ids,
                ).float()
            )
            old_value_batches.append(
                models.value_head(
                    actor_outputs.hidden_states[-1][:, :-1]
                )
            )
            del actor_outputs

            reference_outputs = models.reference(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                use_cache=False,
            )
            reference_logprob_batches.append(
                token_logprobs(
                    reference_outputs.logits,
                    batch_input_ids,
                ).float()
            )
            del reference_outputs

    old_logprobs = torch.cat(old_logprob_batches)
    old_values = torch.cat(old_value_batches)
    reference_logprobs = torch.cat(reference_logprob_batches)

    reward_scores = score_responses(
        models=models,
        raw_prompts=repeated_raw_prompts,
        responses=responses,
        config=config,
    )

    sampled_kl = old_logprobs - reference_logprobs
    rewards = -config.kl_coef * sampled_kl
    rewards = torch.where(
        action_mask,
        rewards,
        torch.zeros_like(rewards),
    )

    for sample_index in range(rewards.shape[0]):
        response_positions = torch.nonzero(
            action_mask[sample_index],
            as_tuple=False,
        ).flatten()
        final_position = response_positions[-1]
        rewards[sample_index, final_position] += reward_scores[
            sample_index
        ]

    advantages, returns = compute_gae(
        rewards=rewards,
        values=old_values,
        action_mask=action_mask,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    return RolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        action_mask=action_mask,
        old_logprobs=old_logprobs.detach(),
        old_values=old_values.detach(),
        rewards=rewards.detach(),
        advantages=advantages.detach(),
        returns=returns.detach(),
        reward_scores=reward_scores.detach(),
        sampled_kl=sampled_kl.detach(),
        raw_prompts=repeated_raw_prompts,
        responses=responses,
    )
