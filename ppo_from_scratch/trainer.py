from collections import defaultdict

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .config import PPOConfig
from .models import PPOModels, save_checkpoint
from .reporting import TrainingReporter
from .rollout import (
    RolloutBatch,
    collect_rollout,
    generate_evaluation_responses,
    masked_mean,
    score_responses,
    token_logprobs,
)


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        models: PPOModels,
        actor_tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.config = config
        self.models = models
        self.actor_tokenizer = actor_tokenizer
        self.reporter = TrainingReporter(config)

        self.actor_optimizer = torch.optim.AdamW(
            models.actor.parameters(),
            lr=config.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.AdamW(
            models.value_head.parameters(),
            lr=config.critic_learning_rate,
        )

    def train(self, dataset: Dataset) -> None:
        try:
            for episode in range(self.config.num_episodes):
                examples = self._select_rollout_examples(dataset, episode)
                rollout = collect_rollout(
                    models=self.models,
                    actor_tokenizer=self.actor_tokenizer,
                    formatted_prompts=list(examples["formatted_prompt"]),
                    raw_prompts=list(examples["raw_prompt"]),
                    config=self.config,
                )

                metrics = self._rollout_metrics(rollout)
                metrics.update(self._ppo_update(rollout))
                self.reporter.log_episode(episode, metrics)
                self._print_episode(
                    episode,
                    metrics,
                    rollout.responses[0],
                )

            self.save()
            evaluation_metrics = self._evaluate(dataset)
            self._print_evaluation(evaluation_metrics)
        finally:
            self.reporter.close()

    def save(self) -> None:
        save_checkpoint(
            models=self.models,
            actor_tokenizer=self.actor_tokenizer,
            config=self.config,
        )
        print(f"Saved checkpoint to: {self.config.output_dir}")

    def _select_rollout_examples(
        self,
        dataset: Dataset,
        episode: int,
    ) -> Dataset:
        start = (
            episode * self.config.rollout_batch_size
        ) % len(dataset)
        indices = [
            (start + offset) % len(dataset)
            for offset in range(self.config.rollout_batch_size)
        ]
        return dataset.select(indices)

    def _ppo_update(
        self,
        rollout: RolloutBatch,
    ) -> dict[str, float]:
        self.models.actor.train()
        self.models.value_head.train()
        metrics: defaultdict[str, list[float]] = defaultdict(list)

        for _ in range(self.config.num_ppo_epochs):
            permutation = torch.randperm(
                len(rollout),
                device=self.models.device,
            )

            for start in range(
                0,
                len(rollout),
                self.config.train_batch_size,
            ):
                indices = permutation[
                    start : start + self.config.train_batch_size
                ]
                minibatch_metrics = self._update_minibatch(
                    rollout,
                    indices,
                )

                for name, value in minibatch_metrics.items():
                    metrics[name].append(value)

        return {
            name: sum(values) / len(values)
            for name, values in metrics.items()
        }

    def _update_minibatch(
        self,
        rollout: RolloutBatch,
        indices: torch.Tensor,
    ) -> dict[str, float]:
        input_ids = rollout.input_ids[indices]
        attention_mask = rollout.attention_mask[indices]
        action_mask = rollout.action_mask[indices]
        old_logprobs = rollout.old_logprobs[indices]
        old_values = rollout.old_values[indices]
        advantages = rollout.advantages[indices]
        returns = rollout.returns[indices]

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

        actor_outputs = self.models.actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        current_logprobs = token_logprobs(
            actor_outputs.logits,
            input_ids,
        ).float()

        log_ratio = current_logprobs - old_logprobs
        ratio = log_ratio.exp()
        policy_loss = self._policy_loss(
            ratio=ratio,
            advantages=advantages,
            action_mask=action_mask,
        )

        current_values = self.models.value_head(
            actor_outputs.hidden_states[-1][:, :-1].detach()
        )
        value_loss = self._value_loss(
            current_values=current_values,
            old_values=old_values,
            returns=returns,
            action_mask=action_mask,
        )

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.models.actor.parameters(),
            self.config.max_grad_norm,
        )
        self.actor_optimizer.step()

        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.models.value_head.parameters(),
            self.config.max_grad_norm,
        )
        self.critic_optimizer.step()

        with torch.no_grad():
            approx_kl = masked_mean(
                (ratio - 1.0) - log_ratio,
                action_mask,
            )
            clip_fraction = masked_mean(
                (torch.abs(ratio - 1.0) > self.config.policy_clip_eps)
                .to(ratio.dtype),
                action_mask,
            )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": clip_fraction.item(),
        }

    def _policy_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * ratio.clamp(
            1.0 - self.config.policy_clip_eps,
            1.0 + self.config.policy_clip_eps,
        )
        return masked_mean(
            torch.maximum(unclipped_loss, clipped_loss),
            action_mask,
        )

    def _value_loss(
        self,
        current_values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        clipped_values = old_values + (
            current_values - old_values
        ).clamp(
            -self.config.value_clip_eps,
            self.config.value_clip_eps,
        )
        unclipped_loss = (current_values - returns).square()
        clipped_loss = (clipped_values - returns).square()
        return 0.5 * masked_mean(
            torch.maximum(unclipped_loss, clipped_loss),
            action_mask,
        )

    def _rollout_metrics(
        self,
        rollout: RolloutBatch,
    ) -> dict[str, float]:
        response_lengths = rollout.action_mask.sum(dim=1).float()
        return {
            "reward": rollout.reward_scores.mean().item(),
            "sampled_kl": masked_mean(
                rollout.sampled_kl,
                rollout.action_mask,
            ).item(),
            "response_length": response_lengths.mean().item(),
        }

    def _evaluate(self, dataset: Dataset) -> dict[str, float]:
        evaluation_count = min(
            self.config.num_eval_samples,
            len(dataset),
        )
        start = len(dataset) - evaluation_count
        examples = dataset.select(
            range(start, len(dataset))
        )
        raw_prompts = list(examples["raw_prompt"])
        formatted_prompts = list(examples["formatted_prompt"])

        original_responses = generate_evaluation_responses(
            model=self.models.reference,
            tokenizer=self.actor_tokenizer,
            formatted_prompts=formatted_prompts,
            config=self.config,
            device=self.models.device,
        )
        trained_responses = generate_evaluation_responses(
            model=self.models.actor,
            tokenizer=self.actor_tokenizer,
            formatted_prompts=formatted_prompts,
            config=self.config,
            device=self.models.device,
        )

        original_scores = score_responses(
            models=self.models,
            raw_prompts=raw_prompts,
            responses=original_responses,
            config=self.config,
        ).cpu().tolist()
        trained_scores = score_responses(
            models=self.models,
            raw_prompts=raw_prompts,
            responses=trained_responses,
            config=self.config,
        ).cpu().tolist()

        return self.reporter.log_evaluation(
            raw_prompts=raw_prompts,
            original_responses=original_responses,
            trained_responses=trained_responses,
            original_scores=original_scores,
            trained_scores=trained_scores,
        )

    def _print_episode(
        self,
        episode: int,
        metrics: dict[str, float],
        sample_response: str,
    ) -> None:
        metric_text = " | ".join(
            f"{name}: {value:.4f}"
            for name, value in metrics.items()
        )
        print(
            f"Episode {episode + 1}/{self.config.num_episodes} | "
            f"{metric_text}"
        )
        print(f"Sample response: {sample_response}\n")

    def _print_evaluation(
        self,
        metrics: dict[str, float],
    ) -> None:
        print(
            "Evaluation | "
            f"original reward: {metrics['original_reward']:.4f} | "
            f"PPO reward: {metrics['trained_reward']:.4f} | "
            f"win rate: {metrics['win_rate'] * 100:.1f}%"
        )
        print(f"Reports saved to: {self.config.report_dir}")
