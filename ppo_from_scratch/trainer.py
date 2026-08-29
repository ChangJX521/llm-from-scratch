from collections import defaultdict

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .config import PPOConfig
from .models import PPOModels, autocast_context, save_checkpoint
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
        validation_examples = self._select_validation_examples(
            dataset
        )
        reference_validation_reward = self._validation_reward(
            model=self.models.reference,
            examples=validation_examples,
        )
        best_validation_reward = reference_validation_reward
        best_actor_state = self._copy_state_dict(
            self.models.actor
        )
        best_value_state = self._copy_state_dict(
            self.models.value_head
        )
        best_episode = 0
        checks_without_improvement = 0

        print(
            "Validation baseline | "
            f"reference reward: {reference_validation_reward:.4f}"
        )

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
                reference_kl = metrics["sampled_kl"]

                if reference_kl > self.config.max_reference_kl:
                    print(
                        "Stopping before PPO update: reference KL "
                        f"{reference_kl:.4f} exceeded safety limit "
                        f"{self.config.max_reference_kl:.4f}."
                    )
                    break

                metrics["kl_coef"] = self.config.kl_coef
                metrics.update(self._ppo_update(rollout))
                self.reporter.log_episode(episode, metrics)
                self._print_episode(
                    episode,
                    metrics,
                    rollout.responses[0],
                )

                self._adjust_kl_coefficient(reference_kl)

                if (
                    episode + 1
                ) % self.config.validation_interval != 0:
                    continue

                actor_validation_reward = self._validation_reward(
                    model=self.models.actor,
                    examples=validation_examples,
                )
                improved = actor_validation_reward > (
                    best_validation_reward
                    + self.config.min_validation_improvement
                )

                if improved:
                    best_validation_reward = actor_validation_reward
                    best_actor_state = self._copy_state_dict(
                        self.models.actor
                    )
                    best_value_state = self._copy_state_dict(
                        self.models.value_head
                    )
                    best_episode = episode + 1
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1

                self.reporter.log_validation(
                    episode=episode,
                    reference_reward=reference_validation_reward,
                    actor_reward=actor_validation_reward,
                    best_reward=best_validation_reward,
                )
                print(
                    f"Validation {episode + 1} | "
                    f"actor reward: {actor_validation_reward:.4f} | "
                    f"best reward: {best_validation_reward:.4f} | "
                    f"best episode: {best_episode}"
                )

                if (
                    checks_without_improvement
                    >= self.config.early_stopping_patience
                ):
                    print(
                        "Early stopping: validation reward did not "
                        f"improve for {checks_without_improvement} "
                        "checks."
                    )
                    break

            self.models.actor.load_state_dict(best_actor_state)
            self.models.value_head.load_state_dict(best_value_state)
            print(
                f"Restored best checkpoint from episode {best_episode} "
                f"with validation reward {best_validation_reward:.4f}."
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

    def _select_validation_examples(
        self,
        dataset: Dataset,
    ) -> Dataset:
        test_start = len(dataset) - self.config.num_eval_samples
        validation_start = (
            test_start - self.config.num_validation_samples
        )
        training_count = (
            self.config.num_episodes
            * self.config.rollout_batch_size
        )

        if training_count > validation_start:
            raise ValueError(
                "Training, validation, and test splits overlap."
            )

        return dataset.select(
            range(validation_start, test_start)
        )

    @staticmethod
    def _copy_state_dict(
        model: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    def _validation_reward(
        self,
        model: torch.nn.Module,
        examples: Dataset,
    ) -> float:
        raw_prompts = list(examples["raw_prompt"])
        responses = generate_evaluation_responses(
            model=model,
            tokenizer=self.actor_tokenizer,
            formatted_prompts=list(examples["formatted_prompt"]),
            config=self.config,
            device=self.models.device,
        )
        scores = score_responses(
            models=self.models,
            raw_prompts=raw_prompts,
            responses=responses,
            config=self.config,
        )
        return scores.mean().item()

    def _adjust_kl_coefficient(
        self,
        reference_kl: float,
    ) -> None:
        if reference_kl <= self.config.target_reference_kl:
            return

        self.config.kl_coef = min(
            self.config.kl_coef
            * self.config.kl_coef_multiplier,
            1.0,
        )
        print(
            "Increasing KL coefficient to "
            f"{self.config.kl_coef:.4f} because reference KL "
            f"reached {reference_kl:.4f}."
        )

    def _ppo_update(
        self,
        rollout: RolloutBatch,
    ) -> dict[str, float]:
        self.models.actor.train()
        self.models.value_head.train()
        metrics: defaultdict[str, list[float]] = defaultdict(list)

        ppo_epochs_used = 0
        stop_early = False

        for _ in range(self.config.num_ppo_epochs):
            epoch_approx_kl = []
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
                epoch_approx_kl.append(
                    minibatch_metrics["approx_kl"]
                )

                mean_epoch_kl = sum(epoch_approx_kl) / len(
                    epoch_approx_kl
                )
                if mean_epoch_kl > self.config.target_kl:
                    stop_early = True
                    break

            ppo_epochs_used += 1
            if stop_early:
                break

        averaged_metrics = {
            name: sum(values) / len(values)
            for name, values in metrics.items()
        }
        averaged_metrics["ppo_epochs_used"] = float(
            ppo_epochs_used
        )
        return averaged_metrics

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

        with autocast_context(self.models.device):
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
