import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .config import PPOConfig


class TrainingReporter:
    TRAINING_PLOTS = (
        (("reward",), "Mean reward"),
        (("sampled_kl", "approx_kl"), "KL diagnostics"),
        (("policy_loss",), "Policy loss"),
        (("value_loss",), "Value loss"),
        (("clip_fraction",), "Clip fraction"),
        (("response_length",), "Response length"),
    )

    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.report_dir = Path(config.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        run_name = datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
        tensorboard_path = Path(config.tensorboard_dir) / run_name
        self.writer = SummaryWriter(log_dir=tensorboard_path)
        self.history: list[dict[str, float]] = []
        self.validation_history: list[dict[str, float]] = []

    def log_episode(
        self,
        episode: int,
        metrics: dict[str, float],
    ) -> None:
        row = {"episode": float(episode + 1), **metrics}
        self.history.append(row)
        self._write_metrics_csv()
        self._write_training_plot()

        for name, value in metrics.items():
            self.writer.add_scalar(
                f"train/{name}",
                value,
                episode + 1,
            )
        self.writer.flush()

    def log_validation(
        self,
        episode: int,
        reference_reward: float,
        actor_reward: float,
        best_reward: float,
    ) -> None:
        row = {
            "episode": float(episode + 1),
            "reference_reward": reference_reward,
            "actor_reward": actor_reward,
            "best_reward": best_reward,
        }
        self.validation_history.append(row)
        self._write_validation_csv()

        self.writer.add_scalar(
            "validation/reference_reward",
            reference_reward,
            episode + 1,
        )
        self.writer.add_scalar(
            "validation/actor_reward",
            actor_reward,
            episode + 1,
        )
        self.writer.add_scalar(
            "validation/best_reward",
            best_reward,
            episode + 1,
        )
        self.writer.flush()

    def log_evaluation(
        self,
        raw_prompts: list[str],
        original_responses: list[str],
        trained_responses: list[str],
        original_scores: list[float],
        trained_scores: list[float],
    ) -> dict[str, float]:
        original_mean = sum(original_scores) / len(original_scores)
        trained_mean = sum(trained_scores) / len(trained_scores)
        win_rate = sum(
            trained > original
            for original, trained in zip(
                original_scores,
                trained_scores,
                strict=True,
            )
        ) / len(original_scores)

        metrics = {
            "original_reward": original_mean,
            "trained_reward": trained_mean,
            "reward_improvement": trained_mean - original_mean,
            "win_rate": win_rate,
        }

        self._write_evaluation_csv(
            raw_prompts=raw_prompts,
            original_responses=original_responses,
            trained_responses=trained_responses,
            original_scores=original_scores,
            trained_scores=trained_scores,
        )
        self._write_response_examples(
            raw_prompts=raw_prompts,
            original_responses=original_responses,
            trained_responses=trained_responses,
            original_scores=original_scores,
            trained_scores=trained_scores,
        )
        self._write_evaluation_plot(
            original_scores=original_scores,
            trained_scores=trained_scores,
            metrics=metrics,
        )

        final_step = self.config.num_episodes
        for name, value in metrics.items():
            self.writer.add_scalar(
                f"evaluation/{name}",
                value,
                final_step,
            )
        self.writer.flush()

        return metrics

    def close(self) -> None:
        if self.history:
            self._write_metrics_csv()
            self._write_training_plot()
        self.writer.close()

    def _write_metrics_csv(self) -> None:
        path = self.report_dir / "metrics.csv"
        fieldnames = list(self.history[0])

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def _write_validation_csv(self) -> None:
        path = self.report_dir / "validation.csv"
        fieldnames = list(self.validation_history[0])

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.validation_history)

    def _write_evaluation_csv(
        self,
        raw_prompts: list[str],
        original_responses: list[str],
        trained_responses: list[str],
        original_scores: list[float],
        trained_scores: list[float],
    ) -> None:
        path = self.report_dir / "evaluation.csv"
        fieldnames = [
            "prompt",
            "original_response",
            "trained_response",
            "original_reward",
            "trained_reward",
            "reward_change",
            "trained_wins",
        ]

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for prompt, original_response, trained_response, original, trained in zip(
                raw_prompts,
                original_responses,
                trained_responses,
                original_scores,
                trained_scores,
                strict=True,
            ):
                writer.writerow(
                    {
                        "prompt": prompt,
                        "original_response": original_response,
                        "trained_response": trained_response,
                        "original_reward": original,
                        "trained_reward": trained,
                        "reward_change": trained - original,
                        "trained_wins": trained > original,
                    }
                )

    def _write_response_examples(
        self,
        raw_prompts: list[str],
        original_responses: list[str],
        trained_responses: list[str],
        original_scores: list[float],
        trained_scores: list[float],
    ) -> None:
        sections = ["# PPO evaluation examples\n"]

        for index, values in enumerate(
            zip(
                raw_prompts,
                original_responses,
                trained_responses,
                original_scores,
                trained_scores,
                strict=True,
            ),
            start=1,
        ):
            prompt, original_response, trained_response, original, trained = values
            sections.extend(
                [
                    f"## Example {index}\n",
                    "### Prompt\n",
                    f"{prompt}\n",
                    f"### Original response — reward {original:.4f}\n",
                    f"{original_response}\n",
                    f"### PPO response — reward {trained:.4f}\n",
                    f"{trained_response}\n",
                ]
            )

        path = self.report_dir / "response_examples.md"
        path.write_text("\n".join(sections), encoding="utf-8")

    def _write_training_plot(self) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        figure, axes = plt.subplots(
            3,
            2,
            figsize=(13, 12),
            constrained_layout=True,
        )
        episodes = [int(row["episode"]) for row in self.history]

        for axis, (metric_names, title) in zip(
            axes.flat,
            self.TRAINING_PLOTS,
            strict=True,
        ):
            for metric_name in metric_names:
                values = [row[metric_name] for row in self.history]
                axis.plot(
                    episodes,
                    values,
                    marker="o",
                    linewidth=2,
                    label=metric_name,
                )

            axis.set_title(title)
            axis.set_xlabel("Episode")
            axis.set_xticks(episodes)
            if len(metric_names) > 1:
                axis.legend()

        figure.suptitle("PPO training curves", fontsize=18)
        figure.savefig(
            self.report_dir / "training_curves.png",
            dpi=160,
        )
        plt.close(figure)

    def _write_evaluation_plot(
        self,
        original_scores: list[float],
        trained_scores: list[float],
        metrics: dict[str, float],
    ) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(17, 5),
            constrained_layout=True,
        )

        axes[0].bar(
            ["Original", "PPO"],
            [metrics["original_reward"], metrics["trained_reward"]],
            color=["#6b7280", "#2563eb"],
        )
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_title("Mean reward")
        axes[0].set_ylabel("Reward model score")

        for index, (original, trained) in enumerate(
            zip(original_scores, trained_scores, strict=True)
        ):
            color = "#16a34a" if trained > original else "#dc2626"
            axes[1].plot(
                [0, 1],
                [original, trained],
                marker="o",
                color=color,
                alpha=0.65,
            )
        axes[1].set_xticks([0, 1], ["Original", "PPO"])
        axes[1].set_title("Per-prompt reward change")
        axes[1].set_ylabel("Reward model score")

        reward_changes = [
            trained - original
            for original, trained in zip(
                original_scores,
                trained_scores,
                strict=True,
            )
        ]
        colors = [
            "#16a34a" if change > 0 else "#dc2626"
            for change in reward_changes
        ]
        axes[2].bar(
            range(1, len(reward_changes) + 1),
            reward_changes,
            color=colors,
        )
        axes[2].axhline(0.0, color="black", linewidth=0.8)
        axes[2].set_title(
            f"Reward delta — win rate {metrics['win_rate'] * 100:.1f}%"
        )
        axes[2].set_xlabel("Evaluation prompt")
        axes[2].set_ylabel("PPO reward - original reward")

        figure.suptitle("Original model vs PPO model", fontsize=18)
        figure.savefig(
            self.report_dir / "before_after_evaluation.png",
            dpi=160,
        )
        plt.close(figure)
