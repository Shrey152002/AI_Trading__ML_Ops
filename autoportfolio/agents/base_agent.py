"""Abstract base class shared by all RL agents, plus a common MLflow logging callback."""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

logger = logging.getLogger(__name__)


class MLflowLoggingCallback(BaseCallback):
    """Logs rolling reward and loss metrics to the active MLflow run every `log_freq` steps."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0 and mlflow.active_run() is not None:
            metrics = {}

            ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
            if ep_info_buffer and len(ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in ep_info_buffer if "r" in ep]
                if rewards:
                    metrics["reward_mean"] = float(np.mean(rewards))

            logger_dict = getattr(self.model.logger, "name_to_value", {})
            for key in ("train/policy_loss", "train/value_loss", "train/actor_loss", "train/critic_loss"):
                if key in logger_dict:
                    metrics[key.split("/")[-1]] = float(logger_dict[key])

            if metrics:
                mlflow.log_metrics(metrics, step=self.num_timesteps)

        return True


def combine_callbacks(*callbacks: Optional[BaseCallback]) -> BaseCallback:
    """Combines the always-on MLflow logging callback with an optional extra one
    (e.g. progress reporting) into a single SB3 callback."""
    return CallbackList([cb for cb in callbacks if cb is not None])


class BaseAgent(ABC):
    """Common interface implemented by PPO/A2C/SAC/DDPG wrappers around stable-baselines3."""

    def __init__(self, hyperparams: Optional[dict] = None):
        self.hyperparams = hyperparams or {}
        self.model = None

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def train(
        self,
        env,
        total_timesteps: int,
        mlflow_run_id: Optional[str] = None,
        extra_callback: Optional[BaseCallback] = None,
    ) -> None:
        ...

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.name} agent has not been trained or loaded yet")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError(f"{self.name} agent has no model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("Saved %s model to %s", self.name, path)

    @abstractmethod
    def load(self, path: str) -> None:
        ...
