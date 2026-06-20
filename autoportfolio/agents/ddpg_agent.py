"""DDPG agent wrapper around stable-baselines3."""
from typing import Optional

from stable_baselines3 import DDPG

from agents.base_agent import BaseAgent, MLflowLoggingCallback


class DDPGAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "DDPG"

    def train(self, env, total_timesteps: int, mlflow_run_id: Optional[str] = None) -> None:
        params = {
            "learning_rate": 1e-3,
            "buffer_size": 50000,
            "batch_size": 128,
            "tau": 0.005,
            **self.hyperparams,
        }
        self.model = DDPG("MlpPolicy", env, verbose=0, **params)
        callback = MLflowLoggingCallback(log_freq=1000)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def load(self, path: str) -> None:
        self.model = DDPG.load(path)
