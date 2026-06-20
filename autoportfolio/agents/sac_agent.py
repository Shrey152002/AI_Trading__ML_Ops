"""SAC agent wrapper around stable-baselines3."""
from typing import Optional

from stable_baselines3 import SAC

from agents.base_agent import BaseAgent, MLflowLoggingCallback


class SACAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "SAC"

    def train(self, env, total_timesteps: int, mlflow_run_id: Optional[str] = None) -> None:
        params = {
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": 256,
            **self.hyperparams,
        }
        self.model = SAC("MlpPolicy", env, verbose=0, **params)
        callback = MLflowLoggingCallback(log_freq=1000)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def load(self, path: str) -> None:
        self.model = SAC.load(path)
