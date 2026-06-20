"""A2C agent wrapper around stable-baselines3."""
from typing import Optional

from stable_baselines3 import A2C

from agents.base_agent import BaseAgent, MLflowLoggingCallback


class A2CAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "A2C"

    def train(self, env, total_timesteps: int, mlflow_run_id: Optional[str] = None) -> None:
        params = {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "ent_coef": 0.0,
            **self.hyperparams,
        }
        self.model = A2C("MlpPolicy", env, verbose=0, **params)
        callback = MLflowLoggingCallback(log_freq=1000)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def load(self, path: str) -> None:
        self.model = A2C.load(path)
