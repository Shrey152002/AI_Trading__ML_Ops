"""PPO agent wrapper around stable-baselines3."""
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agents.base_agent import BaseAgent, MLflowLoggingCallback, combine_callbacks


class PPOAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "PPO"

    def train(
        self,
        env,
        total_timesteps: int,
        mlflow_run_id: Optional[str] = None,
        extra_callback: Optional[BaseCallback] = None,
    ) -> None:
        params = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "ent_coef": 0.0,
            **self.hyperparams,
        }
        self.model = PPO("MlpPolicy", env, verbose=0, **params)
        callback = combine_callbacks(MLflowLoggingCallback(log_freq=1000), extra_callback)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def load(self, path: str) -> None:
        self.model = PPO.load(path)
