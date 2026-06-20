from agents.a2c_agent import A2CAgent
from agents.base_agent import BaseAgent
from agents.ddpg_agent import DDPGAgent
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "PPO": PPOAgent,
    "A2C": A2CAgent,
    "SAC": SACAgent,
    "DDPG": DDPGAgent,
}

__all__ = ["BaseAgent", "PPOAgent", "A2CAgent", "SACAgent", "DDPGAgent", "AGENT_REGISTRY"]
