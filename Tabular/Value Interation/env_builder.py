import random
from typing import Optional
import numpy as np
import gymnasium as gym
from env_config import EnvConfig

class EnvBuilder:
    @staticmethod
    def set_global_seed(seed: int):
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def make_env(cfg: EnvConfig) -> gym.Env:
        """Erstellt eine FrozenLake-Env (mit optionaler Custom-Map) + Seeding."""
        kwargs = {"is_slippery": cfg.is_slippery}
        if cfg.desc is not None:
            kwargs["desc"] = cfg.desc
        if cfg.map_name is not None:
            kwargs["map_name"] = cfg.map_name

        env = gym.make("FrozenLake-v1", **kwargs)
        env.reset(seed=cfg.seed)
        try:
            env.action_space.seed(cfg.seed)
            env.observation_space.seed(cfg.seed)
        except Exception:
            pass
        return env
