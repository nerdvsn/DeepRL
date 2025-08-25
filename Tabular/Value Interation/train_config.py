from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
    gamma: float = 0.99
    theta: float = 1e-8
    max_iters: int = 10_000
    eval_every: int = 5
    eval_episodes: int = 50
    eval_seed: Optional[int] = None
    verbose: bool = True
    logdir: str = "runs"
