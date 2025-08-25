from typing import Optional, Tuple
import numpy as np
import gymnasium as gym

class PolicyEvaluator:
    @staticmethod
    def run_episode(env: gym.Env, policy: np.ndarray,
                    reset_seed: Optional[int] = None,
                    max_steps: int = 10_000) -> Tuple[float, int, bool]:
        if reset_seed is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=reset_seed)

        total_r, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            a = int(policy[obs])
            obs, r, done, truncated, _ = env.step(a)
            total_r += r
            steps += 1
            if truncated:
                break
        return total_r, steps, done

    @staticmethod
    def evaluate(env: gym.Env, policy: np.ndarray,
                 episodes: int = 50,
                 base_seed: Optional[int] = None) -> Tuple[float, float, float]:
        """Gibt (avg_return, success_rate, avg_steps) zurück."""
        rng = np.random.default_rng(base_seed) if base_seed is not None else None
        wins = 0
        total_ret = 0.0
        total_steps = 0
        for _ in range(episodes):
            ep_seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else None
            ret, steps, done = PolicyEvaluator.run_episode(env, policy, reset_seed=ep_seed)
            total_ret += ret
            total_steps += steps
            wins += int(done and ret > 0)
        return total_ret / episodes, wins / episodes, total_steps / episodes
