import numpy as np
import gymnasium as gym

class PolicyRenderer:
    # Gymnasium FrozenLake Actions: 0=Left, 1=Down, 2=Right, 3=Up
    ARROW_MAP = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    @staticmethod
    def render_policy_grid(env: gym.Env, policy: np.ndarray):
        """Druckt die Policy als Pfeil-Grid. Keine Pfeile auf S/H/G."""
        n = int(np.sqrt(env.observation_space.n))
        desc = env.unwrapped.desc.astype(str).copy()

        for s, a in enumerate(policy):
            r, c = divmod(s, n)
            if desc[r, c] in ("H", "G", "S"):
                continue
            desc[r, c] = PolicyRenderer.ARROW_MAP[int(a)]

        print("\nPolicy (Pfeile) — S=Start, G=Goal, H=Hole:")
        for r in range(n):
            print(" ".join(desc[r]))
