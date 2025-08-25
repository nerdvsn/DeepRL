from typing import Optional
import numpy as np
import gymnasium as gym
from tensorboardX import SummaryWriter

from train_config import TrainConfig
from policy_evaluator import PolicyEvaluator

class ValueIterationAgent:
    def __init__(self, cfg: TrainConfig, writer: Optional[SummaryWriter] = None):
        self.cfg = cfg
        self.writer = writer

    def fit(self, env: gym.Env) -> np.ndarray:
        """
        Reine Value Iteration nach Bellman-Optimalität.
        Returns: V (np.ndarray) der Größe nS
        """
        P = env.unwrapped.P
        nS = env.observation_space.n
        nA = env.action_space.n
        V = np.zeros(nS, dtype=np.float64)

        for it in range(self.cfg.max_iters):
            delta = 0.0
            for s in range(nS):
                v_old = V[s]
                best = -np.inf
                for a in range(nA):
                    q = 0.0
                    for prob, s_next, r, done in P[s][a]:
                        q += prob * (r + (0.0 if done else self.cfg.gamma * V[s_next]))
                    if q > best:
                        best = q
                V[s] = best
                delta = max(delta, abs(v_old - V[s]))

            # TensorBoard-Logs
            if self.writer is not None:
                self.writer.add_scalar("train/delta", float(delta), it)
                self.writer.add_scalar("train/mean_V", float(np.mean(V)), it)
                self.writer.add_scalar("train/max_V", float(np.max(V)), it)

            # Zwischen-Eval
            if self.cfg.eval_every and (it % self.cfg.eval_every == 0 or it == self.cfg.max_iters - 1):
                policy = self.extract_policy(env, V)
                avg_ret, success, avg_steps = PolicyEvaluator.evaluate(
                    env, policy,
                    episodes=self.cfg.eval_episodes,
                    base_seed=self.cfg.eval_seed
                )
                if self.writer is not None:
                    self.writer.add_scalar("eval/avg_return", float(avg_ret), it)
                    self.writer.add_scalar("eval/success_rate", float(success), it)
                    self.writer.add_scalar("eval/avg_steps", float(avg_steps), it)

            if self.cfg.verbose and it % 50 == 0:
                print(f"Iter {it:4d}  delta={delta:.3e}  meanV={np.mean(V):.4f}")

            if delta < self.cfg.theta:
                if self.cfg.verbose:
                    print(f"Konvergiert nach {it} Iterationen (delta={delta:.3e}).")
                break

        return V

    def extract_policy(self, env: gym.Env, V: np.ndarray) -> np.ndarray:
        """π*(s) = argmax_a Σ p(s'|s,a) [r + γ V(s')]"""
        P = env.unwrapped.P
        nS = env.observation_space.n
        nA = env.action_space.n
        policy = np.zeros(nS, dtype=np.int64)

        for s in range(nS):
            q_values = np.zeros(nA, dtype=np.float64)
            for a in range(nA):
                for prob, s_next, r, done in P[s][a]:
                    q_values[a] += prob * (r + (0.0 if done else self.cfg.gamma * V[s_next]))
            policy[s] = int(np.argmax(q_values))
        return policy
