import time
import numpy as np
from tensorboardX import SummaryWriter

from env_config import EnvConfig
from train_config import TrainConfig
from env_builder import EnvBuilder
from value_iteration_agent import ValueIterationAgent
from policy_evaluator import PolicyEvaluator
from policy_renderer import PolicyRenderer


def main():
    # ---------- Konfiguration ----------
    env_cfg = EnvConfig(
        desc=["SFFF", "FHFH", "FFFH", "HFFG"],  # eigene Map; alternativ: map_name="4x4"
        map_name=None,
        is_slippery=False,
        seed=12345
    )
    train_cfg = TrainConfig(
        gamma=0.99,
        theta=1e-8,
        max_iters=10_000,
        eval_every=5,
        eval_episodes=50,
        eval_seed=env_cfg.seed,
        verbose=True,
        logdir="runs"
    )

    # ---------- Setup ----------
    EnvBuilder.set_global_seed(env_cfg.seed)
    env = EnvBuilder.make_env(env_cfg)

    run_name = f"VI_FrozenLake_seed{env_cfg.seed}_{int(time.time())}"
    writer = SummaryWriter(logdir=f"{train_cfg.logdir}/{run_name}")
    writer.add_text("meta/run_name", run_name, 0)
    writer.add_text("meta/seed", str(env_cfg.seed), 0)

    # Robuste Meta-Logs, ohne auf nicht vorhandene Env-Attribute zuzugreifen
    map_label = "custom" if env_cfg.desc else (env_cfg.map_name or "unknown")
    writer.add_text("meta/env", f"is_slippery={env_cfg.is_slippery}, map={map_label}", 0)
    writer.add_text(
        "meta/hparams",
        f"gamma={train_cfg.gamma}, theta={train_cfg.theta}, "
        f"eval_every={train_cfg.eval_every}, eval_eps={train_cfg.eval_episodes}",
        0
    )

    # ---------- Train ----------
    agent = ValueIterationAgent(train_cfg, writer=writer)
    V = agent.fit(env)
    policy = agent.extract_policy(env, V)

    # ---------- Finale Evaluation ----------
    avg_ret, success, avg_steps = PolicyEvaluator.evaluate(
        env, policy, episodes=200, base_seed=train_cfg.eval_seed
    )
    writer.add_scalar("final/avg_return", float(avg_ret))
    writer.add_scalar("final/success_rate", float(success))
    writer.add_scalar("final/avg_steps", float(avg_steps))
    writer.close()

    # ---------- Ausgaben ----------
    n = int(np.sqrt(env.observation_space.n))
    print("\nValue-Funktion (reshaped):")
    print(V.reshape(n, n))
    PolicyRenderer.render_policy_grid(env, policy)
    print(f"\nFinale Eval — avg_return={avg_ret:.3f}, success_rate={success:.2%}, avg_steps={avg_steps:.1f}")


if __name__ == "__main__":
    main()
