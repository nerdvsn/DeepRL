# policy_player.py
# Spielt eine Policy in FrozenLake ab (sichtbar im Terminal).
# Nutzung:
#   python policy_player.py                   # berechnet VI-Policy und spielt 3 Episoden
#   python policy_player.py --load model.npz  # lädt gespeicherte Policy und spielt
#   python policy_player.py --episodes 10 --sleep 0.3
#   python policy_player.py --manual          # Schritt-für-Schritt per Enter

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym

from env_config import EnvConfig
from env_builder import EnvBuilder
from train_config import TrainConfig
from value_iteration_agent import ValueIterationAgent

ACTION_NAMES = ["Left", "Down", "Right", "Up"]
ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def clear():
    # terminal clear (portable-ish)
    print("\033[H\033[2J", end="")
    sys.stdout.flush()


def load_policy_npz(path: str):
    data = np.load(path, allow_pickle=False)
    policy = data["policy"].astype(int)
    V = data["V"] if "V" in data.files else None
    return policy, V


def make_play_env(cfg: EnvConfig, render_mode: str = "ansi") -> gym.Env:
    """
    Erstellt eine FrozenLake-Env mit render_mode=ansi (für Terminal-Frames).
    Nutzt die Werte aus EnvConfig, ohne andere Dateien zu ändern.
    """
    kwargs = {"is_slippery": cfg.is_slippery, "render_mode": render_mode}
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


def render_frame(env: gym.Env, episode: int, step: int, action: int | None,
                 reward: float | None, total_reward: float, sleep: float,
                 manual: bool):
    # In Gymnasium toy_text liefert render() bei render_mode="ansi" einen String
    frame = env.render()
    clear()
    print(f"Episode {episode} | Step {step}")
    if action is not None:
        arrow = ACTION_ARROWS.get(int(action), "?")
        print(f"Action: {ACTION_NAMES[int(action)]} {arrow}")
    if reward is not None:
        print(f"Reward: {reward:.3f} | Return so far: {total_reward:.3f}")
    print("-" * 28)
    print(str(frame))
    print("-" * 28)
    if manual:
        input("weiter mit [Enter] …")
    else:
        time.sleep(max(0.0, sleep))


def run_episode(env: gym.Env, policy: np.ndarray, episode_idx: int,
                sleep: float = 0.4, manual: bool = False,
                max_steps: int = 10_000) -> tuple[float, int, bool]:
    obs, _ = env.reset()
    total_r = 0.0
    done = False
    steps = 0

    # initialen Frame zeigen
    render_frame(env, episode_idx, steps, action=None, reward=None,
                 total_reward=total_r, sleep=sleep, manual=manual)

    while not done and steps < max_steps:
        a = int(policy[obs])
        obs, r, done, truncated, _ = env.step(a)
        total_r += r
        steps += 1
        render_frame(env, episode_idx, steps, action=a, reward=r,
                     total_reward=total_r, sleep=sleep, manual=manual)
        if truncated:
            break
    return total_r, steps, done


def build_policy(args, env_cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Liefert (policy, V). Wenn --load angegeben ist, wird von npz geladen,
    sonst per Value Iteration berechnet.
    """
    if args.load:
        policy, V = load_policy_npz(args.load)
        return policy, V
    # Compute via Value Iteration (deterministisch & reproduzierbar)
    train_cfg = TrainConfig(
        gamma=args.gamma,
        theta=args.theta,
        max_iters=args.max_iters,
        eval_every=0,
        eval_episodes=0,
        eval_seed=env_cfg.seed,
        verbose=False,
    )
    EnvBuilder.set_global_seed(env_cfg.seed)
    # Für Training brauchen wir kein render_mode
    train_env = gym.make(
        "FrozenLake-v1",
        desc=env_cfg.desc,
        map_name=env_cfg.map_name,
        is_slippery=env_cfg.is_slippery,
    )
    train_env.reset(seed=env_cfg.seed)
    try:
        train_env.action_space.seed(env_cfg.seed)
        train_env.observation_space.seed(env_cfg.seed)
    except Exception:
        pass

    agent = ValueIterationAgent(train_cfg, writer=None)
    V = agent.fit(train_env)
    policy = agent.extract_policy(train_env, V)
    return policy, V


def parse_args():
    p = argparse.ArgumentParser(description="Play a policy on FrozenLake with visible steps.")
    # Policy:
    p.add_argument("--load", type=str, default=None,
                   help="Pfad zu .npz mit 'policy' (und optional 'V').")
    # Env:
    p.add_argument("--slippery", action="store_true", help="is_slippery=True (Standard False).")
    p.add_argument("--map-name", type=str, default=None, help="z.B. 4x4 oder 8x8 (exclusive zu custom map).")
    p.add_argument("--custom-map", type=str, nargs="*", default=["SFFF", "FHFH", "FFFH", "HFFG"],
                   help="Eigene Map als Liste von Strings (Standard ist Beispielmap).")
    p.add_argument("--seed", type=int, default=12345, help="Seed.")
    # VI params (nur relevant, wenn nicht --load):
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--theta", type=float, default=1e-8)
    p.add_argument("--max-iters", type=int, default=10_000)
    # Play:
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.4, help="Sekunden Pause zwischen Schritten.")
    p.add_argument("--manual", action="store_true", help="Manuell weiter (Enter) statt sleep.")
    return p.parse_args()


def main():
    args = parse_args()

    # Env-Konfiguration
    desc = None if args.map_name else args.custom_map
    env_cfg = EnvConfig(
        desc=desc,
        map_name=args.map_name,
        is_slippery=args.slippery,
        seed=args.seed
    )

    # Policy laden oder berechnen
    policy, V = build_policy(args, env_cfg)

    # Spiel-Env mit sichtbarer Ausgabe (ANSI)
    EnvBuilder.set_global_seed(env_cfg.seed)
    env = make_play_env(env_cfg, render_mode="ansi")

    # Episoden abspielen
    wins = 0
    total_ret = 0.0
    total_steps = 0
    for ep in range(1, args.episodes + 1):
        ret, steps, done = run_episode(env, policy, ep, sleep=args.sleep, manual=args.manual)
        total_ret += ret
        total_steps += steps
        wins += int(done and ret > 0)
        # kurze Pause zwischen Episoden
        if not args.manual:
            time.sleep(0.6)

    clear()
    print("=== Zusammenfassung ===")
    print(f"Episoden:         {args.episodes}")
    print(f"Success-Rate:     {wins}/{args.episodes} = {wins/args.episodes:.2%}")
    print(f"Ø Return:         {total_ret/args.episodes:.3f}")
    print(f"Ø Schritte:       {total_steps/args.episodes:.1f}")
    if V is not None:
        n = int(np.sqrt(len(V)))
        print("\nValue-Funktion (reshaped):")
        print(V.reshape(n, n))


if __name__ == "__main__":
    main()
