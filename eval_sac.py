import os
os.environ.setdefault("MUJOCO_GL", "glfw")
# If you ever exported MUJOCO_EGL for headless runs, make sure it is unset for windowed rendering:
os.environ.pop("MUJOCO_EGL", None)

import argparse
import numpy as np
from pathlib import Path

from stable_baselines3 import SAC

# import your env class
from env_g1_inspire_can import G1InspireCanGrasp


def make_env(xml_path: Path,
             hand: str = "right",
             max_steps: int = 300,
             render_mode: str = "human",
             randomize_init: bool = True,
             rand_scale: float = 1.0):
    env = G1InspireCanGrasp(
        scene_xml_path=str(xml_path),
        hand=hand,
        render_mode=render_mode,
        max_steps=max_steps,
        randomize_init=randomize_init,
    )
    # optional: curriculum / randomization intensity if your env exposes it
    if hasattr(env, "set_randomization_scale"):
        env.set_randomization_scale(rand_scale)
    return env


def run_rollouts(model_path: Path,
                 xml_path: Path,
                 episodes: int = 5,
                 hand: str = "right",
                 max_steps: int = 750,
                 render_mode: str = "human",
                 randomize_init: bool = True,
                 rand_scale: float = 1.0,
                 deterministic: bool = True,
                 seed: int = 123):
    # Single, plain env (no VecEnv) makes human rendering simplest.
    env = make_env(xml_path, hand, max_steps, render_mode, randomize_init, rand_scale)

    print(f"[eval] Loading model: {model_path}")
    model = SAC.load(str(model_path))  # env not required for predict()

    ep_rewards = []
    successes = []
    dists = []
    xy_dists = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        trunc = False
        total_r = 0.0
        steps = 0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, trunc, info = env.step(action)
            if render_mode == "human":
                env.render()
            total_r += float(reward)
            steps += 1
            # env.render() is called inside env when render_mode="human"

            # Collect some diagnostics if present
            if isinstance(info, dict):
                if "d_target" in info:
                    dists.append(info["d_target"])
                if "radial" in info:
                    xy_dists.append(info["radial"])

        ep_rewards.append(total_r)
        successes.append(float(info.get("is_success", False)))
        print(f"[eval] ep {ep+1}/{episodes}: R={total_r:.2f}  "
              f"success={bool(info.get('is_success', False))}  steps={steps}")

    env.close()

    # Summary
    sr = np.mean(successes) if successes else 0.0
    print("\n=== Evaluation Summary ===")
    print(f"episodes: {episodes}")
    print(f"avg reward: {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"success rate: {sr*100:.1f}%")
    if dists:
        print(f"mean target distance (m): {np.mean(dists):.4f}  (lower is better)")
    if xy_dists:
        print(f"mean radial distance wrt can (m): {np.mean(xy_dists):.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to sac_XXXX_steps.zip or final_sac.zip")
    ap.add_argument("--xml", required=False, default=None,
                    help="Path to InspireFTX.xml (defaults to the one next to your train script)")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=750)
    ap.add_argument("--render_mode", choices=["human", "none"], default="human")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--randomize_init", action="store_true", default=True)
    ap.add_argument("--rand_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    # Default XML if not provided: the one you used in training
    if args.xml is None:
        xml_default = script_dir / "g1_inspire_can_grasp" / "assets" / "InspireFTX.xml"
    else:
        xml_default = Path(args.xml).expanduser().resolve()

    run_rollouts(
        model_path=Path(args.model).expanduser().resolve(),
        xml_path=xml_default,
        episodes=args.episodes,
        hand=args.hand,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
        randomize_init=args.randomize_init,
        rand_scale=args.rand_scale,
        deterministic=args.deterministic,
        seed=args.seed,
    )



# python eval_sac.py   --model logs/g1_inspire_can_sac/sac_4998400_steps.zip   --episodes 10   --hand right   --render_mode human   --randomize_init   --rand_scale 1.0