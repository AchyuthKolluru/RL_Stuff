import os
# Make sure this is set BEFORE importing mujoco/env
os.environ.setdefault("MUJOCO_GL", "glfw")
# If you previously exported MUJOCO_EGL, unset it for windowed rendering
if "MUJOCO_EGL" in os.environ:
    os.environ.pop("MUJOCO_EGL")

import torch
import argparse
import numpy as np
from pathlib import Path
import multiprocessing as mp

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from env_g1_inspire_can import G1InspireCanGrasp


class RenderCallback(BaseCallback):
    """
    Renders the first environment periodically.
    For SubprocVecEnv, this uses env_method; for DummyVecEnv, it calls env.render() directly.
    """
    def __init__(self, render_every=1):
        super().__init__()
        self.render_every = render_every

    def _on_step(self) -> bool:
        if self.n_calls % self.render_every != 0:
            return True
        try:
            # Works for both DummyVecEnv and SubprocVecEnv
            if hasattr(self.training_env, "env_method"):
                self.training_env.env_method("render", indices=0)
            else:
                # Fallback (DummyVecEnv)
                env0 = self.training_env.envs[0]
                env0.render()
        except Exception as e:
            print(f"[RenderCallback] render failed: {e}")
        return True


def make_env(scene_xml, hand, seed, rank, max_steps=400, render_mode="none"):
    """Factory that creates a single env; exceptions are bubbled up for clear logs."""
    def _thunk():
        try:
            env = G1InspireCanGrasp(
                scene_xml_path=str(scene_xml),
                hand=hand,
                render_mode=render_mode,
                max_steps=max_steps,
                randomize_init=True,
            )
            env.reset(seed=seed + rank)
            return env
        except Exception as e:
            import traceback, sys
            print(f"\n[Worker {rank}] Failed to create env:\n{traceback.format_exc()}\n",
                  file=sys.stderr, flush=True)
            raise
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    # Choose which model to use (legacy 3-finger scene vs new Inspire FTP 5-finger)
    parser.add_argument("--robot",
                        choices=["legacy", "ftx"],
                        default="ftx",
                        help="Which packaged XML to use: 'legacy' (3-finger scene) or 'ftx' (Inspire FTP 5-finger).")
    # Optional explicit XML path (overrides --robot)
    parser.add_argument("--xml", type=str, default=None,
                        help="Optional path to a specific XML scene file; overrides --robot.")
    # (Kept for backward compatibility… but --xml / --robot take precedence)
    parser.add_argument("--scene", type=str,
                        default="g1_inspire_can_grasp/assets/scene_g1_inspire_can.xml",
                        help="Legacy scene path (ignored if --xml is provided; for 'legacy' mode).")

    parser.add_argument("--hand", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--logdir", type=str, default="logs/g1_inspire_can_sac")
    parser.add_argument("--checkpoint_every_steps", type=int, default=50_000)
    parser.add_argument("--render_mode", type=str, default="none", choices=["none", "human"])

    args = parser.parse_args()

    # Resolve which XML to load
    script_dir = Path(__file__).resolve().parent
    assets_dir = script_dir / "g1_inspire_can_grasp" / "assets"
    legacy_default = assets_dir / "scene_g1_inspire_can.xml"
    ftx_default    = assets_dir / "InspireFTX.xml"

    if args.xml is not None:
        scene_abs = Path(args.xml).expanduser().resolve()
    else:
        if args.robot == "legacy":
            # use the packaged legacy scene (3-finger)
            scene_abs = (Path(args.scene).expanduser().resolve()
                         if args.scene else legacy_default)
        else:
            # use the new Inspire FTP (5-finger) scene
            scene_abs = ftx_default

    if not scene_abs.exists():
        raise FileNotFoundError(f"Cannot find XML at: {scene_abs}")

    print(f"[train_sac] Using scene XML: {scene_abs}")

    # Human rendering must run in the main process → force DummyVec with 1 env
    if args.render_mode == "human" and args.num_envs != 1:
        print("[train_sac] NOTE: --render_mode human requires a single env in main process."
              f" Overriding --num_envs {args.num_envs} → 1.")
        args.num_envs = 1

    # Best to set spawn inside main on POSIX
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    torch.set_num_threads(1)
    os.makedirs(args.logdir, exist_ok=True)

    # Build vectorized env
    if args.num_envs <= 1:
        env = DummyVecEnv([make_env(scene_abs, args.hand, seed=42, rank=0,
                                    render_mode=args.render_mode)])
    else:
        env_fns = [make_env(scene_abs, args.hand, seed=42, rank=i,
                            render_mode=args.render_mode)
                   for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns, start_method="spawn")

    vec_env = VecMonitor(env, filename=None)

    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,          # was 3e-4
        buffer_size=800_000,         # more replay → smoother
        batch_size=512,              # smaller updates
        tau=0.01,                    # slower target network updates
        gamma=0.995,                 # slightly longer horizon
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.1",         # a bit less exploration pressure
        target_update_interval=1,
        verbose=1,
        tensorboard_log=args.logdir,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
    )

    save_freq = max(args.checkpoint_every_steps // max(args.num_envs, 1), 1)
    ckpt_cb = CheckpointCallback(save_freq=save_freq,
                                 save_path=args.logdir,
                                 name_prefix="sac")

    # Render every step when human mode; otherwise render rarely or not at all
    callback = RenderCallback(render_every=1 if args.render_mode == "human" else 0)

    model.learn(total_timesteps=args.total_timesteps,
                callback=[cb for cb in [ckpt_cb, callback] if cb],
                log_interval=10)
    model.save(os.path.join(args.logdir, "final_sac"))

    vec_env.close()


if __name__ == "__main__":
    main()

    # python train_sac.py --robot ftx --num_envs 1 --render_mode human --total_timesteps 1000000