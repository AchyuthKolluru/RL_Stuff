# train_sac.py
import os
# Make sure this is set BEFORE importing mujoco/env
os.environ.setdefault("MUJOCO_GL", "glfw")
# If you previously exported MUJOCO_EGL, unset it for windowed rendering
os.environ.pop("MUJOCO_EGL", None)

import argparse
import multiprocessing as mp
from pathlib import Path
from collections import deque

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from env_g1_inspire_can import G1InspireCanGrasp


class RenderCallback(BaseCallback):
    """
    Renders env 0 periodically.
    Works with both DummyVecEnv and SubprocVecEnv via env_method.
    """
    def __init__(self, render_every: int = 1):
        super().__init__()
        self.render_every = int(render_every)

    def _on_step(self) -> bool:
        if self.render_every <= 0:
            return True
        if self.n_calls % self.render_every != 0:
            return True
        try:
            if hasattr(self.training_env, "env_method"):
                self.training_env.env_method("render", indices=0)
            else:
                self.training_env.envs[0].render()
        except Exception as e:
            print(f"[RenderCallback] render failed: {e}")
        return True


class InfoStatsCallback(BaseCallback):
    """
    Gathers lightweight stats from `infos` and logs moving averages.
    """
    def __init__(self, log_every: int = 2048):
        super().__init__()
        self.log_every = int(log_every)
        self.succ = deque(maxlen=10000)
        self.d_target = deque(maxlen=10000)
        self.radial = deque(maxlen=10000)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "is_success" in info:
                self.succ.append(float(info["is_success"]))
            if "d_target" in info:
                self.d_target.append(float(info["d_target"]))
            if "radial" in info:
                self.radial.append(float(info["radial"]))

        if self.num_timesteps % self.log_every == 0:
            if self.succ:
                self.logger.record("train/success_rate", np.mean(self.succ))
            if self.d_target:
                self.logger.record("train/mean_d_target", np.mean(self.d_target))
            if self.radial:
                self.logger.record("train/mean_radial", np.mean(self.radial))
        return True


class CurriculumCallback(BaseCallback):
    """
    Linearly ramps env randomization scale from start→end across a number of timesteps.
    """
    def __init__(self, start_scale: float, end_scale: float, total_steps: int, log_key: str = "train/rand_scale"):
        super().__init__()
        self.s0 = float(start_scale)
        self.s1 = float(end_scale)
        self.T = max(int(total_steps), 1)
        self.log_key = log_key

    def _on_step(self) -> bool:
        t = min(self.num_timesteps, self.T)
        alpha = t / self.T
        scale = (1 - alpha) * self.s0 + alpha * self.s1
        try:
            # Broadcast to all envs (works for both DummyVecEnv/SubprocVecEnv)
            self.training_env.env_method("set_randomization_scale", float(scale))
        except Exception as e:
            print(f"[Curriculum] set_randomization_scale failed: {e}")
        self.logger.record(self.log_key, float(scale))
        return True


def make_env(scene_xml, hand, seed, rank, max_steps=400, render_mode="none", headcam_train=False):
    """Factory for one env. Exceptions bubble up for clear logs."""
    def _thunk():
        # Keep training light: head-cam off by default (no offscreen rendering cost)
        env = G1InspireCanGrasp(
            scene_xml_path=str(scene_xml),
            hand=hand,
            render_mode=render_mode,
            max_steps=max_steps,
            randomize_init=True,
            enable_headcam=bool(headcam_train),
            headcam_for_eval_only=True,      # even if enabled, don't render frames each step
            auto_choose_nearer_side=False,   # fixed side per chosen hand
        )
        env.reset(seed=seed + rank)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    # Which packaged XML to use (kept close to your flow)
    parser.add_argument("--robot", choices=["legacy", "ftx"], default="ftx",
                        help="Use 'legacy' (3-finger scene) or 'ftx' (Inspire FTX).")
    # Optional explicit XML path (overrides --robot)
    parser.add_argument("--xml", type=str, default=None,
                        help="Explicit path to an XML scene; overrides --robot.")
    # Legacy scene (ignored if --xml provided; kept for compatibility)
    parser.add_argument("--scene", type=str,
                        default="g1_inspire_can_grasp/assets/scene_g1_inspire_can.xml",
                        help="Legacy scene path (ignored if --xml is provided; used if --robot legacy).")

    parser.add_argument("--hand", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--logdir", type=str, default="logs/g1_inspire_can_sac")
    parser.add_argument("--checkpoint_every_steps", type=int, default=50_000)
    parser.add_argument("--render_mode", type=str, default="none", choices=["none", "human"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--headcam_train", action="store_true", default=False,
                        help="Enable head-cam model fields during training (no RGB rendering).")

    # Curriculum / DR knobs
    parser.add_argument("--curriculum_start", type=float, default=0.4,
                        help="Initial randomization scale.")
    parser.add_argument("--curriculum_end", type=float, default=1.0,
                        help="Final randomization scale.")
    parser.add_argument("--curriculum_steps", type=int, default=300_000,
                        help="Timesteps over which to ramp start→end.")
    parser.add_argument("--log_every", type=int, default=2048,
                        help="Steps between info-stats logs.")

    args = parser.parse_args()

    # Resolve XML
    script_dir = Path(__file__).resolve().parent
    assets_dir = script_dir / "g1_inspire_can_grasp" / "assets"
    legacy_default = assets_dir / "scene_g1_inspire_can.xml"
    ftx_default    = assets_dir / "InspireFTX.xml"               # close to your original
    # If you want the head-cam/free-can in training, point to your copy:
    headcam_default = assets_dir / "InspireFTX_headcam.xml"      # optional

    if args.xml is not None:
        scene_abs = Path(args.xml).expanduser().resolve()
    else:
        if args.robot == "legacy":
            scene_abs = Path(args.scene).expanduser().resolve() if args.scene else legacy_default
        else:
            # Prefer headcam version if present; otherwise fall back to base FTX
            scene_abs = headcam_default if headcam_default.exists() else ftx_default

    if not scene_abs.exists():
        raise FileNotFoundError(f"Cannot find XML at: {scene_abs}")

    print(f"[train_sac] Using scene XML: {scene_abs}")

    # Human rendering has to run in main process → single env
    if args.render_mode == "human" and args.num_envs != 1:
        print("[train_sac] NOTE: --render_mode human requires a single env in main process."
              f" Overriding --num_envs {args.num_envs} → 1.")
        args.num_envs = 1

    # POSIX best practice
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    os.makedirs(args.logdir, exist_ok=True)

    # Build vectorized env
    if args.num_envs <= 1:
        env = DummyVecEnv([make_env(scene_abs, args.hand, seed=args.seed, rank=0,
                                    max_steps=args.max_steps, render_mode=args.render_mode,
                                    headcam_train=args.headcam_train)])
    else:
        env_fns = [make_env(scene_abs, args.hand, seed=args.seed, rank=i,
                            max_steps=args.max_steps, render_mode=args.render_mode,
                            headcam_train=args.headcam_train)
                   for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns, start_method="spawn")

    vec_env = VecMonitor(env, filename=None)

    # SAC config (kept very close to what you had)
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        buffer_size=800_000,
        batch_size=512,
        tau=0.01,
        gamma=0.995,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.1",
        target_update_interval=1,
        verbose=1,
        tensorboard_log=args.logdir,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
        seed=args.seed,
        device="auto",
    )

    # Checkpointing – align save freq with vectorized stepping
    save_freq = max(args.checkpoint_every_steps // max(args.num_envs, 1), 1)
    ckpt_cb = CheckpointCallback(save_freq=save_freq, save_path=args.logdir, name_prefix="sac")

    # Render every step when human mode; else disable (0)
    render_cb = RenderCallback(render_every=1 if args.render_mode == "human" else 0)

    # Info stats & curriculum
    stats_cb = InfoStatsCallback(log_every=args.log_every)
    curriculum_cb = CurriculumCallback(
        start_scale=args.curriculum_start,
        end_scale=args.curriculum_end,
        total_steps=args.curriculum_steps,
    )

    callbacks = [ckpt_cb, render_cb, stats_cb, curriculum_cb]

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, log_interval=10)
    model.save(os.path.join(args.logdir, "final_sac"))

    vec_env.close()


if __name__ == "__main__":
    main()

# examples:
#   python train_sac.py --robot ftx --num_envs 8  --total_timesteps 1000000
#   python train_sac.py --robot ftx --num_envs 1  --render_mode human --total_timesteps 100000
#   python train_sac.py --xml g1_inspire_can_grasp/assets/InspireFTX_headcam.xml --num_envs 8