import os
import torch
import argparse
import numpy as np
from pathlib import Path
import multiprocessing as mp
from env_g1_inspire_can import G1InspireCanGrasp
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

mp.set_start_method("spawn", force=True)

os.environ.setdefault("MUJOCO_GL", "glfw")

script_dir = Path(__file__).resolve().parent
default_scene = script_dir / "RL-shenanigans" / "g1_inspire_can_grasp" / "assets" / "scene_g1_inspire_can.xml"

class RenderCallback(BaseCallback):
    def __init__(self, render_every=1):
        super().__init__()
        self.render_every = render_every

    def _on_step(self) -> bool:
        if self.n_calls % self.render_every == 0:
            try:
                env0 = self.training_env.envs[0]
                env0.render()  # opens / updates the MuJoCo viewer
            except Exception as e:
                print(f"[RenderCallback] render failed: {e}")
        return True

def make_env(scene_xml, hand, seed, rank, max_steps=400, render_mode="none"):
    """Wrap env ctor to surface errors from worker processes."""
    def _thunk():
        try:
            env = G1InspireCanGrasp(
                scene_xml_path=scene_xml,
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
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=str,
                   default="RL-shenanigans/g1_inspire_can_grasp/assets/scene_g1_inspire_can.xml")
    p.add_argument("--hand", type=str, default="right", choices=["right", "left"])
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--logdir", type=str, default="logs/g1_inspire_can_sac")
    p.add_argument("--checkpoint_every_steps", type=int, default=50_000)
    p.add_argument("--render_mode", type=str, default="none", choices=["none", "human"])
    args = p.parse_args()
    
    scene_abs = Path(args.scene).resolve() if args.scene else default_scene
    scene_abs = scene_abs.resolve()

    if not scene_abs.is_file():
        raise FileNotFoundError(f"Scene XML not found: {scene_abs}")

    scene_abs = os.path.abspath(args.scene)

    torch.set_num_threads(1)

    os.makedirs(args.logdir, exist_ok=True)

    if args.num_envs <= 1:
        env = DummyVecEnv([make_env(scene_abs, args.hand, seed=42, rank=0,
                                    render_mode=args.render_mode)])
    else:
        env_fns = [make_env(scene_abs, args.hand, seed=42, rank=i,
                            render_mode=args.render_mode)
                   for i in range(args.num_envs)]
        # explicitly pass spawn here too
        env = SubprocVecEnv(env_fns, start_method="spawn")

    vec_env = VecMonitor(env, filename=None)

    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=1024,
        tau=0.02,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        tensorboard_log=args.logdir,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
    )

    save_freq = max(args.checkpoint_every_steps // max(args.num_envs, 1), 1)
    ckpt_cb = CheckpointCallback(save_freq=save_freq,
                                 save_path=args.logdir,
                                 name_prefix="sac")
    
    callback = RenderCallback(render_every=1)  # or 5–10 to reduce overhead
    model.learn(total_timesteps=args.total_timesteps, callback=callback, log_interval=10)
    model.save(os.path.join(args.logdir, "final_sac"))

    vec_env.close()


if __name__ == "__main__":
    main()