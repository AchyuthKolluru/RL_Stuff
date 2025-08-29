import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from env_g1_inspire_can import G1InspireCanGrasp

def make_env(scene_xml, hand, seed, rank):
    def _thunk():
        env = G1InspireCanGrasp(scene_xml_path=scene_xml,
                                render_mode="none",
                                hand=hand,
                                max_steps=400,
                                randomize_init=True)
        env.reset(seed=seed + rank)
        return env
    return _thunk

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=str, default="RL-shenanigans/g1_inspire_can_grasp/assets/scene_g1_inspire_can.xml")
    p.add_argument("--hand", type=str, default="right", choices=["right","left"])
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--logdir", type=str, default="logs/g1_inspire_can_sac")
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    env_fns = [make_env(args.scene, args.hand, seed=42, rank=i) for i in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=None)

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
        policy_kwargs=dict(net_arch=[512, 512, 256])
    )

    ckpt_cb = CheckpointCallback(save_freq=50_000 // args.num_envs, save_path=args.logdir, name_prefix="sac")
    model.learn(total_timesteps=args.total_steps, log_interval=10, callback=[ckpt_cb])
    model.save(os.path.join(args.logdir, "final_sac"))