# eval_sac.py
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.pop("MUJOCO_EGL", None)

import argparse, numpy as np, cv2
from pathlib import Path
from stable_baselines3 import SAC

from env_g1_inspire_can import G1InspireCanGrasp

def make_env(xml_path: Path, hand: str, max_steps: int, render_mode: str, randomize_init: bool, rand_scale: float):
    env = G1InspireCanGrasp(
        scene_xml_path=str(xml_path),
        hand=hand,
        render_mode=render_mode,
        max_steps=max_steps,
        randomize_init=randomize_init,
        init_rand_scale=rand_scale,
        enable_headcam=True,                # turn on head-cam in sim
        headcam_size=(640, 480),
        headcam_for_eval_only=False,        # allow fetch during eval loop
        auto_choose_nearer_side=False,      # FIXED side per chosen hand
    )
    return env

def run_rollouts(model_path: Path, xml_path: Path, episodes: int, hand: str,
                 max_steps: int, render_mode: str, randomize_init: bool, rand_scale: float,
                 deterministic: bool, seed: int,
                 show_headcam: bool, yolo_weights: str|None):
    env = make_env(xml_path, hand, max_steps, render_mode, randomize_init, rand_scale)

    yolo = None
    if show_headcam and yolo_weights:
        try:
            from ultralytics import YOLO
            yolo = YOLO(yolo_weights)
            print(f"[eval] YOLO loaded: {yolo_weights}")
        except Exception as e:
            print(f"[eval] Could not load YOLO ({e}) – continuing without it.")

    print(f"[eval] Loading model: {model_path}")
    model = SAC.load(str(model_path))

    ep_rewards, successes, dists, xy_dists = [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed+ep)
        done = trunc = False
        total_r = 0.0; steps = 0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, trunc, info = env.step(action)
            total_r += float(reward); steps += 1

            if render_mode == "human":
                env.render()

            # optional head-cam preview
            if show_headcam:
                frame = env.get_headcam_rgb()
                if frame is not None:
                    # overlay projected can pixel if available
                    uv = info.get("headcam_uv", None)
                    if uv is not None:
                        u,v = int(uv[0]), int(uv[1])
                        cv2.circle(frame, (u,v), 8, (0,0,255), -1)
                        dist = info.get("headcam_dist_m", None)
                        if dist is not None:
                            cv2.putText(frame, f"{dist:.2f} m", (u+10, max(20,v-10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    if yolo is not None:
                        try:
                            res = yolo.predict(source=frame, conf=0.3, imgsz=max(frame.shape[:2]),
                                               device=0, verbose=False)[0]
                            frame = res.plot()  # draws boxes on BGR
                        except Exception as e:
                            cv2.putText(frame, f"YOLO error: {e}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv2.imshow("HeadCam", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        done = True; trunc = True

            if isinstance(info, dict):
                if "d_target" in info: dists.append(info["d_target"])
                if "radial" in info:   xy_dists.append(info["radial"])

        ep_rewards.append(total_r)
        successes.append(float(info.get("is_success", False)))
        print(f"[eval] ep {ep+1}/{episodes}: R={total_r:.2f}  success={bool(info.get('is_success', False))}  steps={steps}")

    env.close()
    if show_headcam:
        cv2.destroyAllWindows()

    sr = np.mean(successes) if successes else 0.0
    print("\n=== Evaluation Summary ===")
    print(f"episodes: {episodes}")
    print(f"avg reward: {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"success rate: {sr*100:.1f}%")
    if dists:    print(f"mean target distance (m): {np.mean(dists):.4f}  (lower is better)")
    if xy_dists: print(f"mean radial distance wrt can (m): {np.mean(xy_dists):.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--xml", default=None, help="Path to *copied* XML with head_cam (e.g., assets/InspireFTX_headcam.xml)")
    ap.add_argument("--hand", default="right", choices=["right","left"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=750)
    ap.add_argument("--render_mode", choices=["human","none"], default="human")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--randomize_init", action="store_true", default=True)
    ap.add_argument("--rand_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--show_headcam", action="store_true", default=False)
    ap.add_argument("--yolo_weights", default=None, help="Optional yolov8 *.pt for overlay")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.xml is None:
        xml_path = script_dir / "g1_inspire_can_grasp" / "assets" / "InspireFTX_headcam.xml"
    else:
        xml_path = Path(args.xml).expanduser().resolve()

    run_rollouts(
        model_path=Path(args.model).expanduser().resolve(),
        xml_path=xml_path,
        episodes=args.episodes,
        hand=args.hand,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
        randomize_init=args.randomize_init,
        rand_scale=args.rand_scale,
        deterministic=args.deterministic,
        seed=args.seed,
        show_headcam=args.show_headcam,
        yolo_weights=args.yolo_weights,
    )


# Use your “best so far” model:
# python eval_sac.py \
#   --model logs/g1_inspire_can_sac/sac_5348288_steps.zip \
#   --xml   g1_inspire_can_grasp/assets/InspireFTX_headcam.xml \
#   --episodes 5 --hand right --render_mode human --randomize_init \
#   --show_headcam \
#   --yolo_weights /path/to/yolov8s/weights/best.pt