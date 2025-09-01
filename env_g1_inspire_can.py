import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
from mujoco import MjModel, MjData

# Optional viewer (MuJoCo >= 2.3.3/2.3.5)
try:
    import mujoco.viewer  # noqa: F401
    HAVE_MJ_VIEWER = True
except Exception:
    HAVE_MJ_VIEWER = False


def _site_quat(data, sid):
    """Get site quaternion across MuJoCo versions (xquat newer, xmat→quat fallback)."""
    if hasattr(data, "site_xquat"):
        return data.site_xquat[sid].copy()
    R = data.site_xmat[sid].reshape(3, 3)
    q = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def _all_actuator_names(model):
    names = []
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if nm is not None:
            names.append(nm)
    return names


def find_actuators_by_name(model, names_wanted):
    """Return actuator ids whose name is in names_wanted (exact match)."""
    name_set = set(names_wanted)
    ids = []
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if nm in name_set:
            ids.append(i)
    return sorted(ids)


def named_site_id(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Site '{name}' not found. Add it in XML.")
    return sid


class G1InspireCanGrasp(gym.Env):
    """
    RL env that controls only the hand actuators.
    Works with both 'legacy' and Inspire FTP (5-finger, 12-DOF) scenes if actuator
    names follow the side prefix convention: left_hand_*, right_hand_*.
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self,
                 scene_xml_path,
                 render_mode="none",
                 hand="right",
                 hand_names=None,        # optional explicit list of actuator names
                 max_steps=400,
                 target_lift=0.03,
                 randomize_init=True):
        super().__init__()

        if not os.path.isfile(scene_xml_path):
            raise FileNotFoundError(scene_xml_path)

        self.model = MjModel.from_xml_path(scene_xml_path)
        self.data = MjData(self.model)

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.target_lift = target_lift
        self.randomize_init = randomize_init
        self.step_count = 0

        # --------- Select hand actuators ----------
        side = "right" if hand.lower().startswith("r") else "left"

        # If user passed explicit names, use them; otherwise auto-detect, then fallback.
        if hand_names is not None:
            candidate_names = list(hand_names)
        else:
            # auto: everything starting with "<side>_hand_"
            all_names = _all_actuator_names(self.model)
            candidate_names = [n for n in all_names if n.startswith(f"{side}_hand_")]
            candidate_names.sort()

            # If auto-detect found nothing, fallback to known sets (Inspire FTP 12-DOF)
            if not candidate_names:
                if side == "right":
                    candidate_names = [
                        "right_hand_thumb_0_joint","right_hand_thumb_1_joint",
                        "right_hand_thumb_2_joint","right_hand_thumb_3_joint",
                        "right_hand_index_0_joint","right_hand_index_1_joint",
                        "right_hand_middle_0_joint","right_hand_middle_1_joint",
                        "right_hand_ring_0_joint","right_hand_ring_1_joint",
                        "right_hand_pinky_0_joint","right_hand_pinky_1_joint",
                    ]
                else:
                    candidate_names = [
                        "left_hand_thumb_0_joint","left_hand_thumb_1_joint",
                        "left_hand_thumb_2_joint","left_hand_thumb_3_joint",
                        "left_hand_index_0_joint","left_hand_index_1_joint",
                        "left_hand_middle_0_joint","left_hand_middle_1_joint",
                        "left_hand_ring_0_joint","left_hand_ring_1_joint",
                        "left_hand_pinky_0_joint","left_hand_pinky_1_joint",
                    ]

        self.hand_actuator_ids = find_actuators_by_name(self.model, candidate_names)
        if len(self.hand_actuator_ids) == 0:
            # Helpful error printout with available names
            avail = _all_actuator_names(self.model)
            raise RuntimeError(
                f"No actuators found for names: {candidate_names}\n"
                f"Available actuators: {avail}"
            )

        # Map actuators → joints (one joint per actuator expected)
        self.act_to_joint = np.array(
            [int(self.model.actuator_trnid[a, 0]) for a in self.hand_actuator_ids],
            dtype=int
        )
        self.jnt_qposadr = self.model.jnt_qposadr[self.act_to_joint]
        self.jnt_dofadr  = self.model.jnt_dofadr[self.act_to_joint]
        self.jnt_range   = self.model.jnt_range[self.act_to_joint].copy()

        # PD gains + torque limits (soft clamp)
        self.kp = np.full(len(self.hand_actuator_ids), 20.0, dtype=np.float64)
        self.kd = np.full(len(self.hand_actuator_ids), 1.0, dtype=np.float64)

        self.hand_actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            for a in self.hand_actuator_ids
        ]
        self.torque_limit = np.minimum(
            np.array([2.45 if ("thumb_0" in n) else 1.4 for n in self.hand_actuator_names],
                     dtype=np.float64),
            1.0
        )

        # Action scaling and action space
        self.action_scale = 0.03
        self.des_q = self.data.qpos[self.jnt_qposadr].copy()
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(len(self.hand_actuator_ids),),
                                       dtype=np.float32)

        # Observation space: qpos, qvel, can pose (pos+quat), relative palm->can
        self.can_sid = named_site_id(self.model, "can_site")
        # If can is kinematic (no free joint), we won't randomize or lift it
        try:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
            self.can_free_joint = j if j >= 0 else None
        except Exception:
            self.can_free_joint = None

        palm_name = "palm_site_right" if side == "right" else "palm_site_left"
        self.palm_sid = named_site_id(self.model, palm_name)

        obs_dim = (len(self.jnt_qposadr) * 2) + 3 + 4 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

        self.viewer = None

    # ---------------- utils ----------------
    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        qpos = self.data.qpos[self.jnt_qposadr].copy()
        qvel = self.data.qvel[self.jnt_dofadr].copy()
        can_pos = self.data.site_xpos[self.can_sid].copy()
        can_quat = _site_quat(self.data, self.can_sid)
        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        rel_vec = can_pos - palm_pos
        return np.concatenate([qpos, qvel, can_pos, can_quat, rel_vec]).astype(np.float32)

    def _apply_action(self, action):
        action = np.clip(action.astype(np.float64), -1.0, 1.0)
        self.des_q = np.clip(self.des_q + self.action_scale * action,
                             self.jnt_range[:, 0], self.jnt_range[:, 1])
        q  = self.data.qpos[self.jnt_qposadr].astype(np.float64)
        qd = self.data.qvel[self.jnt_dofadr].astype(np.float64)
        tau = self.kp * (self.des_q - q) - self.kd * qd
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.hand_actuator_ids] = tau

    def _randomize(self):
        # If we don't have a free joint, the can is static in the XML → nothing to randomize.
        if self.can_free_joint is None:
            return
        adr = self.model.jnt_qposadr[self.can_free_joint]
        # Place near palm (tweak for your robot)
        x = 0.42 + np.random.uniform(-0.02, 0.02)
        y = -0.03 + np.random.uniform(-0.02, 0.02)
        z = 1.03 + np.random.uniform(-0.02, 0.02)
        self.data.qpos[adr:adr+3] = np.array([x, y, z], dtype=np.float64)

        # Small tilt
        axis = np.random.randn(3); axis /= (np.linalg.norm(axis) + 1e-8)
        angle = np.random.uniform(-0.15, 0.15)
        s = math.sin(angle/2.0)
        self.data.qpos[adr+3:adr+7] = np.array([math.cos(angle/2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)

        # Zero velocities
        dof = self.model.jnt_dofadr[self.can_free_joint]
        self.data.qvel[dof:dof+6] = 0.0

    # ---------------- gym api ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.des_q = self.data.qpos[self.jnt_qposadr].copy()
        if self.randomize_init:
            self._randomize()
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._apply_action(action)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward, success = self._compute_reward_and_success()
        self.step_count += 1
        terminated = success
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {"is_success": success}

    def _compute_reward_and_success(self):
        mujoco.mj_forward(self.model, self.data)

        can_pos = self.data.site_xpos[self.can_sid]
        palm_pos = self.data.site_xpos[self.palm_sid]
        dist = np.linalg.norm(can_pos - palm_pos)

        # Count contacts with the can
        touch_bonus = 0.0
        touched = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 >= 0 and c.geom2 >= 0:
                name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                if name1 == "can_geom" or name2 == "can_geom":
                    touched = True
                    touch_bonus += 0.005

        ctrl_penalty = 1e-3 * float(np.sum(self.data.ctrl[self.hand_actuator_ids] ** 2))

        # If the can is free, keep the old lift term; otherwise 0
        height_bonus = 0.0
        if self.can_free_joint is not None:
            z = self.data.qpos[self.model.jnt_qposadr[self.can_free_joint] + 2]
            height_bonus = 6.0 * max(0.0, z - 1.02)

        # Distance shaping encourages approaching the can; touch gives extra reward
        reward = 2.0 * (1.0 / (0.05 + dist)) + touch_bonus + height_bonus - ctrl_penalty

        # Success: for STATIC can → close & touching. For FREE can → lifted & close.
        if self.can_free_joint is None:
            success = (dist < 0.04) and touched
        else:
            success = (dist < 0.05) and (z > 1.02 + self.target_lift)

        return reward, bool(success)

    def render(self):
        if self.render_mode != "human":
            return
        if not HAVE_MJ_VIEWER:
            raise RuntimeError(
                "mujoco.viewer not available. Install/upgrade MuJoCo (e.g., pip install 'mujoco>=2.3.5') "
                "and ensure MUJOCO_GL=glfw on a machine with a display."
            )
        if self.viewer is None:
            # first call opens the window
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            # subsequent calls just sync the already-open window
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None