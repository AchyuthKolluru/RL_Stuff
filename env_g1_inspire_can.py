import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
from mujoco import MjModel, MjData

try:
    import mujoco.viewer  # noqa: F401
    HAVE_MJ_VIEWER = True
except Exception:
    HAVE_MJ_VIEWER = False


# ---------- utils (unchanged) ----------
def _site_quat(data, sid):
    """Return site quaternion (wxyz)."""
    if hasattr(data, "site_xquat"):
        return data.site_xquat[sid].copy()
    R = data.site_xmat[sid].reshape(3, 3)
    q = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q

def named_site_id(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Site '{name}' not found. Add it in XML.")
    return sid

def find_actuators_by_name(model, names_wanted):
    """Return actuator ids for the given list of actuator (i.e., joint) names."""
    name_set = set(names_wanted)
    ids = []
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if nm in name_set:
            ids.append(i)
    return sorted(ids)

def _joint_ids(model, names):
    ids = []
    for n in names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            ids.append(j)
    return ids

def _set_joint_if_exists(model, data, joint_name, value):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if j >= 0:
        adr = model.jnt_qposadr[j]
        data.qpos[adr] = float(value)



class G1InspireCanGrasp(gym.Env):
    """
    SIMPLE arm-only approach:
      - Controls ONLY 4 DOF: shoulder pitch/roll/yaw + elbow
      - Wrist is hard-locked so rubber hand is vertical, thumb-up, palm facing can
      - Reward: get close in XY to a standoff from the can, correct lateral side, face can, no touching
      - Other arm is frozen
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        scene_xml_path: str,
        render_mode: str = "none",
        hand: str = "right",          # "right" or "left"
        max_steps: int = 300,
        randomize_init: bool = False, # keep off for deterministic approach unless you want it
        standoff: float | None = None,# if None -> can_radius + 0.012
        standoff_tol: float = 0.01,   # success band
        side_margin: float = 0.02,    # keep on correct lateral side of can
        touch_penalty: float = 4.0,   # penalty on any can contact
        kp: float = 10.0,             # gentle gains
        kd: float = 3.0,              # extra damping for slow motion
        action_scale: float = 0.003,  # VERY slow per-step increments
    ):
        super().__init__()
        if not os.path.isfile(scene_xml_path):
            raise FileNotFoundError(scene_xml_path)

        self.model = MjModel.from_xml_path(scene_xml_path)
        self.data = MjData(self.model)

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.randomize_init = bool(randomize_init)
        self.side_margin = float(side_margin)
        self.standoff_tol = float(standoff_tol)
        self.touch_penalty = float(touch_penalty)
        self.kp = float(kp)
        self.kd = float(kd)
        self.action_scale = float(action_scale)

        # Which side
        self.right_side = hand.lower().startswith("r")

        # ---------- names ----------
        if self.right_side:
            self.ctrl_joint_names = [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
            ]
            self.wrist_joint_names = [
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
            self.palm_site_name = "palm_site_right"
            self.wrist_body_name = "right_wrist_yaw_link"  # orientation anchor
            self.other_arm_joint_names = [
                "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
                "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
            ]
            # LOCK POSE (thumb up, palm forward/sideways). Adjust if needed.
            self.wrist_lock_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.ctrl_joint_names = [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
            ]
            self.wrist_joint_names = [
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]
            self.palm_site_name = "palm_site_left"
            self.wrist_body_name = "left_wrist_yaw_link"
            self.other_arm_joint_names = [
                "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
                "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
            ]
            self.wrist_lock_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # ---------- resolve ids ----------
        self.ctrl_joint_ids = _joint_ids(self.model, self.ctrl_joint_names)
        if len(self.ctrl_joint_ids) != 4:
            raise RuntimeError("Expected 4 controllable joints; check joint names/actuators in XML.")

        self.ctrl_actuator_ids = np.array(
            find_actuators_by_name(self.model, self.ctrl_joint_names), dtype=int
        )
        if len(self.ctrl_actuator_ids) != 4:
            raise RuntimeError("Actuators for the 4 control joints not found; check <actuator> names.")

        self.ctrl_jnt_qposadr = self.model.jnt_qposadr[self.ctrl_joint_ids]
        self.ctrl_jnt_dofadr = self.model.jnt_dofadr[self.ctrl_joint_ids]
        self.ctrl_jnt_range  = self.model.jnt_range[self.ctrl_joint_ids].copy()

        self.wrist_joint_ids  = _joint_ids(self.model, self.wrist_joint_names)
        self.wrist_qpos_adrs  = np.array([self.model.jnt_qposadr[j] for j in self.wrist_joint_ids], dtype=int)
        self.wrist_dof_adrs   = np.array([self.model.jnt_dofadr[j]  for j in self.wrist_joint_ids], dtype=int)
        self.wrist_qpos_fixed = None

        self.other_joint_ids = _joint_ids(self.model, self.other_arm_joint_names)
        self.other_qpos_adrs = np.array([self.model.jnt_qposadr[j] for j in self.other_joint_ids], dtype=int)
        self.other_dof_adrs  = np.array([self.model.jnt_dofadr[j]  for j in self.other_joint_ids], dtype=int)
        self.other_qpos_fixed = None

        self.can_sid = named_site_id(self.model, "can_site")
        self.palm_sid = named_site_id(self.model, self.palm_site_name)
        self.wrist_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.wrist_body_name)

        try:
            self.can_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom")
        except Exception:
            self.can_geom_id = -1

        # can radius / half-height
        if self.can_geom_id >= 0:
            sz = self.model.geom_size[self.can_geom_id]
            self.can_radius = float(sz[0])
            self.can_half_h = float(sz[1])
        else:
            self.can_radius = 0.03
            self.can_half_h = 0.06

        # default standoff (edge distance in XY)
        self.standoff = float(self.can_radius + 0.012) if (standoff is None) else float(standoff)
        self.min_xy_gap = self.can_radius + 0.006  # safety margin inside barrier

        # spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # obs: qpos(4) + qvel(4) + can_pos(3) + palm->can vec(3)
        obs_dim = 4 + 4 + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # controllers
        self.kp_vec = np.full(4, self.kp, dtype=np.float64)
        self.kd_vec = np.full(4, self.kd, dtype=np.float64)
        self.torque_limit_vec = np.array([12, 10, 8, 8], dtype=np.float64)

        self.des_q = np.zeros(4, dtype=np.float64)
        self.step_count = 0
        self.viewer = None


    # ---------- helpers ----------
    def _record_other_side_fixed_pose(self):
        if self.other_qpos_adrs.size == 0:
            self.other_qpos_fixed = None
            return
        self.other_qpos_fixed = self.data.qpos[self.other_qpos_adrs].copy()

    def _enforce_other_side_fixed(self):
        if self.other_qpos_fixed is None:
            return
        self.data.qpos[self.other_qpos_adrs] = self.other_qpos_fixed
        self.data.qvel[self.other_dof_adrs] = 0.0

    def _record_wrist_fixed_pose(self):
        if self.wrist_qpos_adrs.size == 0:
            self.wrist_qpos_fixed = None
            return
        self.wrist_qpos_fixed = self.data.qpos[self.wrist_qpos_adrs].copy()

    def _enforce_wrist_fixed(self):
        if self.wrist_qpos_fixed is None:
            return
        self.data.qpos[self.wrist_qpos_adrs] = self.wrist_qpos_fixed
        self.data.qvel[self.wrist_dof_adrs] = 0.0

    def _touching_can(self) -> bool:
        if self.can_geom_id < 0:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if int(c.geom1) == self.can_geom_id or int(c.geom2) == self.can_geom_id:
                return True
        return False

    def _on_correct_side(self) -> bool:
        can_y = float(self.data.site_xpos[self.can_sid][1])
        palm_y = float(self.data.site_xpos[self.palm_sid][1])
        return (palm_y <= can_y - self.side_margin) if self.right_side else (palm_y >= can_y + self.side_margin)

    # ---------- obs, act ----------
    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        qpos = self.data.qpos[self.ctrl_jnt_qposadr].copy()
        qvel = self.data.qvel[self.ctrl_jnt_dofadr].copy()
        can_pos = self.data.site_xpos[self.can_sid].copy()
        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        rel_vec = can_pos - palm_pos
        return np.concatenate([qpos, qvel, can_pos, rel_vec]).astype(np.float32)

    def _apply_action(self, action):
        action = np.clip(action.astype(np.float64), -1.0, 1.0)

        # incremental target (VERY small steps)
        self.des_q = np.clip(
            self.des_q + (self.action_scale * action),
            self.ctrl_jnt_range[:, 0],
            self.ctrl_jnt_range[:, 1],
        )

        q  = self.data.qpos[self.ctrl_jnt_qposadr].astype(np.float64)
        qd = self.data.qvel[self.ctrl_jnt_dofadr].astype(np.float64)
        tau = self.kp_vec * (self.des_q - q) - self.kd_vec * qd
        tau = np.clip(tau, -self.torque_limit_vec, self.torque_limit_vec)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.ctrl_actuator_ids] = tau


    # ---------- reset/step ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # neutral arm poses (wrists will be locked)
        if self.right_side:
            _set_joint_if_exists(self.model, self.data, "right_shoulder_pitch_joint", -0.15)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_roll_joint",  -0.25)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_yaw_joint",    0.00)
            _set_joint_if_exists(self.model, self.data, "right_elbow_joint",           0.35)
        else:
            _set_joint_if_exists(self.model, self.data, "left_shoulder_pitch_joint",  -0.15)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_roll_joint",    0.25)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_yaw_joint",     0.00)
            _set_joint_if_exists(self.model, self.data, "left_elbow_joint",            0.35)

        # lock wrist to a known "thumb-up, palm-facing" pose
        for jn, val in zip(self.wrist_joint_names, self.wrist_lock_pose):
            _set_joint_if_exists(self.model, self.data, jn, val)

        mujoco.mj_forward(self.model, self.data)

        # freeze other arm and the wrist stack
        self._record_other_side_fixed_pose()
        self._enforce_other_side_fixed()
        self._record_wrist_fixed_pose()
        self._enforce_wrist_fixed()

        # set desired = current for controlled joints
        self.des_q = self.data.qpos[self.ctrl_jnt_qposadr].copy()

        # optional can randomization (off by default)
        if self.randomize_init:
            # small random within a front window
            can_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "can_body")
            if can_body >= 0:
                # If you later add a free joint, switch to joint-based randomization.
                self.model.body_pos[can_body, 0] = 0.42 + np.random.uniform(-0.02, 0.02)  # x
                self.model.body_pos[can_body, 1] = -0.03 + np.random.uniform(-0.03, 0.03) # y
                self.model.body_pos[can_body, 2] = 1.03 + np.random.uniform(-0.01, 0.01)  # z
            mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # keep other side and wrist pinned before/while stepping
        self._enforce_other_side_fixed()
        self._enforce_wrist_fixed()

        self._apply_action(action)

        # substeps for stability (slow)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            self._enforce_other_side_fixed()
            self._enforce_wrist_fixed()

        obs = self._get_obs()

        # geometry
        can_pos  = self.data.site_xpos[self.can_sid].copy()
        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        v = can_pos - palm_pos
        v_xy = v[:2]
        dist_xy = float(np.linalg.norm(v_xy))

        # orientation: use wrist body x-axis as the palm normal (because palm mesh offset +x)
        if self.wrist_bid >= 0:
            R = self.data.xmat[self.wrist_bid].reshape(3, 3)
            palm_normal = R[:, 0]  # body x-axis
        else:
            palm_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        v_hat = v / (np.linalg.norm(v) + 1e-8)
        face_dot = float(np.clip(np.dot(palm_normal, v_hat), -1.0, 1.0))
        face_r = 0.3 * max(0.0, face_dot)   # reward only when facing the can

        # side correctness (use palm y)
        can_y   = float(can_pos[1])
        palm_y  = float(palm_pos[1])
        desired_sign = -1.0 if self.right_side else +1.0
        side_align = desired_sign * (palm_y - can_y)
        side_r = 0.3 * max(0.0, side_align)

        # approach but don't touch: peak at standoff in XY
        sigma = 0.010
        approach_r = float(np.exp(-((dist_xy - self.standoff)**2) / (2*sigma*sigma)))

        # barrier to keep hand outside the can in XY
        inner_barrier = 0.0
        if dist_xy < self.min_xy_gap:
            inner_barrier = 8.0 * (self.min_xy_gap - dist_xy)

        # effort + contact
        ctrl_pen = 1e-3 * float(np.sum(self.data.ctrl[self.ctrl_actuator_ids] ** 2))
        touching = self._touching_can()
        touch_pen = self.touch_penalty if touching else 0.0

        reward = approach_r + side_r + face_r - ctrl_pen - inner_barrier - touch_pen

        # success: near standoff, correct side, facing, not touching, and outside barrier
        self.step_count += 1
        success = (
            (abs(dist_xy - self.standoff) < self.standoff_tol) and
            (side_align > 0.0) and
            (face_dot > 0.6) and    # reasonably facing the can
            (not touching) and
            (inner_barrier == 0.0)
        )
        terminated = bool(success)
        truncated  = bool(self.step_count >= self.max_steps)

        info = {
            "is_success": success,
            "dist_xy": dist_xy,
            "approach_r": approach_r,
            "side_r": side_r,
            "face_r": face_r,
            "touching": touching,
            "inner_barrier": inner_barrier,
        }
        return obs, float(reward), terminated, truncated, info

    # ---------- render/close (unchanged) ----------
    def render(self):
        if self.render_mode != "human":
            return
        if not HAVE_MJ_VIEWER:
            raise RuntimeError(
                "mujoco.viewer not available. Install mujoco>=2.3.5 and set MUJOCO_GL=glfw."
            )
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None