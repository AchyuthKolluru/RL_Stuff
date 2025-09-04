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
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        scene_xml_path: str,
        render_mode: str = "none",
        hand: str = "right",
        max_steps: int = 300,
        standoff: float | None = None,    # if None: can_radius + 0.012
        standoff_tol: float = 0.01,
        side_margin: float = 0.02,
        touch_penalty: float = 4.0,
        kp: float = 10.0,
        kd: float = 3.0,
        action_scale: float = 0.003,      # VERY slow steps
    ):
        super().__init__()
        if not os.path.isfile(scene_xml_path):
            raise FileNotFoundError(scene_xml_path)
        self.model = MjModel.from_xml_path(scene_xml_path)
        self.data  = MjData(self.model)
        self.render_mode   = render_mode
        self.max_steps     = int(max_steps)
        self.side_margin   = float(side_margin)
        self.standoff_tol  = float(standoff_tol)
        self.touch_penalty = float(touch_penalty)
        self.kp, self.kd   = float(kp), float(kd)
        self.action_scale  = float(action_scale)

        self.right_side = hand.lower().startswith("r")

        # -------- joints we control (4 DOF) --------
        if self.right_side:
            self.ctrl_joint_names = ["right_shoulder_pitch_joint",
                                     "right_shoulder_roll_joint",
                                     "right_shoulder_yaw_joint",
                                     "right_elbow_joint"]
            self.wrist_joint_names = ["right_wrist_roll_joint",
                                      "right_wrist_pitch_joint",
                                      "right_wrist_yaw_joint"]
            self.palm_site_name = "palm_site_right"
            self.wrist_body_name = "right_wrist_yaw_link"
            self.other_arm_joint_names = [
                "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
                "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
            ]
        else:
            self.ctrl_joint_names = ["left_shoulder_pitch_joint",
                                     "left_shoulder_roll_joint",
                                     "left_shoulder_yaw_joint",
                                     "left_elbow_joint"]
            self.wrist_joint_names = ["left_wrist_roll_joint",
                                      "left_wrist_pitch_joint",
                                      "left_wrist_yaw_joint"]
            self.palm_site_name = "palm_site_left"
            self.wrist_body_name = "left_wrist_yaw_link"
            self.other_arm_joint_names = [
                "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
                "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
            ]

        # Waist joints to pin
        self.waist_joint_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

        # Resolve IDs/addresses
        self.ctrl_joint_ids = _joint_ids(self.model, self.ctrl_joint_names)
        self.ctrl_actuator_ids = np.array(find_actuators_by_name(self.model, self.ctrl_joint_names), dtype=int)
        self.ctrl_jnt_qposadr = self.model.jnt_qposadr[self.ctrl_joint_ids]
        self.ctrl_jnt_dofadr  = self.model.jnt_dofadr[self.ctrl_joint_ids]
        self.ctrl_jnt_range   = self.model.jnt_range[self.ctrl_joint_ids].copy()
        self.wrist_joint_ids  = _joint_ids(self.model, self.wrist_joint_names)
        self.wrist_qpos_adrs  = np.array([self.model.jnt_qposadr[j] for j in self.wrist_joint_ids], dtype=int)
        self.wrist_dof_adrs   = np.array([self.model.jnt_dofadr[j]  for j in self.wrist_joint_ids], dtype=int)
        self.waist_joint_ids  = _joint_ids(self.model, self.waist_joint_names)
        self.waist_qpos_adrs  = np.array([self.model.jnt_qposadr[j] for j in self.waist_joint_ids], dtype=int)
        self.waist_dof_adrs   = np.array([self.model.jnt_dofadr[j]  for j in self.waist_joint_ids], dtype=int)
        self.other_joint_ids  = _joint_ids(self.model, self.other_arm_joint_names)
        self.other_qpos_adrs  = np.array([self.model.jnt_qposadr[j] for j in self.other_joint_ids], dtype=int)
        self.other_dof_adrs   = np.array([self.model.jnt_dofadr[j]  for j in self.other_joint_ids], dtype=int)

        self.palm_sid  = named_site_id(self.model, self.palm_site_name)
        self.can_sid   = named_site_id(self.model, "can_site")
        self.wrist_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.wrist_body_name)

        try:
            self.can_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom")
        except Exception:
            self.can_geom_id = -1

        if self.can_geom_id >= 0:
            sz = self.model.geom_size[self.can_geom_id]  # cylinder [radius, half_height]
            self.can_radius = float(sz[0])
            self.can_half_h = float(sz[1])
        else:
            self.can_radius, self.can_half_h = 0.03, 0.06

        self.standoff = float(self.can_radius + 0.012) if (standoff is None) else float(standoff)
        self.min_xy_gap = self.can_radius + 0.006

        # controllers
        self.kp_vec = np.full(4, self.kp, dtype=np.float64)
        self.kd_vec = np.full(4, self.kd, dtype=np.float64)
        self.torque_limit_vec = np.array([12, 10, 8, 8], dtype=np.float64)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        obs_dim = 4 + 4 + 3 + 3  # qpos,qvel,can_pos,palm->can
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.des_q = np.zeros(4, dtype=np.float64)
        self.step_count = 0
        self.viewer = None

        # storage for pinned joints
        self.wrist_qpos_fixed = None
        self.waist_qpos_fixed = None
        self.other_qpos_fixed = None

    # ----- simple pins -----
    def _record_fixed(self, adrs, attr_name):
        if adrs.size == 0:
            setattr(self, attr_name, None)
        else:
            setattr(self, attr_name, self.data.qpos[adrs].copy())

    def _enforce_fixed(self, qpos_adrs, dof_adrs, attr_name):
        fixed = getattr(self, attr_name)
        if fixed is None or qpos_adrs.size == 0:
            return
        self.data.qpos[qpos_adrs] = fixed
        self.data.qvel[dof_adrs]  = 0.0

    def _touching_can(self) -> bool:
        if self.can_geom_id < 0:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if int(c.geom1) == self.can_geom_id or int(c.geom2) == self.can_geom_id:
                return True
        return False

    # ----- obs/act -----
    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        qpos = self.data.qpos[self.ctrl_jnt_qposadr].copy()
        qvel = self.data.qvel[self.ctrl_jnt_dofadr].copy()
        can_pos  = self.data.site_xpos[self.can_sid].copy()
        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        rel_vec  = can_pos - palm_pos
        return np.concatenate([qpos, qvel, can_pos, rel_vec]).astype(np.float32)

    def _apply_action(self, action):
        action = np.clip(action.astype(np.float64), -1.0, 1.0)
        self.des_q = np.clip(
            self.des_q + (self.action_scale * action),
            self.ctrl_jnt_range[:, 0], self.ctrl_jnt_range[:, 1]
        )
        q  = self.data.qpos[self.ctrl_jnt_qposadr].astype(np.float64)
        qd = self.data.qvel[self.ctrl_jnt_dofadr].astype(np.float64)
        tau = self.kp_vec * (self.des_q - q) - self.kd_vec * qd
        tau = np.clip(tau, -self.torque_limit_vec, self.torque_limit_vec)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.ctrl_actuator_ids] = tau

    # ----- gym API -----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # gentle neutral arms
        if self.right_side:
            _set_joint_if_exists(self.model, self.data, "right_shoulder_pitch_joint", -0.15)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_roll_joint",  -0.25)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_yaw_joint",    0.00)
            _set_joint_if_exists(self.model, self.data, "right_elbow_joint",           0.35)
            # Lock wrist straight (vertical hand, thumb up, palm toward +X)
            _set_joint_if_exists(self.model, self.data, "right_wrist_roll_joint",  0.0)
            _set_joint_if_exists(self.model, self.data, "right_wrist_pitch_joint", 0.0)
            _set_joint_if_exists(self.model, self.data, "right_wrist_yaw_joint",   0.0)
        else:
            _set_joint_if_exists(self.model, self.data, "left_shoulder_pitch_joint",  -0.15)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_roll_joint",    0.25)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_yaw_joint",     0.00)
            _set_joint_if_exists(self.model, self.data, "left_elbow_joint",            0.35)
            _set_joint_if_exists(self.model, self.data, "left_wrist_roll_joint",  0.0)
            _set_joint_if_exists(self.model, self.data, "left_wrist_pitch_joint", 0.0)
            _set_joint_if_exists(self.model, self.data, "left_wrist_yaw_joint",   0.0)

        # waist pinned (even if XML ranges are 0)
        _set_joint_if_exists(self.model, self.data, "waist_yaw_joint",   0.0)
        _set_joint_if_exists(self.model, self.data, "waist_roll_joint",  0.0)
        _set_joint_if_exists(self.model, self.data, "waist_pitch_joint", 0.0)

        mujoco.mj_forward(self.model, self.data)

        # record pins
        self._record_fixed(self.wrist_qpos_adrs, "wrist_qpos_fixed")
        self._record_fixed(self.waist_qpos_adrs, "waist_qpos_fixed")
        self._record_fixed(self.other_qpos_adrs, "other_qpos_fixed")

        # desired target = current
        self.des_q = self.data.qpos[self.ctrl_jnt_qposadr].copy()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        # pin waist, wrist, other side
        self._enforce_fixed(self.waist_qpos_adrs, self.waist_dof_adrs, "waist_qpos_fixed")
        self._enforce_fixed(self.wrist_qpos_adrs, self.wrist_dof_adrs, "wrist_qpos_fixed")
        self._enforce_fixed(self.other_qpos_adrs, self.other_dof_adrs, "other_qpos_fixed")

        self._apply_action(action)

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            self._enforce_fixed(self.waist_qpos_adrs, self.waist_dof_adrs, "waist_qpos_fixed")
            self._enforce_fixed(self.wrist_qpos_adrs, self.wrist_dof_adrs, "wrist_qpos_fixed")
            self._enforce_fixed(self.other_qpos_adrs, self.other_dof_adrs, "other_qpos_fixed")

        obs = self._get_obs()

        # approach in XY to standoff
        can_pos  = self.data.site_xpos[self.can_sid].copy()
        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        v   = can_pos - palm_pos
        vxy = v[:2]
        dist_xy = float(np.linalg.norm(vxy))

        # palm facing can: use wrist body +X as palm normal; reward when aligned
        if self.wrist_bid >= 0:
            R = self.data.xmat[self.wrist_bid].reshape(3, 3)
            palm_normal = R[:, 0]
        else:
            palm_normal = np.array([1.0, 0.0, 0.0])
        v_hat = v / (np.linalg.norm(v) + 1e-8)
        face_dot = float(np.clip(np.dot(palm_normal, v_hat), -1.0, 1.0))
        face_r = 0.3 * max(0.0, face_dot)

        # correct side (palm y w.r.t. can y)
        desired_sign = -1.0 if self.right_side else +1.0
        side_align = desired_sign * (palm_pos[1] - can_pos[1])
        side_r = 0.3 * max(0.0, side_align)

        # soft peak at standoff (Gaussian) + inner barrier to avoid touching
        sigma = 0.010
        approach_r = float(np.exp(-((dist_xy - self.standoff)**2) / (2*sigma*sigma)))
        inner_barrier = 8.0 * (self.min_xy_gap - dist_xy) if dist_xy < self.min_xy_gap else 0.0

        ctrl_pen = 1e-3 * float(np.sum(self.data.ctrl[self.ctrl_actuator_ids] ** 2))
        touching = self._touching_can()
        touch_pen = self.touch_penalty if touching else 0.0

        reward = approach_r + side_r + face_r - ctrl_pen - inner_barrier - touch_pen

        self.step_count += 1
        success = (
            (abs(dist_xy - self.standoff) < self.standoff_tol) and
            (side_align > 0.0) and
            (face_dot > 0.6) and
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

    def render(self):
        if self.render_mode != "human": return
        if not HAVE_MJ_VIEWER:
            raise RuntimeError("mujoco.viewer not available.")
        if not hasattr(self, "viewer") or self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None