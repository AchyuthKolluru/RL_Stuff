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


def _site_quat(data, sid):
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
    Simple arm-only approach env.
    - Control: shoulder pitch/roll/yaw + elbow (4 DoF) on one arm.
    - Wrist hard-locked level (palm parallel to ground, facing can).
    - Unused arm frozen. Waist joints hard-locked to stop any lean.
    - Reward: very close to can at a desired standoff, correct lateral side, no touch.
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        scene_xml_path: str,
        render_mode: str = "none",
        hand: str = "right",
        max_steps: int = 300,
        randomize_init: bool = True,
        standoff: float = 0.025,
        standoff_tol: float = 0.01,
        side_margin: float = 0.02,
        side_weight: float = 2.0,
        action_scale: float = 0.01,
        kp: float = 12.0,
        kd: float = 1.0,
        torque_limits=(12, 10, 10, 8),
        touch_penalty: float = 6.0,
        ctrl_cost_scale: float = 1e-3,
        freeze_other: bool = True,
        ik_warm_start: bool = False,
        **kwargs,
    ):
        if kwargs:
            import warnings
            warnings.warn(f"G1InspireCanGrasp: ignoring unexpected kwargs: {list(kwargs.keys())}")

        if not os.path.isfile(scene_xml_path):
            raise FileNotFoundError(scene_xml_path)

        self.model = MjModel.from_xml_path(scene_xml_path)
        self.data = MjData(self.model)

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.randomize_init = bool(randomize_init)
        self.freeze_other = bool(freeze_other)
        self.ik_warm_start = bool(ik_warm_start)

        self.standoff = float(standoff)
        self.standoff_tol = float(standoff_tol)
        self.side_margin = float(side_margin)
        self.side_weight = float(side_weight)
        self.touch_penalty = float(touch_penalty)
        self.ctrl_cost_scale = float(ctrl_cost_scale)

        self.kp = float(kp)
        self.kd = float(kd)
        self.action_scale = float(action_scale)
        self.torque_limit_vec = np.array(torque_limits, dtype=np.float64)

        self.wrist_forward_pose = np.array((0.0, 0.0, 0.0), dtype=np.float64)

        # side selection
        self.right_side = hand.lower().startswith("r")
        if self.right_side:
            self.arm_joint_names_full = [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
            self.palm_site_name = "palm_site_right"
            self.wrist_joint_names = [
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
            self.other_arm_joint_names = [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]
        else:
            self.arm_joint_names_full = [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]
            self.palm_site_name = "palm_site_left"
            self.wrist_joint_names = [
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]
            self.other_arm_joint_names = [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]

        # only control upper arm + elbow (4 DoF)
        self.ctrl_joint_names = self.arm_joint_names_full[:4]

        # actuators we control
        self.ctrl_actuator_ids = np.array(
            find_actuators_by_name(self.model, self.ctrl_joint_names), dtype=int
        )
        if len(self.ctrl_actuator_ids) != len(self.ctrl_joint_names):
            raise RuntimeError(f"Actuators not found for {self.ctrl_joint_names}. Check XML.")
        self.n_total = len(self.ctrl_actuator_ids)

        # map actuators → joints
        self.ctrl_to_joint = np.array(
            [int(self.model.actuator_trnid[a, 0]) for a in self.ctrl_actuator_ids],
            dtype=int
        )
        self.ctrl_jnt_qposadr = self.model.jnt_qposadr[self.ctrl_to_joint]
        self.ctrl_jnt_dofadr = self.model.jnt_dofadr[self.ctrl_to_joint]
        self.ctrl_jnt_range = self.model.jnt_range[self.ctrl_to_joint].copy()

        # wrist lock bookkeeping
        self.wrist_joint_ids = _joint_ids(self.model, self.wrist_joint_names)
        self.wrist_qpos_adrs = np.array(
            [self.model.jnt_qposadr[j] for j in self.wrist_joint_ids], dtype=int
        ) if self.wrist_joint_ids else np.array([], dtype=int)
        self.wrist_dof_adrs = np.array(
            [self.model.jnt_dofadr[j] for j in self.wrist_joint_ids], dtype=int
        ) if self.wrist_joint_ids else np.array([], dtype=int)
        self.wrist_qpos_fixed = None

        # === NEW: hard-lock waist joints to stop any lean ===
        self.waist_joint_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        self.waist_joint_ids = _joint_ids(self.model, self.waist_joint_names)
        self.waist_qpos_adrs = np.array(
            [self.model.jnt_qposadr[j] for j in self.waist_joint_ids], dtype=int
        ) if self.waist_joint_ids else np.array([], dtype=int)
        self.waist_dof_adrs = np.array(
            [self.model.jnt_dofadr[j] for j in self.waist_joint_ids], dtype=int
        ) if self.waist_joint_ids else np.array([], dtype=int)
        self.waist_qpos_fixed = None

        # other side frozen
        self.other_joint_ids = _joint_ids(self.model, self.other_arm_joint_names)
        self.other_qpos_adrs = np.array(
            [self.model.jnt_qposadr[j] for j in self.other_joint_ids], dtype=int
        )
        self.other_dof_adrs = np.array(
            [self.model.jnt_dofadr[j] for j in self.other_joint_ids], dtype=int
        )
        self.other_qpos_fixed = None

        # PD gains & scaling
        self.kp_vec = np.full(self.n_total, float(kp), dtype=np.float64)
        self.kd_vec = np.full(self.n_total, float(kd), dtype=np.float64)
        self.action_scale_vec = np.full(self.n_total, float(action_scale), dtype=np.float64)

        # desired q for controlled joints
        self.des_q = self.data.qpos[self.ctrl_jnt_qposadr].copy()

        # observation setup
        self.can_sid = named_site_id(self.model, "can_site")
        self.palm_sid = named_site_id(self.model, self.palm_site_name)

        # can geom
        try:
            self.can_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom")
        except Exception:
            self.can_geom_id = -1
        if self.can_geom_id >= 0:
            sz = self.model.geom_size[self.can_geom_id]
            self.can_radius = float(sz[0])
            self.can_half_h = float(sz[1])
        else:
            self.can_radius = 0.03
            self.can_half_h = 0.06
        self.min_xy_gap = self.can_radius + 0.006

        # optional free joint for can
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
        self.can_free_joint = j if j >= 0 else None

        obs_dim = (len(self.ctrl_jnt_qposadr) * 2) + 3 + 4 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_total,), dtype=np.float32)

        self.step_count = 0
        self.viewer = None

    # ---------- locks ----------
    def _record_other_side_fixed_pose(self):
        if not self.freeze_other or self.other_qpos_adrs.size == 0:
            self.other_qpos_fixed = None
            return
        self.other_qpos_fixed = self.data.qpos[self.other_qpos_adrs].copy()

    def _enforce_other_side_fixed(self):
        if not self.freeze_other or self.other_qpos_fixed is None:
            return
        self.data.qpos[self.other_qpos_adrs] = self.other_qpos_fixed
        self.data.qvel[self.other_dof_adrs] = 0.0

    def _record_wrist_fixed_pose(self):
        if self.wrist_qpos_adrs.size == 0:
            self.wrist_qpos_fixed = None
            return
        self.wrist_qpos_fixed = self.data.qpos[self.wrist_qpos_adrs].copy()

    def _enforce_wrist_fixed(self):
        if self.wrist_qpos_fixed is None or self.wrist_qpos_adrs.size == 0:
            return
        self.data.qpos[self.wrist_qpos_adrs] = self.wrist_qpos_fixed
        self.data.qvel[self.wrist_dof_adrs] = 0.0

    # NEW: waist hard-lock
    def _record_waist_fixed_pose(self):
        if self.waist_qpos_adrs.size == 0:
            self.waist_qpos_fixed = None
            return
        self.waist_qpos_fixed = self.data.qpos[self.waist_qpos_adrs].copy()

    def _enforce_waist_fixed(self):
        if self.waist_qpos_fixed is None or self.waist_qpos_adrs.size == 0:
            return
        self.data.qpos[self.waist_qpos_adrs] = self.waist_qpos_fixed
        self.data.qvel[self.waist_dof_adrs] = 0.0

    # ---------- helpers ----------
    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        qpos = self.data.qpos[self.ctrl_jnt_qposadr].copy()
        qvel = self.data.qvel[self.ctrl_jnt_dofadr].copy()
        can_pos = self.data.site_xpos[self.can_sid].copy()
        can_quat = _site_quat(self.data, self.can_sid)
        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        rel_vec = can_pos - palm_pos
        return np.concatenate([qpos, qvel, can_pos, can_quat, rel_vec]).astype(np.float32)

    def _apply_action(self, action):
        action = np.clip(action.astype(np.float64), -1.0, 1.0)
        self.des_q = np.clip(
            self.des_q + (self.action_scale_vec * action),
            self.ctrl_jnt_range[:, 0],
            self.ctrl_jnt_range[:, 1],
        )
        q = self.data.qpos[self.ctrl_jnt_qposadr].astype(np.float64)
        qd = self.data.qvel[self.ctrl_jnt_dofadr].astype(np.float64)
        tau = self.kp_vec * (self.des_q - q) - self.kd_vec * qd
        tau = np.clip(tau, -self.torque_limit_vec[:self.n_total], self.torque_limit_vec[:self.n_total])
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.ctrl_actuator_ids] = tau

    def _randomize(self):
        if self.can_free_joint is None:
            return
        adr = self.model.jnt_qposadr[self.can_free_joint]
        x = 0.42 + np.random.uniform(-0.03, 0.03)
        y = 0.00 + np.random.uniform(-0.05, 0.05)
        z = 1.03 + np.random.uniform(-0.02, 0.02)
        self.data.qpos[adr:adr + 3] = np.array([x, y, z], dtype=np.float64)
        yaw = np.random.uniform(-0.2, 0.2)
        self.data.qpos[adr + 3:adr + 7] = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], dtype=np.float64)
        dof = self.model.jnt_dofadr[self.can_free_joint]
        self.data.qvel[dof:dof + 6] = 0.0

    def _set_safe_arm_pose(self, side: str):
        if side == "right":
            _set_joint_if_exists(self.model, self.data, "right_shoulder_pitch_joint", -0.15)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_roll_joint", -0.25)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_yaw_joint",  0.00)
            _set_joint_if_exists(self.model, self.data, "right_elbow_joint",         0.35)
            _set_joint_if_exists(self.model, self.data, "right_wrist_roll_joint",    0.00)
            _set_joint_if_exists(self.model, self.data, "right_wrist_pitch_joint",   0.00)
            _set_joint_if_exists(self.model, self.data, "right_wrist_yaw_joint",     0.00)
        else:
            _set_joint_if_exists(self.model, self.data, "left_shoulder_pitch_joint", -0.15)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_roll_joint",   0.25)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_yaw_joint",    0.00)
            _set_joint_if_exists(self.model, self.data, "left_elbow_joint",           0.35)
            _set_joint_if_exists(self.model, self.data, "left_wrist_roll_joint",      0.00)
            _set_joint_if_exists(self.model, self.data, "left_wrist_pitch_joint",     0.00)
            _set_joint_if_exists(self.model, self.data, "left_wrist_yaw_joint",       0.00)

    def _touching_can(self) -> bool:
        if getattr(self, "can_geom_id", -1) < 0:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if int(c.geom1) == self.can_geom_id or int(c.geom2) == self.can_geom_id:
                return True
        return False

    def _side_penalty(self):
        mujoco.mj_forward(self.model, self.data)
        can_y = float(self.data.site_xpos[self.can_sid][1])
        palm_y = float(self.data.site_xpos[self.palm_sid][1])
        if self.right_side:
            violation = max(0.0, (palm_y - can_y) + self.side_margin)
        else:
            violation = max(0.0, -(palm_y - can_y) + self.side_margin)
        return float(violation * violation)

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._set_safe_arm_pose("right")
        self._set_safe_arm_pose("left")

        # level/forward wrist and lock it
        for jn, val in zip(self.wrist_joint_names, self.wrist_forward_pose):
            _set_joint_if_exists(self.model, self.data, jn, val)

        if self.randomize_init:
            self._randomize()

        mujoco.mj_forward(self.model, self.data)

        # record and enforce all locks
        self._record_other_side_fixed_pose()
        self._enforce_other_side_fixed()
        self._record_wrist_fixed_pose()
        self._enforce_wrist_fixed()
        self._record_waist_fixed_pose()
        self._enforce_waist_fixed()

        # desired = current
        self.des_q = self.data.qpos[self.ctrl_jnt_qposadr].copy()

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # keep everything pinned
        self._enforce_other_side_fixed()
        self._enforce_wrist_fixed()
        self._enforce_waist_fixed()

        self._apply_action(action)

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            self._enforce_other_side_fixed()
            self._enforce_wrist_fixed()
            self._enforce_waist_fixed()

        obs = self._get_obs()

        can_pos = self.data.site_xpos[self.can_sid].copy()
        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        diff = can_pos - palm_pos
        dist = float(np.linalg.norm(diff))

        approach_r = 2.0 * (1.0 / (0.02 + abs(dist - self.standoff)))

        side_pen = self._side_penalty()

        can_y = float(can_pos[1])
        palm_y = float(palm_pos[1])
        desired_sign = -1.0 if self.right_side else +1.0
        side_align = desired_sign * (palm_y - can_y)
        side_align_r = 0.5 * max(0.0, side_align)

        dist_xy = float(np.linalg.norm((can_pos - palm_pos)[:2]))
        inner_barrier = 0.0
        if dist_xy < self.min_xy_gap:
            inner_barrier = 8.0 * (self.min_xy_gap - dist_xy)

        can_mid_z = float(can_pos[2])
        palm_z = float(palm_pos[2])
        vertical_dev = abs(palm_z - can_mid_z)
        topdown_pen = 0.5 * max(0.0, vertical_dev - self.can_half_h * 0.4)

        touching = self._touching_can()
        touch_pen = self.touch_penalty if touching else 0.0

        ctrl_penalty = self.ctrl_cost_scale * float(np.sum(self.data.ctrl[self.ctrl_actuator_ids] ** 2))

        reward = (
            approach_r + side_align_r
            - self.side_weight * side_pen
            - inner_barrier
            - topdown_pen
            - touch_pen
            - ctrl_penalty
        )

        self.step_count += 1
        close_enough = (abs(dist - self.standoff) < self.standoff_tol)
        success = close_enough and (side_align > 0.0) and (not touching) and (inner_barrier == 0.0)

        terminated = bool(success)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "is_success": success,
            "dist": dist,
            "dist_xy": dist_xy,
            "touching": touching,
            "side_pen": side_pen,
            "approach_r": approach_r,
            "side_align_r": side_align_r,
            "inner_barrier": inner_barrier,
            "topdown_pen": topdown_pen,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if not HAVE_MJ_VIEWER:
            raise RuntimeError("mujoco.viewer not available. Install mujoco>=2.3.5 and set MUJOCO_GL=glfw.")
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None