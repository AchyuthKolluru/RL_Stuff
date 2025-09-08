# env_g1_inspire_can.py

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

def _geom_quat(data, gid):
    R = data.geom_xmat[gid].reshape(3, 3)
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
    Make the palm move to the chosen lateral side of the can and stop outside it.
    Key ideas:
      - Target is a standoff point on ±Y of the can (not its center).
      - Strong barrier if the palm is inside the can radius + margin.
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        scene_xml_path: str,
        render_mode: str = "none",
        hand: str = "right",
        max_steps: int = 300,
        randomize_init: bool = False,
        # distances (meters)
        standoff: float = 0.015,
        standoff_tol: float = 0.008,
        side_margin: float = 0.01,
        # rewards/penalties
        side_weight: float = 2.0,
        touch_penalty: float = 6.0,
        ctrl_cost_scale: float = 1e-3,
        # control
        action_scale: float = 0.003,
        kp: float = 10.0,  # softer PD → less overshoot
        kd: float = 1.2,
        torque_limits=(12, 10, 10, 8),
        # misc
        freeze_other: bool = True,
        ik_warm_start: bool = False,

        # === NEW: generalization + curriculum knobs ===
        auto_choose_nearer_side: bool = True,     # choose the laterally nearer side automatically
        perception_noise_std: float = 0.002,         # meters of Gaussian noise on can center used for target
        domain_randomize: bool = True,            # jitter can size/friction a bit at reset
        size_jitter_frac: float = 0.1,             # ±10% size jitter for can radius/height
        friction_jitter_frac: float = 0.25,        # ±25% friction jitter
        # workspace center & half-ranges for randomization (m)
        workspace_center=(0.45, 0.0, 1.02),        # x,y,z nominal can pose
        workspace_half_range=(0.12, 0.18, 0.04),   # Δx, Δy, Δz ranges (half extents)
        yaw_range: float = 0.6,                    # yaw range (± radians)
        pitch_range: float = 0.05,                 # small tilt
        roll_range: float = 0.05,                  # small tilt
        init_rand_scale: float = 1.0,              # curriculum scale (0.2..1.0 typical)

        # --- orientation shaping (add these to __init__ args with sane defaults) ---
        upright_coef: float = 2.0,       # penalize palm tilting down
        lookat_coef: float = 3.0,        # encourage palm to face can
        elbow_pref: float = 0.70,        # preferred elbow angle (rad) ~40°
        elbow_coef: float = 0.3,         # how strong to keep elbow away from tuck
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

        self.upright_coef = float(upright_coef)
        self.lookat_coef  = float(lookat_coef)
        self.elbow_pref   = float(elbow_pref)
        self.elbow_coef   = float(elbow_coef)
        self._last_marker_t = -1


        # === NEW: store generalization knobs
        self.auto_choose_nearer_side = bool(auto_choose_nearer_side)
        self.perception_noise_std = float(perception_noise_std)
        self.domain_randomize = bool(domain_randomize)
        self.size_jitter_frac = float(size_jitter_frac)
        self.friction_jitter_frac = float(friction_jitter_frac)
        self.ws_center = np.array(workspace_center, dtype=np.float64)
        self.ws_half = np.array(workspace_half_range, dtype=np.float64)
        self.yaw_range = float(yaw_range)
        self.pitch_range = float(pitch_range)
        self.roll_range = float(roll_range)
        self.rand_scale = float(init_rand_scale)

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

        # elbow joint id (+ qpos adr) for whichever side we control
        elbow_name = "right_elbow_joint" if self.right_side else "left_elbow_joint"
        self.elbow_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, elbow_name)
        self.elbow_qadr = self.model.jnt_qposadr[self.elbow_jid] if self.elbow_jid >= 0 else None

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

        # waist hard-lock
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

        self.max_joint_step = 0.05  # radians per env step; slow & safe

        self.progress_coef = 200.0    # how much to pay for getting closer
        self.time_penalty  = 0.002    # per-step tax to discourage loafing

        # desired q for controlled joints
        self.des_q = self.data.qpos[self.ctrl_jnt_qposadr].copy()

        # ---- observation / targets ----
        # palm site
        self.palm_sid = named_site_id(self.model, self.palm_site_name)

        # can geometry (for size + pose)
        self.can_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom")
        if self.can_geom_id < 0:
            raise RuntimeError("Geom 'can_geom' not found in XML.")

        sz = self.model.geom_size[self.can_geom_id]
        self.can_radius = float(sz[0])
        self.can_half_h = float(sz[1])

        # keep original (for domain randomization reset)
        # === NEW:
        self._base_can_size = sz.copy()
        self._base_can_friction = self.model.geom_friction[self.can_geom_id].copy()

        # inside-approach barrier threshold: radius + small clearance
        self.min_radial_gap = self.can_radius + 0.006

        # optional free joint for can
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
        self.can_free_joint = j if j >= 0 else None

        # observation: qpos,qvel + can_pos(3) + can_quat(4) + rel_vec(3)
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

    # ---------- target helpers ----------
    def _can_frame(self):
        """
        Returns can center, can local +Y axis, can local +Z axis, and its rotation matrix.
        Uses can_geom (works whether the can is static or has a free joint with yaw).
        """
        pos = self.data.geom_xpos[self.can_geom_id].copy()
        R = self.data.geom_xmat[self.can_geom_id].reshape(3, 3).copy()
        y_axis = R[:, 1]
        z_axis = R[:, 2]
        return pos, y_axis, z_axis, R
    
    def _draw_target_marker(self, radius_m=0.05):
        if self.viewer is None:
            return
        try:
            pos = self._target_pos()
            # MASSIVE, bright magenta, semi-transparent
            self.viewer.add_marker(
                pos=pos,
                size=(radius_m, radius_m, radius_m),         # radius in meters (sphere)
                rgba=(1.0, 0.0, 1.0, 0.9),                    # magenta, nearly opaque
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label="TARGET"
            )
        except Exception as _:
            pass


    # === NEW:
    def _can_center_noisy(self, center: np.ndarray):
        if self.perception_noise_std <= 0.0:
            return center
        return center + np.random.normal(scale=self.perception_noise_std, size=3)

    def _target_pos(self):
        """
        Lateral standoff target outside the can.
        """
        can_center, y_axis, _, _ = self._can_frame()
        # === NEW: optional auto-choose nearer lateral side
        if self.auto_choose_nearer_side:
            palm_pos = self.data.site_xpos[self.palm_sid]
            sgn = -1.0 if np.dot(palm_pos - can_center, y_axis) >= 0 else +1.0
        else:
            sgn = -1.0 if self.right_side else +1.0  # right = negative Y side

        can_center = self._can_center_noisy(can_center)
        return can_center + sgn * (self.can_radius + self.standoff) * y_axis

    # ---------- helpers ----------
    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        qpos = self.data.qpos[self.ctrl_jnt_qposadr].copy()
        qvel = self.data.qvel[self.ctrl_jnt_dofadr].copy()

        can_center, _, _, R = self._can_frame()
        can_center = self._can_center_noisy(can_center)   # === NEW: noisy obs if enabled
        can_pos = can_center
        can_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(can_quat, R.ravel())

        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        rel_vec = self._target_pos() - palm_pos

        return np.concatenate([qpos, qvel, can_pos, can_quat, rel_vec]).astype(np.float32)

    # replace _apply_action with this version:
    def _apply_action(self, action):
        action = np.clip(action.astype(np.float64), -1.0, 1.0)
        # proposed new desired q
        proposed = self.des_q + (self.action_scale_vec * action)
        proposed = np.clip(proposed, self.ctrl_jnt_range[:, 0], self.ctrl_jnt_range[:, 1])

        # rate-limit the change in desired q
        delta = proposed - self.des_q
        delta = np.clip(delta, -self.max_joint_step, self.max_joint_step)
        self.des_q = self.des_q + delta

        # PD to current q
        q  = self.data.qpos[self.ctrl_jnt_qposadr].astype(np.float64)
        qd = self.data.qvel[self.ctrl_jnt_dofadr].astype(np.float64)
        tau = self.kp_vec * (self.des_q - q) - self.kd_vec * qd
        tau = np.clip(tau, -self.torque_limit_vec[:self.n_total], self.torque_limit_vec[:self.n_total])
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.ctrl_actuator_ids] = tau

    # === NEW: curriculum control from training script
    def set_randomization_scale(self, scale: float):
        """Set [0.2, 1.0] scaling for randomization ranges (for curriculum)."""
        self.rand_scale = float(np.clip(scale, 0.2, 1.5))

    # === NEW: domain randomization (can size/friction)
    def _domain_randomize_can(self):
        # reset to base
        self.model.geom_size[self.can_geom_id] = self._base_can_size
        self.model.geom_friction[self.can_geom_id] = self._base_can_friction

        if not self.domain_randomize:
            return

        # jitter size (radius & half-height)
        sz = self._base_can_size.copy()
        frac = self.size_jitter_frac
        sz[0] *= (1.0 + np.random.uniform(-frac, +frac))   # radius
        sz[1] *= (1.0 + np.random.uniform(-frac, +frac))   # half-height
        self.model.geom_size[self.can_geom_id] = sz

        # update cached radius/height & min gap
        self.can_radius = float(sz[0])
        self.can_half_h = float(sz[1])
        self.min_radial_gap = self.can_radius + 0.006

        # jitter friction (slide/roll/spin)
        base_fric = self._base_can_friction.copy()
        fj = self.friction_jitter_frac
        fric = base_fric * (1.0 + np.random.uniform(-fj, +fj, size=3))
        self.model.geom_friction[self.can_geom_id] = fric

    def _randomize(self):
        if self.can_free_joint is None:
            return

        # === NEW: domain randomization first (affects radius in reward/barrier)
        self._domain_randomize_can()

        adr = self.model.jnt_qposadr[self.can_free_joint]

        # === NEW: workspace randomization with curriculum scaling
        s = self.rand_scale
        x0, y0, z0 = self.ws_center
        hx, hy, hz = self.ws_half
        x = x0 + np.random.uniform(-hx*s, +hx*s)
        y = y0 + np.random.uniform(-hy*s, +hy*s)
        z = z0 + np.random.uniform(-hz*s, +hz*s)
        self.data.qpos[adr:adr + 3] = np.array([x, y, z], dtype=np.float64)

        # yaw/pitch/roll ranges (scaled)
        yaw   = np.random.uniform(-self.yaw_range*s,   +self.yaw_range*s)
        pitch = np.random.uniform(-self.pitch_range*s, +self.pitch_range*s)
        roll  = np.random.uniform(-self.roll_range*s,  +self.roll_range*s)

        cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cr, sr = math.cos(roll/2),  math.sin(roll/2)
        # Z(yaw) * Y(pitch) * X(roll)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        self.data.qpos[adr + 3:adr + 7] = np.array([qw, qx, qy, qz], dtype=np.float64)

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

    # ---------------- gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._prev_d = None

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

        mujoco.mj_forward(self.model, self.data)

        # --------- compute geometry-aware terms ---------
        can_center, y_axis, z_axis, _ = self._can_frame()
        can_center_nz = self._can_center_noisy(can_center)  # === NEW: same noise used for reward if enabled
        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        target_pos = self._target_pos()

        palm_R = self.data.site_xmat[self.palm_sid].reshape(3, 3).copy()

        # Try +X as the "out-of-palm" (approach) axis. If it seems wrong in viz,
        # switch to [:, 1] or [:, 2].
        palm_forward = palm_R[:, 0]      # direction palm "points" to
        palm_up      = palm_R[:, 2]      # palm normal / "up" axis (swap if your CAD differs)

        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # 2a) Upright penalty: discourage palm facing down or steep tilt
        # dot = 1 means perfectly upright; dot = 0 tilted 90°; dot < 0 upside down.
        upright_dot = float(np.clip(np.dot(palm_up, world_z), -1.0, 1.0))
        upright_pen = self.upright_coef * (1.0 - max(0.0, upright_dot))  # only penalize loss of uprightness


        # 2b) Look-at-can penalty: encourage the palm to face the can center
        to_can = can_center - palm_pos
        nc = np.linalg.norm(to_can) + 1e-9
        to_can /= nc
        lookat_pen = self.lookat_coef * (1.0 - max(0.0, float(np.dot(palm_forward, to_can))))

        # 2c) Elbow posture penalty: softly prefer a non-tucked angle
        elbow_pen = 0.0
        if self.elbow_qadr is not None:
            q_elbow = float(self.data.qpos[self.elbow_qadr])
            elbow_pen = self.elbow_coef * (q_elbow - self.elbow_pref) ** 2


        # distance to standoff target (this is what we want to minimize)
        d_target = float(np.linalg.norm(target_pos - palm_pos))

        if self._prev_d is None:
            self._prev_d = d_target
        progress = self._prev_d - d_target
        self._prev_d = d_target


        # radial distance from cylinder axis (penalize being inside)
        vec_cp_true = palm_pos - can_center  # use true center for radial/contacts
        vec_perp = vec_cp_true - np.dot(vec_cp_true, z_axis) * z_axis  # remove z-axis component
        radial = float(np.linalg.norm(vec_perp))
        inside_gap = (self.min_radial_gap - radial)
        inner_barrier = 120.0 * (inside_gap * inside_gap) if inside_gap > 0.0 else 0.0

        # Horizontal/planar distance to cylinder axis (ignore height)
        vec_planar = vec_cp_true - np.dot(vec_cp_true, z_axis) * z_axis
        planar_radial = float(np.linalg.norm(vec_planar))
        # positive if outside the desired surface (radius + standoff), negative if inside
        planar_gap = planar_radial - (self.can_radius + self.standoff)
        planar_gap_outside = max(0.0, planar_gap)

        # Strong planar shaping: higher when closer to the standoff circle in XY
        planar_shaping = 6.0 * (1.0 / (0.01 + planar_gap_outside))


        # correct side: projection onto ±Y axis should exceed a margin
        side_progress = (-1.0 if (self.right_side and not self.auto_choose_nearer_side) else +1.0)
        # If auto side, compute violation against chosen side:
        if self.auto_choose_nearer_side:
            # decide sign by comparing palm vs can along y-axis
            sgn = -1.0 if np.dot(palm_pos - can_center_nz, y_axis) >= 0 else +1.0
            side_progress = sgn * float(np.dot(palm_pos - can_center_nz, y_axis))
        else:
            side_progress = (-1.0 if self.right_side else +1.0) * float(np.dot(palm_pos - can_center_nz, y_axis))

        side_violation = max(0.0, self.side_margin - side_progress)
        side_pen = side_violation * side_violation

        # vertical alignment: stay near can mid-height (tolerant)
        vertical_dev = abs(float(np.dot(vec_cp_true, z_axis)))
        # ---- REPLACE your topdown pen with this gated version ----
        near_lateral = radial < (self.can_radius + self.standoff + 0.015)  # ~1.5 cm beyond standoff
        topdown_pen = 0.0
        if near_lateral:
            vertical_dev = abs(float(np.dot(vec_cp_true, z_axis)))
            topdown_pen = 0.2 * max(0.0, vertical_dev - self.can_half_h * 0.4)
# ----------------------------------------------------------


        # contacts + control cost
        touching = self._touching_can()
        touch_pen = self.touch_penalty if touching else 0.0
        ctrl_penalty = self.ctrl_cost_scale * float(np.sum(self.data.ctrl[self.ctrl_actuator_ids] ** 2))

        # approach reward: now just closeness to the **outside** standoff target
        
        approach_r = self.progress_coef * progress

        qd = self.data.qvel[self.ctrl_jnt_dofadr]
        posture_smooth_pen = 1e-4 * float(np.sum(qd**2))

        reward = (
            planar_shaping + approach_r
            - self.side_weight * side_pen
            - inner_barrier
            - topdown_pen
            - touch_pen
            - ctrl_penalty
            - posture_smooth_pen
            - self.time_penalty
            - upright_pen
            - lookat_pen
            - elbow_pen 
        )

        self.step_count += 1

        success = (d_target < self.standoff_tol) and (side_violation == 0.0) and (not touching) and (inner_barrier == 0.0)
        terminated = bool(success)
        truncated = bool(self.step_count >= self.max_steps)


        # build obs at the end (includes target−palm vector)
        obs = self._get_obs()

        info = {
            "is_success": success,
            "d_target": d_target,
            "radial": radial,
            "side_progress": side_progress,
            "touching": touching,
            "approach_r": approach_r,
            "side_pen": side_pen,
            "inner_barrier": inner_barrier,
            "topdown_pen": topdown_pen,
            # === NEW (nice to log while debugging/randomizing):
            "rand_scale": self.rand_scale,
            "can_radius": self.can_radius,
            "upright_dot": upright_dot,
            "upright_pen": upright_pen,
            "lookat_pen": lookat_pen,
            "elbow_pen": elbow_pen,
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
             # Ensure forward kinematics are up to date
            mujoco.mj_forward(self.model, self.data)

            # 🔴 draw the giant target marker *every frame*
            self._draw_target_marker(radius_m=0.07)

            # Flush to screen
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
