# env_g1_inspire_can.py
import os, math, numpy as np, gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import MjModel, MjData

try:
    import mujoco.viewer  # noqa: F401
    HAVE_MJ_VIEWER = True
except Exception:
    HAVE_MJ_VIEWER = False

# Optional off-screen renderer for head-cam RGB (eval-time YOLO)
try:
    from mujoco import Renderer as MJRenderer
    HAVE_MJ_RENDERER = True
except Exception:
    HAVE_MJ_RENDERER = False


def named_site_id(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Site '{name}' not found in XML.")
    return sid

def _joint_ids(model, names):
    out = []
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            out.append(jid)
    return out

def _set_qpos_if(model, data, joint, val):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid >= 0:
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = float(val)


class G1InspireCanGrasp(gym.Env):
    """Reach a lateral standoff target near a can without touching it."""
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self,
                 scene_xml_path: str,
                 render_mode: str = "none",
                 hand: str = "right",
                 max_steps: int = 300,
                 randomize_init: bool = True,
                 # --- target geometry ---
                 standoff: float = 0.015,           # desired gap beyond can radius
                 standoff_tol: float = 0.008,       # success threshold
                 side_margin: float = 0.010,        # margin along Y-axis side test
                 # --- rewards/penalties ---
                 progress_coef: float = 200.0,
                 side_weight: float   = 2.0,
                 touch_penalty: float = 6.0,
                 time_penalty: float  = 0.002,
                 ctrl_cost_scale: float = 1e-3,
                 upright_coef: float = 4.0,         # stronger: palm parallel to ground
                 lookat_coef: float  = 5.0,         # stronger: palm faces can
                 elbow_pref: float   = 0.70,
                 elbow_coef: float   = 0.30,
                 # --- control ---
                 action_scale: float = 0.003,
                 kp: float = 10.0,
                 kd: float = 1.2,
                 torque_limits=(12, 10, 10, 8),
                 freeze_other: bool = True,
                 # --- generalization/randomization ---
                 domain_randomize: bool = True,
                 size_jitter_frac: float = 0.10,
                 friction_jitter_frac: float = 0.25,
                 # workspace center & half ranges (m)
                 workspace_center=(0.45, 0.0, 1.02),
                 workspace_half_range=(0.10, 0.15, 0.04),
                 yaw_range: float = 0.6,
                 pitch_range: float = 0.05,
                 roll_range: float = 0.05,
                 init_rand_scale: float = 1.0,
                 # --- head-cam / perception knobs ---
                 enable_headcam: bool = True,
                 headcam_name: str = "head_cam",
                 headcam_size=(640, 480),
                 headcam_for_eval_only: bool = True,
                 auto_choose_nearer_side: bool = False,   # IMPORTANT: fixed-right/right, fixed-left/left
                 **kwargs):
        if kwargs:
            import warnings
            warnings.warn(f"Ignoring extra kwargs: {list(kwargs.keys())}")

        if not os.path.isfile(scene_xml_path):
            raise FileNotFoundError(scene_xml_path)

        self.model = MjModel.from_xml_path(scene_xml_path)
        self.data  = MjData(self.model)

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.randomize_init = bool(randomize_init)

        # hand side
        self.right_side = hand.lower().startswith("r")

        # control sets (4-DoF upper arm + elbow)
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
            self.other_joint_names = [
                "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
                "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
            ]
            self.palm_site_name = "palm_site_right"
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
            self.other_joint_names = [
                "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
                "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
            ]
            self.palm_site_name = "palm_site_left"

        # actuators resolved by name (must exist in XML)
        self.ctrl_act_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.ctrl_joint_names
        ], dtype=int)
        if np.any(self.ctrl_act_ids < 0):
            raise RuntimeError(f"Missing actuators for {self.ctrl_joint_names}")

        self.ctrl_jids = np.array([int(self.model.actuator_trnid[a,0]) for a in self.ctrl_act_ids])
        self.ctrl_qadr = self.model.jnt_qposadr[self.ctrl_jids]
        self.ctrl_dadr = self.model.jnt_dofadr[self.ctrl_jids]
        self.ctrl_range = self.model.jnt_range[self.ctrl_jids].copy()

        # PD
        self.kp = float(kp); self.kd = float(kd)
        self.kp_vec = np.full(len(self.ctrl_act_ids), self.kp, dtype=np.float64)
        self.kd_vec = np.full(len(self.ctrl_act_ids), self.kd, dtype=np.float64)
        self.action_scale = float(action_scale)
        self.torque_limit_vec = np.array(torque_limits, dtype=np.float64)
        self.max_joint_step = 0.05  # rad/step

        # rewards
        self.progress_coef = float(progress_coef)
        self.side_weight   = float(side_weight)
        self.touch_penalty = float(touch_penalty)
        self.ctrl_cost_scale = float(ctrl_cost_scale)
        self.time_penalty = float(time_penalty)
        self.upright_coef = float(upright_coef)
        self.lookat_coef  = float(lookat_coef)
        self.elbow_pref   = float(elbow_pref)
        self.elbow_coef   = float(elbow_coef)

        # generalization
        self.domain_randomize = bool(domain_randomize)
        self.size_jitter_frac = float(size_jitter_frac)
        self.friction_jitter_frac = float(friction_jitter_frac)
        self.ws_center = np.array(workspace_center, dtype=np.float64)
        self.ws_half   = np.array(workspace_half_range, dtype=np.float64)
        self.yaw_range   = float(yaw_range)
        self.pitch_range = float(pitch_range)
        self.roll_range  = float(roll_range)
        self.rand_scale  = float(init_rand_scale)
        self.auto_choose_nearer_side = bool(auto_choose_nearer_side)

        # === grasping target parameters ===
        self.standoff      = float(standoff)
        self.standoff_tol  = float(standoff_tol)
        self.side_margin   = float(side_margin)
        self.touch_penalty = float(touch_penalty)
        self.ctrl_cost_scale = float(ctrl_cost_scale)

        # can shape/ids
        self.can_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom")
        if self.can_gid < 0:
            raise RuntimeError("Geom 'can_geom' not found.")
        sz = self.model.geom_size[self.can_gid].copy()
        self.can_radius = float(sz[0]); self.can_half_h = float(sz[1])
        self._base_can_size = sz.copy()
        self._base_can_fric = self.model.geom_friction[self.can_gid].copy()
        self.min_radial_gap = self.can_radius + 0.006
        self.can_free = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
        if self.can_free < 0 and self.randomize_init:
            print("[env] WARNING: randomize_init=True but can has no free joint; pose randomization will be limited.")

        # palm site
        self.palm_sid = named_site_id(self.model, self.palm_site_name)

        # wrist & other‑side locks
        self.freeze_other = bool(freeze_other)
        self.wrist_jids = _joint_ids(self.model, self.wrist_joint_names)
        self.other_jids = _joint_ids(self.model, self.other_joint_names)
        self.wrist_qadr = np.array([self.model.jnt_qposadr[j] for j in self.wrist_jids], dtype=int) if self.wrist_jids else np.array([], dtype=int)
        self.wrist_dadr = np.array([self.model.jnt_dofadr[j] for j in self.wrist_jids], dtype=int) if self.wrist_jids else np.array([], dtype=int)
        self.other_qadr = np.array([self.model.jnt_qposadr[j] for j in self.other_jids], dtype=int)
        self.other_dadr = np.array([self.model.jnt_dofadr[j] for j in self.other_jids], dtype=int)
        self.wrist_q_fixed = None; self.other_q_fixed = None

        # elbow index
        elbow_name = "right_elbow_joint" if self.right_side else "left_elbow_joint"
        ej = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, elbow_name)
        self.elbow_qadr = self.model.jnt_qposadr[ej] if ej >= 0 else None

        # desired q for PD
        self.des_q = np.zeros(len(self.ctrl_qadr), dtype=np.float64)

        # observation space: q,qd + can(7) + target_rel(3)
        obs_dim = len(self.ctrl_qadr)*2 + 3 + 4 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(len(self.ctrl_act_ids),), dtype=np.float32)

        # === head camera ===
        self.enable_headcam = bool(enable_headcam)
        self.headcam_name   = headcam_name
        self.headcam_size   = tuple(int(x) for x in headcam_size)
        self.headcam_for_eval_only = bool(headcam_for_eval_only)
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.headcam_name) if self.enable_headcam else -1
        self.cam_fx = self.cam_fy = self.cam_cx = self.cam_cy = None
        self._maybe_setup_headcam_intrinsics()
        self._off_renderer = None  # allocated lazily to avoid overhead in training

        # misc
        self.viewer = None
        self._prev_d = None
        self.step_count = 0

    # ---------- head-cam helpers ----------
    def _maybe_setup_headcam_intrinsics(self):
        if not self.enable_headcam or self.cam_id < 0:
            return
        # fovy is degrees in MuJoCo model
        fovy_deg = float(self.model.cam_fovy[self.cam_id])
        fovy = np.deg2rad(fovy_deg)
        W, H = self.headcam_size
        self.cam_cx = 0.5 * W
        self.cam_cy = 0.5 * H
        self.cam_fy = 0.5 * H / np.tan(0.5 * fovy)
        self.cam_fx = self.cam_fy  # square pixels assumption

    def _world_to_cam(self, Xw: np.ndarray):
        """Return (Xc, valid) with Xc in camera coords."""
        if self.cam_id < 0: return None, False
        C = self.data.cam_xpos[self.cam_id]
        R = self.data.cam_xmat[self.cam_id].reshape(3,3)
        # world->cam: R^T (X - C)
        Xc = R.T @ (Xw - C)
        # Positive Z means in front, but MuJoCo uses -Z forward convention for GL.
        # For projection we require Xc[2] > 1e-6.
        return Xc, (Xc[2] > 1e-6)

    def _project(self, Xw: np.ndarray):
        """Project world point to pixel (u,v) if possible."""
        if self.cam_fx is None: return None, False
        Xc, ok = self._world_to_cam(Xw)
        if not ok: return None, False
        u = self.cam_fx * (Xc[0] / Xc[2]) + self.cam_cx
        v = self.cam_fy * (Xc[1] / Xc[2]) + self.cam_cy
        return np.array([u, v], dtype=np.float32), True

    def get_headcam_rgb(self):
        """Return BGR (H,W,3) from the head camera for eval/visualization."""
        if not self.enable_headcam:
            return None
        if self.headcam_for_eval_only and self.render_mode != "human":
            # keep train loop fast
            return None
        if not HAVE_MJ_RENDERER:
            return None
        if self._off_renderer is None:
            W,H = self.headcam_size
            self._off_renderer = MJRenderer(self.model, W, H)
        self._off_renderer.update_scene(self.data, camera=self.headcam_name)
        rgb = self._off_renderer.render()
        # Convert RGB->BGR for OpenCV
        return rgb[:, :, ::-1].copy()

    # ---------- geometry ----------
    def _can_frame(self):
        pos = self.data.geom_xpos[self.can_gid].copy()
        R   = self.data.geom_xmat[self.can_gid].reshape(3,3).copy()
        y_axis = R[:,1]; z_axis = R[:,2]
        return pos, y_axis, z_axis, R

    def _target_pos(self):
        can_center, y_axis, _, _ = self._can_frame()
        # fixed side per chosen hand unless auto mode is enabled
        if self.auto_choose_nearer_side:
            palm = self.data.site_xpos[self.palm_sid]
            sgn = -1.0 if np.dot(palm - can_center, y_axis) >= 0 else +1.0
        else:
            sgn = (-1.0 if self.right_side else +1.0)
        return can_center + sgn * (self.can_radius + self.standoff) * y_axis

    def _touching_can(self) -> bool:
        cid = self.can_gid
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if int(c.geom1) == cid or int(c.geom2) == cid:
                return True
        return False

    # ---------- randomization ----------
    def set_randomization_scale(self, scale: float):
        self.rand_scale = float(np.clip(scale, 0.2, 1.5))

    def _domain_randomize_can(self):
        if not self.domain_randomize:
            return
        # jitter size
        sz = self._base_can_size.copy()
        f = self.size_jitter_frac
        sz[0] *= (1.0 + np.random.uniform(-f,+f))  # radius
        sz[1] *= (1.0 + np.random.uniform(-f,+f))  # half-height
        self.model.geom_size[self.can_gid] = sz
        self.can_radius = float(sz[0]); self.can_half_h = float(sz[1])
        self.min_radial_gap = self.can_radius + 0.006
        # jitter friction
        fr = self._base_can_fric.copy()
        g  = self.friction_jitter_frac
        self.model.geom_friction[self.can_gid] = fr * (1.0 + np.random.uniform(-g,+g, size=3))

    def _randomize_pose(self):
        if self.can_free < 0:  # nothing to randomize
            return
        self._domain_randomize_can()
        adr = self.model.jnt_qposadr[self.can_free]
        s   = self.rand_scale
        x0,y0,z0 = self.ws_center
        hx,hy,hz = self.ws_half

        # Hand-aware Y region: right→[0, +hy], left→[-hy, 0]
        if self.right_side:
            y = y0 + np.random.uniform(0, +hy*s)
        else:
            y = y0 + np.random.uniform(-hy*s, 0)

        x = x0 + np.random.uniform(-hx*s, +hx*s)
        z = z0 + np.random.uniform(-hz*s, +hz*s)
        self.data.qpos[adr:adr+3] = np.array([x,y,z], dtype=np.float64)

        yaw   = np.random.uniform(-self.yaw_range*s, +self.yaw_range*s)
        pitch = np.random.uniform(-self.pitch_range*s, +self.pitch_range*s)
        roll  = np.random.uniform(-self.roll_range*s,  +self.roll_range*s)

        cy,sy = math.cos(yaw/2),   math.sin(yaw/2)
        cp,sp = math.cos(pitch/2), math.sin(pitch/2)
        cr,sr = math.cos(roll/2),  math.sin(roll/2)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        self.data.qpos[adr+3:adr+7] = np.array([qw,qx,qy,qz], dtype=np.float64)

        dof = self.model.jnt_dofadr[self.can_free]
        self.data.qvel[dof:dof+6] = 0.0

    # ---------- locks ----------
    def _record_freezes(self):
        if self.freeze_other and self.other_qadr.size>0:
            self.other_q_fixed = self.data.qpos[self.other_qadr].copy()
        if self.wrist_qadr.size>0:
            self.wrist_q_fixed = self.data.qpos[self.wrist_qadr].copy()

    def _enforce_freezes(self):
        if self.freeze_other and self.other_q_fixed is not None:
            self.data.qpos[self.other_qadr] = self.other_q_fixed
            self.data.qvel[self.other_dadr] = 0.0
        if self.wrist_q_fixed is not None and self.wrist_qadr.size>0:
            self.data.qpos[self.wrist_qadr] = self.wrist_q_fixed
            self.data.qvel[self.wrist_dadr] = 0.0

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # safe default poses (both arms)
        if self.right_side:
            _set_qpos_if(self.model, self.data, "right_shoulder_pitch_joint", -0.15)
            _set_qpos_if(self.model, self.data, "right_shoulder_roll_joint",  -0.25)
            _set_qpos_if(self.model, self.data, "right_shoulder_yaw_joint",    0.00)
            _set_qpos_if(self.model, self.data, "right_elbow_joint",           0.35)
        else:
            _set_qpos_if(self.model, self.data, "left_shoulder_pitch_joint", -0.15)
            _set_qpos_if(self.model, self.data, "left_shoulder_roll_joint",   0.25)
            _set_qpos_if(self.model, self.data, "left_shoulder_yaw_joint",    0.00)
            _set_qpos_if(self.model, self.data, "left_elbow_joint",           0.35)

        # level wrists and lock snapshot
        for jn in self.wrist_joint_names:
            _set_qpos_if(self.model, self.data, jn, 0.0)

        if self.randomize_init:
            self._randomize_pose()

        mujoco.mj_forward(self.model, self.data)

        self._record_freezes()
        self._enforce_freezes()

        # PD reference starts at current
        self.des_q = self.data.qpos[self.ctrl_qadr].copy()
        self.step_count = 0; self._prev_d = None

        return self._get_obs(), {}

    def _apply_action(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        proposed = self.des_q + self.action_scale * action
        proposed = np.clip(proposed, self.ctrl_range[:,0], self.ctrl_range[:,1])
        delta = np.clip(proposed - self.des_q, -self.max_joint_step, self.max_joint_step)
        self.des_q = self.des_q + delta

        q  = self.data.qpos[self.ctrl_qadr].astype(np.float64)
        qd = self.data.qvel[self.ctrl_dadr].astype(np.float64)
        tau = self.kp_vec*(self.des_q - q) - self.kd_vec*qd
        tau = np.clip(tau, -self.torque_limit_vec[:len(tau)], self.torque_limit_vec[:len(tau)])
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.ctrl_act_ids] = tau

    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        q  = self.data.qpos[self.ctrl_qadr].copy()
        qd = self.data.qvel[self.ctrl_dadr].copy()
        can_center,_,_,R = self._can_frame()
        can_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(can_quat, R.ravel())
        palm = self.data.site_xpos[self.palm_sid].copy()
        rel  = self._target_pos() - palm
        return np.concatenate([q, qd, can_center, can_quat, rel]).astype(np.float32)

    def step(self, action):
        # keep locks
        self._enforce_freezes()
        self._apply_action(action)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            self._enforce_freezes()
        mujoco.mj_forward(self.model, self.data)

        palm = self.data.site_xpos[self.palm_sid].copy()
        can_center, y_axis, z_axis, _ = self._can_frame()
        target = self._target_pos()

        # orientation: palm frame from site_xmat
        R_palm = self.data.site_xmat[self.palm_sid].reshape(3,3).copy()
        palm_forward = R_palm[:,0]   # +X as “fingers forward”
        palm_up      = R_palm[:,2]   # +Z as palm normal (parallel to world Z)

        world_z = np.array([0,0,1.0], dtype=np.float64)

        # upright (parallel to ground)
        upright_dot = float(np.clip(np.dot(palm_up, world_z), -1.0, 1.0))
        upright_pen = self.upright_coef * (1.0 - max(0.0, upright_dot))

        # look-at-can
        to_can = can_center - palm
        nc = np.linalg.norm(to_can) + 1e-9
        to_can /= nc
        lookat_pen = self.lookat_coef * (1.0 - max(0.0, float(np.dot(palm_forward, to_can))))

        # elbow posture
        elbow_pen = 0.0
        if self.elbow_qadr is not None:
            q_el = float(self.data.qpos[self.elbow_qadr])
            elbow_pen = self.elbow_coef * (q_el - self.elbow_pref)**2

        # distances
        d_target = float(np.linalg.norm(target - palm))
        if self._prev_d is None: self._prev_d = d_target
        progress = self._prev_d - d_target
        self._prev_d = d_target

        vec_cp = palm - can_center
        vec_perp = vec_cp - np.dot(vec_cp, z_axis)*z_axis
        radial = float(np.linalg.norm(vec_perp))
        inside_gap = (self.min_radial_gap - radial)
        inner_barrier = 120.0*(inside_gap*inside_gap) if inside_gap>0 else 0.0

        # side penalty (hand-specific)
        if self.auto_choose_nearer_side:
            sgn = -1.0 if np.dot(palm - can_center, y_axis) >= 0 else +1.0
            side_progress = sgn * float(np.dot(palm - can_center, y_axis))
        else:
            sgn = (-1.0 if self.right_side else +1.0)
            side_progress = sgn * float(np.dot(palm - can_center, y_axis))
        side_violation = max(0.0, self.side_margin - side_progress)
        side_pen = side_violation**2

        # mild vertical gating only when near lateral surface
        near_lateral = radial < (self.can_radius + self.standoff + 0.015)
        topdown_pen = 0.0
        if near_lateral:
            vertical_dev = abs(float(np.dot(vec_cp, z_axis)))
            topdown_pen = 0.2 * max(0.0, vertical_dev - self.can_half_h*0.4)

        touching = self._touching_can()
        touch_pen = self.touch_penalty if touching else 0.0

        ctrl_pen = self.ctrl_cost_scale * float(np.sum(self.data.ctrl[self.ctrl_act_ids]**2))
        qd = self.data.qvel[self.ctrl_dadr]
        vel_smooth = 1e-4 * float(np.sum(qd**2))

        approach_r = self.progress_coef * progress

        reward = (
            approach_r
            - self.side_weight * side_pen
            - inner_barrier
            - topdown_pen
            - touch_pen
            - ctrl_pen
            - vel_smooth
            - self.time_penalty
            - upright_pen
            - lookat_pen
            - elbow_pen
        )

        self.step_count += 1
        success = (d_target < self.standoff_tol) and (side_violation == 0.0) and (not touching) and (inner_barrier == 0.0)
        terminated = bool(success)
        truncated  = bool(self.step_count >= self.max_steps)

        # --- head-cam “perception stub” (distance + pixel) ---
        headcam_seen, head_uv, head_dist = False, None, None
        if self.enable_headcam and self.cam_id >= 0:
            head_uv, ok = self._project(can_center)
            if ok:
                headcam_seen = True
                C = self.data.cam_xpos[self.cam_id]
                head_dist = float(np.linalg.norm(can_center - C))

        obs = self._get_obs()
        info = {
            "is_success": success,
            "d_target": d_target,
            "radial": radial,
            "approach_r": approach_r,
            "side_pen": side_pen,
            "inner_barrier": inner_barrier,
            "topdown_pen": topdown_pen,
            "upright_pen": upright_pen,
            "lookat_pen": lookat_pen,
            "elbow_pen": elbow_pen,
            "headcam_seen": headcam_seen,
            "headcam_uv": None if head_uv is None else head_uv.tolist(),
            "headcam_dist_m": head_dist,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if not HAVE_MJ_VIEWER:
            raise RuntimeError("mujoco.viewer not available; set MUJOCO_GL=glfw and install mujoco>=2.3.6.")
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)
            # Big target marker so you can SEE it
            try:
                pos = self._target_pos()
                self.viewer.add_marker(pos=pos, size=(0.07,0.07,0.07),
                                       rgba=(1,0,1,0.9), type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                       label="TARGET")
            except Exception:
                pass
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None