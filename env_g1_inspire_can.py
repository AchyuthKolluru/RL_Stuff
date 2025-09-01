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


def _set_joint_if_exists(model, data, joint_name, value):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if j >= 0:
        adr = model.jnt_qposadr[j]
        data.qpos[adr] = float(value)


def _joint_ids(model, names):
    ids = []
    for n in names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            ids.append(j)
    return ids


def _collect_hand_actuator_ids(model, side):
    """Collect any actuators that drive joints starting with '<side>_hand_'.
    Works for both the 12-DOF FTX and the legacy 3-finger XMLs.
    """
    ids = []
    prefix = f"{side}_hand_"
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jname and jname.startswith(prefix):
            ids.append(a)
    return sorted(ids)


class G1InspireCanGrasp(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self,
                 scene_xml_path,
                 render_mode="none",
                 hand="right",
                 hand_names=None,
                 max_steps=400,
                 target_lift=0.03,
                 randomize_init=True,
                 ik_warm_start=True):
        super().__init__()

        if not os.path.isfile(scene_xml_path):
            raise FileNotFoundError(scene_xml_path)

        self.model = MjModel.from_xml_path(scene_xml_path)
        self.data = MjData(self.model)

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.target_lift = target_lift
        self.randomize_init = randomize_init
        self.ik_warm_start = ik_warm_start
        self.step_count = 0

        # side-specific names
        self.side = "right" if hand.lower().startswith("r") else "left"
        self.palm_site_name = "palm_site_right" if self.side == "right" else "palm_site_left"

        # choose hand actuators
        if hand_names is not None:
            # explicit list (rare)
            name_set = set(hand_names)
            ids = []
            for i in range(self.model.nu):
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if nm in name_set:
                    ids.append(i)
            self.hand_actuator_ids = sorted(ids)
        else:
            # automatically detect by joint name prefix
            self.hand_actuator_ids = _collect_hand_actuator_ids(self.model, self.side)

        if len(self.hand_actuator_ids) == 0:
            raise RuntimeError("No hand actuators found. Check XML actuator names match '<side>_hand_*'.")

        # map actuators → joints
        self.act_to_joint = np.array(
            [int(self.model.actuator_trnid[a, 0]) for a in self.hand_actuator_ids],
            dtype=int
        )
        self.jnt_qposadr = self.model.jnt_qposadr[self.act_to_joint]
        self.jnt_dofadr  = self.model.jnt_dofadr[self.act_to_joint]
        self.jnt_range   = self.model.jnt_range[self.act_to_joint].copy()

        # PD gains + torque limits
        self.kp = np.full(len(self.hand_actuator_ids), 20.0, dtype=np.float64)
        self.kd = np.full(len(self.hand_actuator_ids), 1.0, dtype=np.float64)
        # guess higher limit for thumb_0, else 1.4; clamp to 1.0 for stability
        self.hand_actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            for a in self.hand_actuator_ids
        ]
        self.torque_limit = np.minimum(
            np.array([2.45 if ("thumb_0" in n) else 1.4 for n in self.hand_actuator_names], dtype=np.float64),
            1.0
        )

        # action + obs
        self.action_scale = 0.03
        self.des_q = self.data.qpos[self.jnt_qposadr].copy()
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(len(self.hand_actuator_ids),),
                                       dtype=np.float32)

        self.can_sid = named_site_id(self.model, "can_site")
        try:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
            self.can_free_joint = j if j >= 0 else None
        except Exception:
            self.can_free_joint = None
        self.palm_sid = named_site_id(self.model, self.palm_site_name)

        obs_dim = (len(self.jnt_qposadr) * 2) + 3 + 4 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

        # arm joints for tiny IK warm start
        self.arm_joint_names = ([
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ] if self.side == "right" else [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
        ])

        self.viewer = None

    # ---------- utils ----------
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
        if self.can_free_joint is None:
            return
        adr = self.model.jnt_qposadr[self.can_free_joint]
        x = 0.42 + np.random.uniform(-0.02, 0.02)
        y = -0.03 + np.random.uniform(-0.02, 0.02)
        z = 1.03 + np.random.uniform(-0.02, 0.02)
        self.data.qpos[adr:adr+3] = np.array([x, y, z], dtype=np.float64)

        axis = np.random.randn(3); axis /= (np.linalg.norm(axis) + 1e-8)
        angle = np.random.uniform(-0.15, 0.15)
        s = math.sin(angle/2.0)
        self.data.qpos[adr+3:adr+7] = np.array([math.cos(angle/2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)

        dof = self.model.jnt_dofadr[self.can_free_joint]
        self.data.qvel[dof:dof+6] = 0.0

    def _set_safe_arm_pose(self):
        """Arms-forward pose so palms start in front, not behind legs."""
        if self.side == "left":
            _set_joint_if_exists(self.model, self.data, "left_shoulder_pitch_joint", -1.20)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_roll_joint",   0.25)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_yaw_joint",    0.00)
            _set_joint_if_exists(self.model, self.data, "left_elbow_joint",           1.60)
            _set_joint_if_exists(self.model, self.data, "left_wrist_roll_joint",      0.00)
            _set_joint_if_exists(self.model, self.data, "left_wrist_pitch_joint",     0.10)
            _set_joint_if_exists(self.model, self.data, "left_wrist_yaw_joint",       0.00)
        else:
            _set_joint_if_exists(self.model, self.data, "right_shoulder_pitch_joint", -1.20)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_roll_joint", -0.25)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_yaw_joint",   0.00)
            _set_joint_if_exists(self.model, self.data, "right_elbow_joint",          1.60)
            _set_joint_if_exists(self.model, self.data, "right_wrist_roll_joint",     0.00)
            _set_joint_if_exists(self.model, self.data, "right_wrist_pitch_joint",    0.10)
            _set_joint_if_exists(self.model, self.data, "right_wrist_yaw_joint",      0.00)

        # light finger curl
        pref = "left_" if self.side == "left" else "right_"
        vals = {
            f"{pref}hand_index_0_joint": 0.20, f"{pref}hand_index_1_joint": 0.20,
            f"{pref}hand_middle_0_joint": 0.20, f"{pref}hand_middle_1_joint": 0.20,
            f"{pref}hand_ring_0_joint": 0.15, f"{pref}hand_ring_1_joint": 0.15,
            f"{pref}hand_pinky_0_joint": 0.10, f"{pref}hand_pinky_1_joint": 0.10,
            f"{pref}hand_thumb_0_joint": 0.00, f"{pref}hand_thumb_1_joint": 0.20,
            f"{pref}hand_thumb_2_joint": 0.20, f"{pref}hand_thumb_3_joint": 0.20,
        }
        for jn, v in vals.items():
            _set_joint_if_exists(self.model, self.data, jn, v)

    def _ik_palm_to_can(self, target_offset=np.array([-0.08, 0.0, 0.03]), iters=18, damping=1e-2):
        """Tiny damped-Least-Squares IK to bring palm near the can (reset only)."""
        j_ids = _joint_ids(self.model, self.arm_joint_names)
        if not j_ids:
            return
        dof_ids = np.array([self.model.jnt_dofadr[j] for j in j_ids], dtype=int)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)
        can_pos = self.data.site_xpos[self.can_sid].copy()
        target = can_pos + target_offset

        for _ in range(iters):
            mujoco.mj_forward(self.model, self.data)
            palm_pos = self.data.site_xpos[self.palm_sid].copy()
            err = (target - palm_pos)
            if np.linalg.norm(err) < 0.01:
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.palm_sid)
            J = jacp[:, dof_ids]  # 3 x ndof
            JJt = J @ J.T
            A = JJt + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, err)

            for k, j in enumerate(j_ids):
                adr = self.model.jnt_qposadr[j]
                lo, hi = self.model.jnt_range[j]
                self.data.qpos[adr] = np.clip(self.data.qpos[adr] + 0.5 * dq[k], lo, hi)

        mujoco.mj_forward(self.model, self.data)

    # ---------- gym api ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._set_safe_arm_pose()
        if self.ik_warm_start:
            # behind & slightly above can to avoid immediate penetration
            self._ik_palm_to_can(target_offset=np.array([-0.08, 0.0, 0.03]), iters=18, damping=1e-2)

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

        # contact bonus
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

        height_bonus = 0.0
        if self.can_free_joint is not None:
            z = self.data.qpos[self.model.jnt_qposadr[self.can_free_joint] + 2]
            height_bonus = 6.0 * max(0.0, z - 1.02)

        reward = 2.0 * (1.0 / (0.05 + dist)) + touch_bonus + height_bonus - ctrl_penalty

        if self.can_free_joint is None:
            success = (dist < 0.04) and touched
        else:
            success = (dist < 0.05) and (z > 1.02 + self.target_lift)

        return reward, bool(success)

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