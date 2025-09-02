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


def find_actuators_by_name(model, names_wanted):
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
                ik_warm_start=False,
                control_arm=True,
                auto_grasp=True,
                grip_synergy=True,           # <-- NEW: enable 1 extra action for group curl/open
                synergy_gain=0.06,
                hand_gate_open_dist=0.20,   # far: hand control ~0
                hand_gate_full_dist=0.08,   # near: full hand control
                auto_grasp_dist=0.07):          # <-- NEW: strength of the synergy bias per step
        """
        control_arm=True  -> the policy controls the arm joints (same side) + the fingers.
        auto_grasp=True   -> tiny closure bias for fingers when palm is near the can (keeps behavior stable).
        grip_synergy=True -> adds one extra scalar action that curls/opens all hand joints together.
        """
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
        self.control_arm = control_arm
        self.auto_grasp = auto_grasp
        self.grip_synergy = grip_synergy
        self.synergy_gain = float(synergy_gain)
        self.step_count = 0
        self.hand_gate_open_dist = float(hand_gate_open_dist)
        self.hand_gate_full_dist = float(hand_gate_full_dist)
        self.auto_grasp_dist = float(auto_grasp_dist)

        # ---------------- side selection ----------------
        right = hand.lower().startswith("r")
        if right:
            hand_actuator_names = [
                "right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint","right_hand_thumb_3_joint",
                "right_hand_index_0_joint","right_hand_index_1_joint",
                "right_hand_middle_0_joint","right_hand_middle_1_joint",
                "right_hand_ring_0_joint","right_hand_ring_1_joint",
                "right_hand_pinky_0_joint","right_hand_pinky_1_joint",
            ]
            self.arm_joint_names = [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
            self.palm_site_name = "palm_site_right"
            self.side_prefix = "right_"
        else:
            hand_actuator_names = [
                "left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint","left_hand_thumb_3_joint",
                "left_hand_index_0_joint","left_hand_index_1_joint",
                "left_hand_middle_0_joint","left_hand_middle_1_joint",
                "left_hand_ring_0_joint","left_hand_ring_1_joint",
                "left_hand_pinky_0_joint","left_hand_pinky_1_joint",
            ]
            self.arm_joint_names = [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]
            self.palm_site_name = "palm_site_left"
            self.side_prefix = "left_"

        # ---------------- actuators we control ----------------
        self.hand_actuator_ids = find_actuators_by_name(self.model, hand_actuator_names)
        if len(self.hand_actuator_ids) == 0:
            raise RuntimeError(f"No actuators found for names {hand_actuator_names}. Check XML.")

        if self.control_arm:
            arm_actuator_names = self.arm_joint_names  # actuator names == joint names in your XML
            self.arm_actuator_ids = find_actuators_by_name(self.model, arm_actuator_names)
            if len(self.arm_actuator_ids) == 0:
                self.control_arm = False
                self.arm_actuator_ids = []
        else:
            self.arm_actuator_ids = []

        # order: ARM (if any) then HAND
        self.ctrl_actuator_ids = np.array(self.arm_actuator_ids + self.hand_actuator_ids, dtype=int)
        self.n_arm = len(self.arm_actuator_ids)
        self.n_hand = len(self.hand_actuator_ids)
        self.n_total = len(self.ctrl_actuator_ids)

        # Map actuators → joints (for the ones we control)
        self.ctrl_to_joint = np.array(
            [int(self.model.actuator_trnid[a, 0]) for a in self.ctrl_actuator_ids],
            dtype=int
        )
        self.ctrl_jnt_qposadr = self.model.jnt_qposadr[self.ctrl_to_joint]
        self.ctrl_jnt_dofadr  = self.model.jnt_dofadr[self.ctrl_to_joint]
        self.ctrl_jnt_range   = self.model.jnt_range[self.ctrl_to_joint].copy()

        # ---------------- PD gains + torque limits ----------------
        kp_hand = 16.0
        kd_hand = 0.8
        hand_names_all = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self.hand_actuator_ids]
        hand_tl = np.array([min(2.45 if ("thumb_0" in n) else 1.4, 1.0) for n in hand_names_all], dtype=np.float64)

        kp_arm = 12.0
        kd_arm = 1.0
        arm_tl = np.array([12, 10, 10, 8, 5, 5, 5], dtype=np.float64)[:self.n_arm]

        self.kp_vec = np.concatenate([
            np.full(self.n_arm,  kp_arm, dtype=np.float64),
            np.full(self.n_hand, kp_hand, dtype=np.float64)
        ])
        self.kd_vec = np.concatenate([
            np.full(self.n_arm,  kd_arm, dtype=np.float64),
            np.full(self.n_hand, kd_hand, dtype=np.float64)
        ])
        self.torque_limit_vec = np.concatenate([arm_tl, hand_tl]) if self.n_arm > 0 else hand_tl

        # ---------------- action scaling ----------------
        arm_action_scale = 0.02
        hand_action_scale = 0.02
        self.action_scale_vec = np.concatenate([
            np.full(self.n_arm,  arm_action_scale, dtype=np.float64),
            np.full(self.n_hand, hand_action_scale, dtype=np.float64)
        ])

        # Desired joint positions (for the joints we control)
        self.des_q = self.data.qpos[self.ctrl_jnt_qposadr].copy()

        # Action/observation spaces
        self.base_action_dim = self.n_total
        self.has_synergy = bool(self.grip_synergy)
        self.action_dim = self.base_action_dim + (1 if self.has_synergy else 0)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

        # Observation: (qpos,qvel of all controlled joints) + can pos (3) + can quat (4) + palm→can vec (3)
        self.can_sid = named_site_id(self.model, "can_site")
        self.palm_sid = named_site_id(self.model, self.palm_site_name)
        # cache can geom id
        self.can_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "can_geom")
        try:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
            self.can_free_joint = j if j >= 0 else None
        except Exception:
            self.can_free_joint = None

        obs_dim = (len(self.ctrl_jnt_qposadr) * 2) + 3 + 4 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # ---------------- sets used for wrap reward ----------------
        labels = {
            "thumb":  [f"{self.side_prefix}thumb_base", f"{self.side_prefix}thumb_cmc", f"{self.side_prefix}thumb_mcp", f"{self.side_prefix}thumb_ip"],
            "index":  [f"{self.side_prefix}index_base", f"{self.side_prefix}index_mid"],
            "middle": [f"{self.side_prefix}middle_base", f"{self.side_prefix}middle_mid"],
            "ring":   [f"{self.side_prefix}ring_base", f"{self.side_prefix}ring_mid"],
            "pinky":  [f"{self.side_prefix}pinky_base", f"{self.side_prefix}pinky_mid"],
        }
        self.bodyid_to_finger = {}
        for lab, bnames in labels.items():
            for bn in bnames:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bn)
                if bid >= 0:
                    self.bodyid_to_finger[bid] = lab

        # flexion joint groups for closure reward
        self.flexion_joint_names = [
            f"{self.side_prefix}hand_index_0_joint",  f"{self.side_prefix}hand_index_1_joint",
            f"{self.side_prefix}hand_middle_0_joint", f"{self.side_prefix}hand_middle_1_joint",
            f"{self.side_prefix}hand_ring_0_joint",   f"{self.side_prefix}hand_ring_1_joint",
            f"{self.side_prefix}hand_pinky_0_joint",  f"{self.side_prefix}hand_pinky_1_joint",
            f"{self.side_prefix}hand_thumb_1_joint",  f"{self.side_prefix}hand_thumb_2_joint", f"{self.side_prefix}hand_thumb_3_joint",
        ]
        self.thumb0_name = f"{self.side_prefix}hand_thumb_0_joint"

        self.flexion_jids = []
        self.flexion_ranges = []
        for jn in self.flexion_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                self.flexion_jids.append(jid)
                self.flexion_ranges.append(self.model.jnt_range[jid].copy())
        self.flexion_jids = np.array(self.flexion_jids, dtype=int)
        self.flexion_ranges = np.array(self.flexion_ranges, dtype=np.float64)

        self.thumb0_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.thumb0_name)
        self.thumb0_range = self.model.jnt_range[self.thumb0_jid].copy() if self.thumb0_jid >= 0 else None

        self.viewer = None

    # ---------------- utils ----------------
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

        # split synergy
        if self.has_synergy:
            a_main = action[:self.base_action_dim]
            s = float(action[-1])
        else:
            a_main, s = action, 0.0

        # --- Proximity gating factor p in [0,1] ---
        palm = self.data.site_xpos[self.palm_sid]
        canp = self.data.site_xpos[self.can_sid]
        d = float(np.linalg.norm(palm - canp))

        d_open = self.hand_gate_open_dist    # e.g., 0.20 m
        d_full = self.hand_gate_full_dist    # e.g., 0.08 m
        if d <= d_full:
            p = 1.0
        elif d >= d_open:
            p = 0.0
        else:
            # smooth ramp; you can square it for a softer start
            p = (d_open - d) / max(1e-6, (d_open - d_full))
            p = max(0.0, min(1.0, p))**1.5   # ^1.5 makes it gentler far away

        # scale hand action and synergy by p; arm actions are untouched
        if self.n_hand > 0:
            a_scaled = a_main.copy()
            a_scaled[self.n_arm:self.n_total] *= p
        else:
            a_scaled = a_main

        # incremental position targets (arm+hand)
        self.des_q = np.clip(
            self.des_q + (self.action_scale_vec * a_scaled),
            self.ctrl_jnt_range[:, 0], self.ctrl_jnt_range[:, 1]
        )

        # --- Auto-grasp only when very close ---
        if self.auto_grasp and self.n_hand > 0 and d < self.auto_grasp_dist:
            hand_slice = slice(self.n_arm, self.n_total)
            # small bias that grows as we approach d_full
            t = (self.auto_grasp_dist - d) / max(1e-6, self.auto_grasp_dist - d_full)
            t = max(0.0, min(1.0, t))
            bias = 0.004 * t
            self.des_q[hand_slice] = np.clip(
                self.des_q[hand_slice] + bias,
                self.ctrl_jnt_range[hand_slice, 0],
                self.ctrl_jnt_range[hand_slice, 1]
            )

        # --- Grip synergy: also scaled by p ---
        if self.has_synergy and self.n_hand > 0 and abs(s) > 1e-6 and p > 0.0:
            hand_slice = slice(self.n_arm, self.n_total)
            lo = self.ctrl_jnt_range[hand_slice, 0]
            hi = self.ctrl_jnt_range[hand_slice, 1]
            qh = self.des_q[hand_slice]
            target = hi if s >= 0.0 else lo
            # scale synergy by p, so synergy is ineffective when far
            bias_step = (self.synergy_gain * p) * s * (target - qh)
            self.des_q[hand_slice] = np.clip(qh + bias_step, lo, hi)

        # PD control (unchanged)
        q  = self.data.qpos[self.ctrl_jnt_qposadr].astype(np.float64)
        qd = self.data.qvel[self.ctrl_jnt_dofadr].astype(np.float64)
        tau = self.kp_vec * (self.des_q - q) - self.kd_vec * qd
        tau = np.clip(tau, -self.torque_limit_vec, self.torque_limit_vec)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.ctrl_actuator_ids] = tau

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

    # ---------- Start pose + (optional) IK warm start ----------
    def _set_safe_arm_pose(self, hand_side):
        """Arms-down pose beside torso, neutral wrists. Gentle finger curl."""
        if hand_side == "left":
            _set_joint_if_exists(self.model, self.data, "left_shoulder_pitch_joint", -0.10)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_roll_joint",   0.20)
            _set_joint_if_exists(self.model, self.data, "left_shoulder_yaw_joint",    0.00)
            _set_joint_if_exists(self.model, self.data, "left_elbow_joint",           0.30)
            _set_joint_if_exists(self.model, self.data, "left_wrist_roll_joint",      0.00)
            _set_joint_if_exists(self.model, self.data, "left_wrist_pitch_joint",     0.00)
            _set_joint_if_exists(self.model, self.data, "left_wrist_yaw_joint",       0.00)
        else:
            _set_joint_if_exists(self.model, self.data, "right_shoulder_pitch_joint", -0.10)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_roll_joint", -0.20)
            _set_joint_if_exists(self.model, self.data, "right_shoulder_yaw_joint",   0.00)
            _set_joint_if_exists(self.model, self.data, "right_elbow_joint",          0.30)
            _set_joint_if_exists(self.model, self.data, "right_wrist_roll_joint",     0.00)
            _set_joint_if_exists(self.model, self.data, "right_wrist_pitch_joint",    0.00)
            _set_joint_if_exists(self.model, self.data, "right_wrist_yaw_joint",      0.00)

        # mild finger curl
        for prefix in (["left_"] if hand_side=="left" else ["right_"]):
                vals = {
                    f"{prefix}hand_index_0_joint": 0.0, f"{prefix}hand_index_1_joint": 0.0,
                    f"{prefix}hand_middle_0_joint": 0.0, f"{prefix}hand_middle_1_joint": 0.0,
                    f"{prefix}hand_ring_0_joint": 0.0,  f"{prefix}hand_ring_1_joint": 0.0,
                    f"{prefix}hand_pinky_0_joint": 0.0, f"{prefix}hand_pinky_1_joint": 0.0,
                    f"{prefix}hand_thumb_0_joint": 0.0,
                    f"{prefix}hand_thumb_1_joint": 0.0, f"{prefix}hand_thumb_2_joint": 0.0, f"{prefix}hand_thumb_3_joint": 0.0,
                }
                for jn, v in vals.items():
                    _set_joint_if_exists(self.model, self.data, jn, v)

    def _ik_palm_to_can(self, target_offset=np.array([-0.06, 0.0, 0.02]), iters=12, damping=1e-2):
        """Small damped-least-squares IK to gently bring the palm near the can (optional)."""
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
            J = jacp[:, dof_ids]
            JJt = J @ J.T
            A = JJt + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, err)

            for k, j in enumerate(j_ids):
                adr = self.model.jnt_qposadr[j]
                lo, hi = self.model.jnt_range[j]
                self.data.qpos[adr] = np.clip(self.data.qpos[adr] + 0.20 * dq[k], lo, hi)

        mujoco.mj_forward(self.model, self.data)

    # ---------------- gym api ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Arms-down pose
        side = "right" if self.palm_site_name.endswith("right") else "left"
        self._set_safe_arm_pose(side)

        if self.ik_warm_start and self.control_arm:
            default_offset = np.array([-0.08, 0.0, 0.03])
            self._ik_palm_to_can(target_offset=default_offset, iters=12, damping=1e-2)

        self.des_q = self.data.qpos[self.ctrl_jnt_qposadr].copy()

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

    # ---------- reward helpers ----------
    @staticmethod
    def _angular_coverage(angles):
        """Return coverage (0..2π) of a set of angles on the circle."""
        if len(angles) < 2:
            return 0.0
        ang = np.sort((np.array(angles, dtype=np.float64) + np.pi) % (2*np.pi))  # [0, 2π)
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2*np.pi]))
        if gaps.size == 0:
            return 0.0
        max_gap = float(np.max(gaps))
        return float(2*np.pi - max_gap)  # covered arc length

    def _compute_reward_and_success(self):
        mujoco.mj_forward(self.model, self.data)

        can_pos = self.data.site_xpos[self.can_sid]
        palm_pos = self.data.site_xpos[self.palm_sid]
        dist = np.linalg.norm(can_pos - palm_pos)

        # ---------------- contact-based wrap features ----------------
        touching_fingers = set()
        contact_angles = []
        penetration_pen = 0.0
        touched = False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue

            if self.can_geom_id >= 0 and (g1 == self.can_geom_id or g2 == self.can_geom_id):
                touched = True
                other = g2 if g1 == self.can_geom_id else g1

                # which finger label does that geom's body belong to?
                bid = int(self.model.geom_bodyid[other])
                lab = self.bodyid_to_finger.get(bid, None)
                if lab is not None:
                    touching_fingers.add(lab)

                # angle around can in XY
                p = np.array(c.pos, dtype=np.float64)
                v = p - can_pos
                ang = math.atan2(v[1], v[0])
                contact_angles.append(ang)

                # mild penalty for deep penetration
                if c.dist < 0.0:
                    penetration_pen += float((-c.dist) ** 2)

        # ---------- reward components ----------
        approach_r = 2.0 * (1.0 / (0.05 + dist))
        touch_r = 0.01 * (1.0 if touched else 0.0)

        distinct_r = 0.03 * min(5, len(touching_fingers))  # up to 5 fingers
        coverage = self._angular_coverage(contact_angles)  # 0..2π
        coverage_r = 0.04 * min(1.0, coverage / (np.pi * 2/3))  # ~120° gives good bonus

        closure_r = 0.0
        if touched and len(self.flexion_jids) > 0:
            vals = []
            for jid, (lo, hi) in zip(self.flexion_jids, self.flexion_ranges):
                adr = self.model.jnt_qposadr[jid]
                q = float(self.data.qpos[adr])
                n = (q - lo) / max(1e-6, (hi - lo))
                vals.append(np.clip(n, 0.0, 1.0))
            closure_r = 0.02 * float(np.mean(vals))

            if self.thumb0_jid >= 0 and self.thumb0_range is not None:
                lo, hi = self.thumb0_range
                adr = self.model.jnt_qposadr[self.thumb0_jid]
                q = float(self.data.qpos[adr])
                mid = 0.5 * (lo + hi)
                band = 0.25 * (hi - lo)
                opp = 1.0 - min(1.0, abs(q - mid) / max(1e-6, band))
                closure_r += 0.01 * opp

        ctrl_penalty = 1e-3 * float(np.sum(self.data.ctrl[self.ctrl_actuator_ids] ** 2))
        pen_penalty = 0.5 * penetration_pen

        height_bonus = 0.0
        lifted = False
        if self.can_free_joint is not None:
            z = self.data.qpos[self.model.jnt_qposadr[self.can_free_joint] + 2]
            height_bonus = 6.0 * max(0.0, z - 1.02)
            lifted = (z > 1.02 + self.target_lift)

        reward = (
            approach_r
            + touch_r
            + distinct_r
            + coverage_r
            + closure_r
            + height_bonus
            - ctrl_penalty
            - pen_penalty
        )

        has_thumb = ("thumb" in touching_fingers)
        multi_finger = (len(touching_fingers) >= 3) and has_thumb
        close_enough = (dist < 0.06)

        if self.can_free_joint is None:
            success = multi_finger and close_enough
        else:
            success = lifted and (len(touching_fingers) >= 2) and close_enough

        return reward, bool(success)

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