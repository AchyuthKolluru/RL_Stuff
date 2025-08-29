import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
from mujoco import MjModel, MjData

# ---------- Helpers to find hand actuators by name ----------
def find_actuators_by_prefix(model, prefix_list):
    """
    Return actuator_ids whose 'name' starts with any of the strings in prefix_list.
    This lets you adapt to actual names in your XML (e.g., 'right_hand_', 'r_hand_', 'r_finger1_', etc).
    """
    act_ids = []
    nameadr = model.name_actuatoradr
    namelen = model.name_actuatoradr[1:] - model.name_actuatoradr[:-1]
    last = model.na
    # Build array of actuator names
    names = []
    for i in range(model.nu):
        start = nameadr[i]
        end = nameadr[i+1] if i+1 < len(nameadr) else last
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

    for i, n in enumerate(names):
        if n is None:
            continue
        for p in prefix_list:
            if n.startswith(p):
                act_ids.append(i)
                break
    return sorted(list(set(act_ids)))

def named_site_id(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Site '{name}' not found. Add it in XML.")
    return sid

class G1InspireCanGrasp(gym.Env):
    """
    Single-hand grasp: control only the chosen Inspire hand actuators.
    Actions: position targets (or torque, depending on actuator type in robot XML).
    Observations: hand qpos/qvel (subset), can pose (6D), relative palm -> can.
    Reward: shaped + sparse success.
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self,
                 scene_xml_path,
                 render_mode="none",
                 hand="right",                # or "left"
                 hand_prefixes=None,          # override if needed
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
        self.step_count = 0
        self.target_lift = target_lift
        self.randomize_init = randomize_init

        # Infer hand actuator names.
        # Adjust these prefixes to match your XML. Common patterns:
        #   'right_hand_', 'right_finger', 'r_hand_', 'r_thumb_', 'r_index_', ...
        if hand_prefixes is None:
            if hand.lower().startswith("r"):
                hand_prefixes = ["right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint","right_hand_index_0_joint","right_hand_index_1_joint","right_hand_middle_0_joint","right_hand_middle_1_joint"]
            else:
                hand_prefixes = ["left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint","left_hand_index_0_joint","left_hand_index_1_joint","left_hand_middle_0_joint","left_hand_middle_1_joint"]

        self.hand_actuator_ids = find_actuators_by_prefix(self.model, hand_prefixes)
        if len(self.hand_actuator_ids) == 0:
            raise RuntimeError(f"No actuators found with prefixes {hand_prefixes}. Please open the robot XML and adjust.")

        # Build action space (one control per selected actuator).
        # If the actuators are position type, we’ll send deltas around a nominal.
        self.action_scale = 0.5  # radians or normalized units; tune if needed
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.hand_actuator_ids),), dtype=np.float32)

        # Observation space:
        # hand joint pos & vel (subset), can pose (pos(3)+quat(4)), palm->can vector(3)
        # We'll resolve a site called "can_site" (provided in scene XML) and a "palm_site" (add in hand body)
        self.can_sid = named_site_id(self.model, "can_site")

        # Try to find a 'palm_site'; if not present, fallback to wrist site/body center
        try:
            self.palm_sid = named_site_id(self.model, "palm_site_right" if hand.startswith("r") else "palm_site_left")
        except RuntimeError:
            # Fall back to a wrist site if your XML has one; else require user to add.
            # You can also compute from a named body transform:
            raise RuntimeError("Missing 'palm_site_*' in the robot XML. Please add a small site at the palm frame.")

        # Identify joint ids driven by hand actuators (for qpos/qvel observations)
        # Each actuator has a 'trnid' to a joint; gather those joint ids:
        self.hand_joint_ids = sorted(list(set(int(self.model.actuator_trnid[a, 0]) for a in self.hand_actuator_ids)))

        obs_dim = (len(self.hand_joint_ids) * 2) + 3 + 4 + 3  # qpos+qvel + can pos+quat + palm->can vec
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Rendering setup (on demand)
        self.viewer = None

    # ---------------------- Utils ----------------------
    def _get_obs(self):
        # Hand joint pos/vel
        qpos = self.data.qpos[self.model.jnt_qposadr[self.hand_joint_ids],].copy()
        qvel = self.data.qvel[self.model.jnt_dofadr[self.hand_joint_ids],].copy()

        # Can pose (from can_site)
        mujoco.mj_forward(self.model, self.data)
        can_pos = self.data.site_xpos[self.can_sid].copy()
        can_quat = self.data.site_xquat[self.can_sid].copy()

        palm_pos = self.data.site_xpos[self.palm_sid].copy()
        rel_vec = can_pos - palm_pos

        return np.concatenate([qpos, qvel, can_pos, can_quat, rel_vec]).astype(np.float32)

    def _apply_action(self, action):
        # Convert normalized action to target deltas for position actuators.
        # If your actuators are torque/velocity type, adapt this block accordingly.
        ctrl = self.data.ctrl
        # Initialize ctrl to zero for all actuators
        ctrl[:] = 0.0

        # Send deltas around current qpos for each target joint via actuator ctrl
        # A simple approach: treat ctrl as desired position if actuators are 'position' type (MuJoCo will PD).
        for a_id, a in enumerate(self.hand_actuator_ids):
            ctrl[a] = np.clip(self.data.ctrl[a] + self.action_scale * float(action[a_id]), -1.0, 1.0)

    def _randomize(self):
        # Place can near palm with small random offsets
        # Modify data.qpos for 'can_free' joint (7 dofs: 3 pos + 4 quat)
        can_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
        if can_jnt < 0:
            raise RuntimeError("Joint 'can_free' not found.")
        adr = self.model.jnt_qposadr[can_jnt]

        # Base position (x,y,z) – tune relative to your hand location
        x = 0.42 + np.random.uniform(-0.03, 0.03)
        y = 0.02 + np.random.uniform(-0.03, 0.03)
        z = 1.02 + np.random.uniform(-0.02, 0.02)
        self.data.qpos[adr:adr+3] = np.array([x, y, z])

        # Random small tilt quaternion
        def rand_quat():
            axis = np.random.randn(3); axis /= (np.linalg.norm(axis) + 1e-8)
            angle = np.random.uniform(-0.2, 0.2)
            s = math.sin(angle/2.0)
            return np.array([math.cos(angle/2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)

        self.data.qpos[adr+3:adr+7] = rand_quat()

        # Reset can velocities
        self.data.qvel[ self.model.jnt_dofadr[can_jnt] : self.model.jnt_dofadr[can_jnt]+6 ] = 0.0

    # ---------------------- Gym API ----------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Put robot into a neutral pose (if your XML has keyframes, you can mj_resetDataKeyframe)
        # Otherwise set a few important joints here as needed (wrist neutral etc).

        if self.randomize_init:
            self._randomize()

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._apply_action(action)

        # Step sim
        for _ in range(5):  # control at 100 Hz if dt=0.002  (5*0.002=0.01s per RL step)
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, success = self._compute_reward_and_success()

        self.step_count += 1
        terminated = success
        truncated = self.step_count >= self.max_steps

        info = {"is_success": success}
        return obs, reward, terminated, truncated, info

    def _compute_reward_and_success(self):
        mujoco.mj_forward(self.model, self.data)

        can_pos = self.data.site_xpos[self.can_sid]
        palm_pos = self.data.site_xpos[self.palm_sid]
        rel = can_pos - palm_pos
        dist = np.linalg.norm(rel)

        # Contacts involving can_geom
        contact_bonus = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            # If either geom is can_geom, give a small bonus (stronger if contact normal points inward)
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            if name1 == "can_geom" or name2 == "can_geom":
                contact_bonus += 0.002

        # Height bonus (lift)
        # Get can free-joint z position from qpos
        can_j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "can_free")
        z = self.data.qpos[self.model.jnt_qposadr[can_j] + 2]  # z in world
        height_bonus = 1.0 * max(0.0, z - 1.02)  # tune baseline

        # Small regularizers
        ctrl_penalty = 1e-3 * float(np.sum(self.data.ctrl[self.hand_actuator_ids]**2))

        # Total reward
        reward = (
            1.5 * (1.0 / (0.05 + dist))      # get close
            + contact_bonus                  # touch/hold
            + 6.0 * height_bonus             # lift
            - ctrl_penalty
        )

        # Success: lifted AND close to palm
        success = (z > 1.02 + self.target_lift) and (dist < 0.05)
        return reward, bool(success)

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # viewer handled outside the step loop

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None