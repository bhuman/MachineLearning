import os

from isaacgym import gymtorch, gymapi, gymutil
assert gymtorch
import torch
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)
import time
import math
import numpy as np
from .base_task import BaseTask
from utils.utils import apply_randomization


class T1_Stand_Up(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]

        asset_cfg = self.cfg["asset"]
        robot_asset_simple = self._get_robot_asset(asset_cfg["file"])
        robot_asset_mesh = self._get_robot_asset(asset_cfg["file_mesh"])
        dof_props = self.gym.get_asset_dof_properties(robot_asset_simple)
        dof_props["damping"].fill(0.1)
        dof_props["friction"].fill(0.0)
        dof_props["armature"].fill(0.01)
            
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset_simple)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset_simple)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset_simple)

        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            self.dof_vel_limits[i] = dof_props["velocity"][i].item()
            self.torque_limits[i] = dof_props["effort"][i].item()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset_simple)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        #termination_contact_names = []
        #for name in self.cfg["rewards"]["terminate_contacts_on"]:
        #    termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset_simple, asset_cfg["base_name"])
        if  self.base_indice == -1:
            raise Exception("Origin Index Unknown!")

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset_simple, penalized_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset_simple)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset_simple, asset_cfg["foot_names"][i])
            if indices == -1:
                raise Exception("Foot Name Index Unknown!")
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        self.lower_body_indices = torch.zeros(2, dtype=torch.long, device=self.device)
        for i in range(len(asset_cfg["lower_body_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset_simple, asset_cfg["lower_body_names"][i])
            if indices == -1:
                raise Exception("Lower Body Index Unknown!")
            self.lower_body_indices[i] = indices

        self.head_indices = torch.zeros(2, dtype=torch.long, device=self.device)
        for i in range(len(asset_cfg["head_name"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset_simple, asset_cfg["head_name"][i])
            if indices == -1:
                raise Exception("Head Index Unknown!")
            self.lower_body_indices[i] = indices

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset_simple if np.fmod(i, 2) == 0 else robot_asset_mesh, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _get_robot_asset(self, file):
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(file)
        asset_file = os.path.basename(file)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        return self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for p in props:
            p.friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            p.compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            p.restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.prev_dof_pos = torch.zeros_like(self.dof_pos)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.custom_dof_vel = torch.zeros_like(self.dof_vel)
        self.filtered_custom_dof_vel = torch.zeros_like(self.dof_vel)

        self.current_kick_pose_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_kick_pose_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies].view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.body_states = gymtorch.wrap_tensor(body_state)[:self.num_envs * self.num_bodies].view(self.num_envs, self.num_bodies, 13)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_quat_z = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.body_states[:, self.feet_indices, 7:10]
        self.feet_vel_filtered = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device)

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.actions_raw = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.executed_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_swing_counter = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.last_feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        self.still_envs = torch.empty(0, dtype=torch.long, device=self.device)
        self.slow_envs = torch.empty(0, dtype=torch.long, device=self.device)
        self.max_lin_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.terminate_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.feet_distance_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

        self.zero_obs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ones_obs = torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        # Data for the get up reference poses
        self.num_front_keyframes = len(self.cfg["commands"]["ref_pose_front"]["poses"])
        self.num_back_keyframes = len(self.cfg["commands"]["ref_pose_back"]["poses"])
        if self.num_front_keyframes != self.num_back_keyframes:
            raise Exception("Stand Up Motions must have the same number of keyframes")

        self.get_up_poses = torch.zeros(2, len(self.cfg["commands"]["ref_pose_front"]["poses"]), len(self.cfg["commands"]["ref_pose_front"]["poses"][0]) + len(self.cfg["commands"]["ref_pose_front"]["torso"][0]) + 2, dtype=torch.float, device=self.device) # (#poses, #joints+torso_x+torso_y+past_time+its_time)

        if len(self.cfg["commands"]["ref_pose_front"]["poses"][0]) != self.num_actions:
            raise Exception("Front Motion Does Not Have the Correct Number Of Joints!")
        if len(self.cfg["commands"]["ref_pose_back"]["poses"][0]) != self.num_actions:
            raise Exception("Back Motion Does Not Have the Correct Number Of Joints!")

        self.get_up_poses[0, :, :self.num_actions] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["poses"], dtype=torch.float32, device=self.device)
        self.get_up_poses[0, :, self.num_actions:-2] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["torso"], dtype=torch.float32, device=self.device)
        self.get_up_poses[0, :, -1] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["time"], dtype=torch.float32, device=self.device)
        self.get_up_poses[1, :, :self.num_actions] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["poses"], dtype=torch.float32, device=self.device)
        self.get_up_poses[1, :, self.num_actions:-2] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["torso"], dtype=torch.float32, device=self.device)
        self.get_up_poses[1, :, -1] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["time"], dtype=torch.float32, device=self.device)

        self.get_up_poses[:, :, -2] = torch.cumsum(self.get_up_poses[:, :, -1], dim=-1)

        self.spawn_height_offset = torch.tensor(self.cfg["commands"]["spawn_height_offset"], dtype=torch.float32, device=self.device)

        self.num_keyframes = torch.zeros(2, dtype=torch.int, device=self.device) # ([front|back])
        self.num_keyframes[0] = len(self.cfg["commands"]["ref_pose_front"]["poses"])
        self.num_keyframes[1] = len(self.cfg["commands"]["ref_pose_back"]["poses"])

        self.current_stand_up_info = torch.zeros(self.num_envs, 3, dtype=torch.int, device=self.device) # ([front|back], keyframe_index)
        self.current_stand_up_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Num_Joints, Torso_Roll, Torso_Pitch, Ref_Min_Height
        self.start_joint_poses = torch.zeros(self.num_envs, self.num_actions + len(self.cfg["commands"]["ref_pose_front"]["torso"][0]) + 1, dtype=torch.float32, device=self.device)
        self.current_target_pose = torch.zeros(self.num_envs, self.num_actions + len(self.cfg["commands"]["ref_pose_front"]["torso"][0]) + 1, dtype=torch.float32, device=self.device)

        self.env_torso_rot = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self.execution_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.torso_weights = torch.zeros(2, len(self.cfg["commands"]["ref_pose_front"]["poses"]), dtype=torch.float32, device=self.device)
        self.torso_weights[0, :] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["torso_weight"], dtype=torch.float32, device=self.device)
        self.torso_weights[1, :] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["torso_weight"], dtype=torch.float32, device=self.device)

        self.arm_weight = torch.zeros(2, len(self.cfg["commands"]["ref_pose_front"]["poses"]), dtype=torch.float32, device=self.device)
        self.arm_weight[0, :] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["arm_weight"], dtype=torch.float32, device=self.device)
        self.arm_weight[1, :] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["arm_weight"], dtype=torch.float32, device=self.device)

        self.leg_weight = torch.zeros(2, len(self.cfg["commands"]["ref_pose_front"]["poses"]), dtype=torch.float32, device=self.device)
        self.leg_weight[0, :] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["leg_weight"], dtype=torch.float32, device=self.device)
        self.leg_weight[1, :] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["leg_weight"], dtype=torch.float32, device=self.device)

        self.ankle_yaw_hip_roll_weights = torch.zeros(2, len(self.cfg["commands"]["ref_pose_front"]["poses"]), 4, dtype=torch.float32, device=self.device)
        self.ankle_yaw_hip_roll_weights[0, :] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["ankle_yaw_hip_roll_weights"], dtype=torch.float32, device=self.device)
        self.ankle_yaw_hip_roll_weights[1, :] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["ankle_yaw_hip_roll_weights"], dtype=torch.float32, device=self.device)

        self.torso_height = torch.zeros(2, len(self.cfg["commands"]["ref_pose_front"]["min_height"]), dtype=torch.float32, device=self.device)
        self.torso_height[0, :] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["min_height"], dtype=torch.float32, device=self.device)
        self.torso_height[1, :] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["min_height"], dtype=torch.float32, device=self.device)

        self.feet_slip_jump_penalty = torch.zeros(2, len(self.cfg["commands"]["ref_pose_front"]["feet_slip_jump_penalty"]), dtype=torch.bool, device=self.device)
        self.feet_slip_jump_penalty[0, :] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["feet_slip_jump_penalty"], dtype=torch.bool, device=self.device)
        self.feet_slip_jump_penalty[1, :] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["feet_slip_jump_penalty"], dtype=torch.bool, device=self.device)

        self.extra_balance_penalty = torch.zeros(2, len(self.cfg["commands"]["ref_pose_front"]["extra_balance_penalty"]), dtype=torch.bool, device=self.device)
        self.extra_balance_penalty[0, :] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["extra_balance_penalty"], dtype=torch.bool, device=self.device)
        self.extra_balance_penalty[1, :] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["extra_balance_penalty"], dtype=torch.bool, device=self.device)

        self.arm_vel_penalty = torch.zeros(2, len(self.cfg["commands"]["ref_pose_front"]["arm_vel_penalty"]), dtype=torch.float, device=self.device)
        self.arm_vel_penalty[0, :] = torch.tensor(self.cfg["commands"]["ref_pose_front"]["arm_vel_penalty"], dtype=torch.float, device=self.device)
        self.arm_vel_penalty[1, :] = torch.tensor(self.cfg["commands"]["ref_pose_back"]["arm_vel_penalty"], dtype=torch.float, device=self.device)

        self.fall_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def reset(self):
        """Reset all robots"""
        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(ids)
        self._resample_commands()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_to_key_frame(self, env_ids):
        """Place robot into a compatible pose to start from an arbitary point in the get up, e.g. at 75% of the execution time"""
        self.current_stand_up_info[env_ids, 0] = (torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1) < 0.5).int()
        self.current_stand_up_info[env_ids, 1] = torch.randint(low=0, high=self.num_keyframes[0], size=(len(env_ids), 1), device=self.device).int().squeeze(1)
        # Force 50% envs to start from the start
        self.current_stand_up_info[env_ids, 1] = torch.where(torch_rand_float(0, 1, (len(env_ids), 1), device=self.device).squeeze(1) < 0.20, 0, self.current_stand_up_info[env_ids, 1])
        self.current_stand_up_timer[env_ids] = 0
        self.start_joint_poses[env_ids, :-1] = self.get_up_poses[self.current_stand_up_info[env_ids, 0], self.current_stand_up_info[env_ids, 1], :self.num_actions+len(self.cfg["commands"]["ref_pose_front"]["torso"][0])]
        self.start_joint_poses[env_ids, -1] = self.torso_height[self.current_stand_up_info[env_ids, 0], self.current_stand_up_info[env_ids, 1]]
        self.gait_process[env_ids] = 0

        self.execution_time[env_ids] = (self.get_up_poses[self.current_stand_up_info[env_ids, 0], self.current_stand_up_info[env_ids, 1], -2] - self.get_up_poses[self.current_stand_up_info[env_ids, 0], self.current_stand_up_info[env_ids, 1], -1]).clip(min=0.0)

        self.dof_pos[env_ids] = self.start_joint_poses[env_ids, :self.num_actions]

        self.current_target_pose[:] = self._get_current_ref_pos()

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._reset_to_key_frame(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.executed_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0
        self.terminate_counter[env_ids] = 0
        self.feet_swing_counter[env_ids, :] = 0
        self.feet_distance_counter[env_ids] = 0
        self.fall_detected[env_ids] = False

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.get_up_poses[self.current_stand_up_info[env_ids, 0], self.current_stand_up_info[env_ids, 1], :self.num_actions], self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        self.prev_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.custom_dof_vel[env_ids] = 0.0
        self.filtered_custom_dof_vel[env_ids] = 0.0
        self.last_actions[env_ids] = self.dof_pos[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        # Robot
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, :2] = apply_randomization(self.root_states[env_ids, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        self.root_states[env_ids, 2] += self.terrain.terrain_heights(self.root_states[env_ids, :2])
        x_rot = torch.zeros(len(env_ids), dtype=torch.float, device=self.device)
        y_rot = torch.zeros(len(env_ids), dtype=torch.float, device=self.device)
        x_rot[:] = self.get_up_poses[self.current_stand_up_info[env_ids, 0], self.current_stand_up_info[env_ids, 1], -4]
        y_rot[:] = self.get_up_poses[self.current_stand_up_info[env_ids, 0], self.current_stand_up_info[env_ids, 1], -3]
        x_rot[:] = apply_randomization(x_rot, self.cfg["noise"].get("initial_torso"))
        y_rot[:] = apply_randomization(y_rot, self.cfg["noise"].get("initial_torso"))
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            x_rot,
            y_rot,
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )

        # dynamic spawn height
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.simulate(self.sim)
        self.render()
        min_z = torch.min(self.body_states[env_ids, :, 2], dim=1).values
        # compute per-env spawn height: if the deepest point of the robot is below 0, lift by -min_z + offset, otherwise just offset
        spawn_height = torch.where(min_z < 0, -min_z + self.spawn_height_offset, self.spawn_height_offset)
        self.root_states[env_ids, 2] += spawn_height
  
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return

        # Robots
        out_x_min = self.root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size

        self.root_states[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size

        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def _resample_commands(self):
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

    def step(self, actions):
        # pre physics step
        self.actions_raw[:] = actions
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = torch.clip(self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions, min=self.dof_pos_limits[:,0], max=self.dof_pos_limits[:,1])

        # perform physics step
        self.torques.zero_()
        #dof_targets[:] = self.current_target_pose[:, :-3]
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.filtered_custom_dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = dof_torques - friction
            dof_torques_clipped = torch.clip(dof_torques, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques_clipped))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

            self.custom_dof_vel[:] = (self.dof_pos - self.prev_dof_pos) / 0.002
            self.filtered_custom_dof_vel[:] = self.filtered_custom_dof_vel[:] * 0.78 + self.custom_dof_vel[:] * 0.22
            self.prev_dof_pos[:] = self.dof_pos

        self.torques /= self.cfg["control"]["decimation"]
        self.render()
        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        _, _, yaw = get_euler_xyz(self.root_states[:, 3:7])
        self.base_quat_z[:] = quat_from_euler_xyz(self.zero_obs,self.zero_obs,yaw)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.feet_vel[:] = self.body_states[:, self.feet_indices, 7:10]
        self.feet_vel_filtered[:] = self.feet_vel[:] * 0.3 + self.feet_vel_filtered[:] * 0.7
        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.execution_time[:] = self.execution_time + self.dt * self.gait_frequency

        self.gait_process[:] = (self.execution_time / self.get_up_poses[self.current_stand_up_info[:, 0], -1, -2])
        self._update_keyframe_indizes()
        self.current_target_pose[:] = self._get_current_ref_pos()

        self._calc_torso_rot()
        self._kick_robots()
        self._push_robots()
        self._check_termination()
        self._compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)

        self._teleport_robot()
        self._resample_commands()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.last_feet_contact[:] = self.feet_contact
        
        self.last_kick_pose_distance[:] = self.current_kick_pose_distance
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _calc_torso_rot(self):
        g = self.projected_gravity  # (N, 3)

        self.env_torso_rot[:, 0] = torch.atan2(g[:, 1],torch.sqrt(g[:, 0]**2 + g[:, 2]**2))
        self.env_torso_rot[:, 1] = torch.atan2(g[:, 0],-g[:, 2])

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            dummy_vel = torch.zeros_like(self.root_states[:, 7:10])
            dummy_vel[:] = apply_randomization(dummy_vel, self.cfg["randomization"].get("kick_lin_vel"))
            dummy_angle_vel = torch.zeros_like(self.root_states[:, 10:13])
            dummy_angle_vel[:] = apply_randomization(dummy_angle_vel, self.cfg["randomization"].get("kick_ang_vel"))
            self.root_states[:, 7:10] += dummy_vel
            self.filtered_lin_vel[:] += dummy_vel
            self.root_states[:, 10:13] += dummy_angle_vel
            self.filtered_ang_vel[:] += dummy_angle_vel
            self.last_root_vel[:, :3] += dummy_vel
            self.last_root_vel[:, 3:6] += dummy_angle_vel
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos) < 0.01).reshape(
                self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]
            ),
            dim=2,
        )

    def _update_keyframe_indizes(self):
        self.current_stand_up_timer[:] += self.dt * self.gait_frequency

        next_keyframe_map = torch.where((self.current_stand_up_info[:, 1] < self.num_keyframes[self.current_stand_up_info[:, 0]] - 1) & (self.current_stand_up_timer > self.get_up_poses[self.current_stand_up_info[:, 0], self.current_stand_up_info[:, 1], -1]), True, False)

        self.start_joint_poses[next_keyframe_map, :-1] = self.get_up_poses[self.current_stand_up_info[next_keyframe_map, 0], self.current_stand_up_info[next_keyframe_map, 1], :self.num_actions+len(self.cfg["commands"]["ref_pose_front"]["torso"][0])]
        self.start_joint_poses[next_keyframe_map, -1] = self.torso_height[self.current_stand_up_info[next_keyframe_map, 0], self.current_stand_up_info[next_keyframe_map, 1]]
        self.current_stand_up_timer[next_keyframe_map] -= self.get_up_poses[self.current_stand_up_info[next_keyframe_map, 0], self.current_stand_up_info[next_keyframe_map, 1], -1]
        self.current_stand_up_info[next_keyframe_map, 1] += 1

        # The last keyframe gets clipped
        self.current_stand_up_timer[:] = torch.min(self.get_up_poses[self.current_stand_up_info[:, 0], self.current_stand_up_info[:, 1], -1], self.current_stand_up_timer)

    def _get_current_ref_pos(self):
        ratio = (self.current_stand_up_timer / self.get_up_poses[self.current_stand_up_info[:, 0], self.current_stand_up_info[:, 1], -1]).clip(min=0.0, max=1.0)
        joint_ref = self.start_joint_poses[:, :-1] + (self.get_up_poses[self.current_stand_up_info[:, 0], self.current_stand_up_info[:, 1], :self.num_actions+len(self.cfg["commands"]["ref_pose_front"]["torso"][0])] - self.start_joint_poses[:, :-1]) * ratio.unsqueeze(-1)

        height_ref = self.start_joint_poses[:, -1] + (self.torso_height[self.current_stand_up_info[:, 0], self.current_stand_up_info[:, 1]] - self.start_joint_poses[:, -1]) * ratio

        return torch.cat((joint_ref, height_ref.unsqueeze(-1)), dim=-1)

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf[:] = False
        self.reset_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]

        torso_threshold = torch.where(self.current_stand_up_info[:, 1] >= self.num_keyframes[self.current_stand_up_info[:, 0]], 0.75, 0.75)

        self.fall_detected[:] = (self.fall_detected
                                 |(((torch.abs(self.env_torso_rot[:, 0] - self.current_target_pose[:, -3]) > torso_threshold)
                                     | (torch.abs(self.env_torso_rot[:, 1] - self.current_target_pose[:, -2]) > torso_threshold))
                                    & (self.episode_length_buf > (0.5 / self.dt))))

        self.terminate_counter += torch.where((self.terminate_counter > 0)
                                               | (self.gait_process >= 1.5)
                                               | self.fall_detected, 1, 0)

        self.reset_buf |= self.terminate_counter > self.cfg["rewards"]["terminate_time"] / self.dt
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0)

    def _compute_observations(self):
        """Computes observations"""
        commands_scale = torch.tensor(
            [self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["ang_vel"]],
            device=self.device,
        )
        env_indices = torch.arange(self.num_envs, device=self.device)

        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                self.gait_process.clip(min=0.0, max=1.0).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.custom_dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )

        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
                #self.current_target_pose,
                self.current_target_pose[:, :-1],
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------ reward functions----------------

    def _reward_tracking_ref_pose(self):
        current_pos_view = torch.cat(
                                (
                                    self.dof_pos,
                                    self.env_torso_rot,
                                    (self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)).unsqueeze(-1),
                                ), dim=-1
                            )
        diff_to_target = current_pos_view - self.current_target_pose
        joint_reward_single = torch.exp(-torch.square(diff_to_target[:,:-3]) / self.cfg["rewards"]["tracking_joint_sigma"])

        # Ankle roll and hip rolls have less weight in the later stages
        #                                     -6            -5          -4     -3       -2          -1
        # Joint sequence: [head], [arms], [hip pitch, hip **roll**, hip yaw, knee, ankle pitch, ankle **roll]
        joint_reward_single[:, 2:10] *= self.arm_weight[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1]].unsqueeze(-1) # arms
        joint_reward_single[:, 10:] *= self.leg_weight[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1]].unsqueeze(-1) # legs

        joint_reward_single[:, -5-6] *= self.ankle_yaw_hip_roll_weights[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1], 0] # hip roll
        joint_reward_single[:, -4-6] *= self.ankle_yaw_hip_roll_weights[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1], 1] # hip yaw
        joint_reward_single[:, -2-6] *= self.ankle_yaw_hip_roll_weights[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1], 2] # ankle pitch
        joint_reward_single[:, -1-6] *= self.ankle_yaw_hip_roll_weights[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1], 3] # ankle roll
        joint_reward_single[:, -5] *= self.ankle_yaw_hip_roll_weights[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1], 0] # hip roll
        joint_reward_single[:, -4] *= self.ankle_yaw_hip_roll_weights[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1], 1] # hip yaw
        joint_reward_single[:, -2] *= self.ankle_yaw_hip_roll_weights[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1], 2] # ankle pitch
        joint_reward_single[:, -1] *= self.ankle_yaw_hip_roll_weights[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1], 3] # ankle roll

        # Scale with torso error to prevent policy learning stupid movements
        torso_diff_joint_scaling = torch.exp(-torch.square(diff_to_target[:,-3:-1]) / self.cfg["rewards"]["tracking_joint_max_torso_diff_sigma"]).sum(dim=-1) / 2.0
        joint_reward = joint_reward_single.sum(dim=-1) * torso_diff_joint_scaling

        torso_reward = torch.exp(-torch.square(diff_to_target[:,-3:-1]) / self.cfg["rewards"]["tracking_torso_sigma"]).sum(dim=-1) * self.torso_weights[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1]]
        height_factor = torch.exp(-torch.square((-diff_to_target[:, -1]).clip(min=0.0)) / 0.01)

        return (joint_reward + torso_reward) * (height_factor + 0.25 * height_factor) * (~self.fall_detected).float() # Learn to do nothing when breaking up

    def _reward_leg_torso_deviation_penalty(self):
        # calc diff
        current_pos_view = torch.cat(
                                (
                                    self.dof_pos,
                                    self.env_torso_rot,
                                    (self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)).unsqueeze(-1),
                                ), dim=-1
                            )
        diff_to_target = current_pos_view - self.current_target_pose

        # calc threshold
        first_keyframe_envs = self.current_stand_up_info[:, 1] == 0
        ratio = (self.current_stand_up_timer[first_keyframe_envs] / self.get_up_poses[self.current_stand_up_info[first_keyframe_envs, 0], self.current_stand_up_info[first_keyframe_envs, 1], -1]).clip(min=0.0, max=1.0)
        joint_penalty_threshold = torch.full(
                                    diff_to_target.shape,
                                    self.cfg["rewards"]["penalty_joint_threshold"],
                                    dtype=torch.float32,
                                    device=self.device,
                                  )

        # First keyframe shall have stronger penalty
        joint_penalty_threshold[first_keyframe_envs, :-3] = self.cfg["rewards"]["penalty_joint_first_keyframe_threshold"] * (1.0 - ratio).unsqueeze(-1) + ratio.unsqueeze(-1) * self.cfg["rewards"]["penalty_joint_threshold"]
        # Head shall always have strong penalty
        joint_penalty_threshold[:, 0:2] = self.cfg["rewards"]["penalty_joint_first_keyframe_threshold"]

        # calc joint reward
        joint_reward = (torch.abs(diff_to_target[:,:-3]) - joint_penalty_threshold[:, :-3]).clip(min=0.0).sum(dim=-1) * (1.0 - self.cfg["rewards"]["tracking_penalty_torso_weight"])
        joint_reward[first_keyframe_envs] *= 10 * (1.0 - ratio) + ratio # First keyframe is punished hard
        # calc torso reward
        torso_reward = (torch.abs(diff_to_target[:,-3:-1]) - self.cfg["rewards"]["penalty_torso_threshold"]).clip(min=0.0).sum(dim=-1) * self.cfg["rewards"]["tracking_penalty_torso_weight"]
        # calc height reward
        height_reward = (-diff_to_target[:, -1]).clip(min=0.0) / 0.1

        return joint_reward + torso_reward + height_reward

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1) * (self.gait_process >= 1.0).float()
        reward[self.episode_length_buf < 20] = 0
        return reward

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        contact_forces = (torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) - 100.0).clip(min=0.0)
        contact_forces[self.episode_length_buf < 20] = 0
        return torch.sum(contact_forces, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        ratio = torch.ones_like(self.dof_vel)
        ratio[:, :10] = 2 # Head and Arms shall get punished hard for fast movements
        ratio[:, 2:10] = self.arm_vel_penalty[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1]].unsqueeze(-1)
        return torch.sum(torch.square(self.dof_vel * ratio), dim=-1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        reward = torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)
        reward[self.extra_balance_penalty[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1]]] *= 10.0
        return reward

    def _reward_action_rate(self):
        # Penalize changes in actions
        action_diff = self.last_actions - self.actions
        action_diff[:, :10] *= 1.5 # Head and arms more penalty
        reward = torch.sum(torch.square(action_diff), dim=-1)
        reward[self.current_stand_up_info[:, 1] == 0] *= 2.0 # First keyframe shall get interpolated smoothly
        return reward

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)
        
    def _reward_actions_in_range(self):
        return torch.sum(torch.abs(self.actions_raw[:,:-1] - torch.clip(self.actions_raw[:,:-1], min=-self.cfg["normalization"]["clip_actions"], max=self.cfg["normalization"]["clip_actions"])), dim=-1)
        
    def _reward_action_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0]
        upper = self.dof_pos_limits[:, 1]
        return torch.sum((((self.default_dof_pos + self.actions[:,:self.num_actions]) < lower) | ((self.default_dof_pos + self.actions[:,:self.num_actions]) > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0),
            dim=-1,
        )

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        ratio = torch.ones_like(self.torques)
        ratio[:, 2:10] = 2
        max_clip = torch.ones_like(self.torques)
        max_clip[:, 2:10] = 2
        return torch.sum(torch.square(self.torques * ratio / self.torque_limits).clip(max=max_clip), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float() * self.feet_slip_jump_penalty[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1]].float()
        )

    def _reward_body_parts_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_fall(self):
        return self.fall_detected.float()

    def _reward_jump(self):
        ''' A walk step like movement is allowed, but no jumps! '''
        return (~self.feet_contact[:, 0] & ~self.feet_contact[:, 1]).float() * self.feet_slip_jump_penalty[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1]].float()

    def _reward_feet_distance(self):
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        return torch.clip(self.cfg["rewards"]["feet_distance_ref"] - feet_distance, min=0.0, max=0.1)

    def _reward_waist(self):
        if not self.cfg["algorithm"]["use_waist"]:
            print("Waist is not supported! Remove reward function!")
            raise Exception("Waist is not supported! Remove reward function!")
        return torch.square(self.dof_pos[:, 10]) + torch.square(self.actions_raw[:, 10])

    def _reward_waist_action(self):
        if not self.cfg["algorithm"]["use_waist"]:
            print("Waist is not supported! Remove reward function!")
            raise Exception("Waist is not supported! Remove reward function!")
        return torch.abs(self.last_actions[:, 10] - self.actions[:, 10])

    def _reward_body_in_soles(self):
        sole_point = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        lower_body = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        head_point = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        for i in self.feet_indices:
            sole_point += self.body_states[:, i, 0:3] + quat_rotate(self.body_states[:, i, 3:7], to_torch(self.cfg["asset"]["feet_sole_pos"], device=self.device).unsqueeze(0).expand(self.num_envs, -1))
        sole_point /= len(self.feet_indices)

        for i in self.lower_body_indices:
            lower_body += self.body_states[:, i, 0:3]
        lower_body /= len(self.lower_body_indices)

        for i in self.head_indices:
            head_point += self.body_states[:, i, 0:3]
        head_point /= len(self.head_indices)

        reward = torch.square(torch.norm(sole_point[:, :2] - ((lower_body + head_point) / 2.0)[:, :2], dim=-1)) * self.extra_balance_penalty[self.current_stand_up_info[:, 0],self.current_stand_up_info[:, 1]].float()

        reward[self.episode_length_buf < 20] = 0
        return reward

