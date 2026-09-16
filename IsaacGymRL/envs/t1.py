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


class T1(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]

        asset_cfg = self.cfg["asset"]
        robot_asset = self._get_robot_asset(asset_cfg["file"])
        robot_asset_box_feet = self._get_robot_asset(asset_cfg["file2"])
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        dof_props["damping"].fill(0)
        dof_props["friction"].fill(0)
        dof_props["armature"].fill(0)

        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_clipping = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            self.dof_vel_limits[i] = dof_props["velocity"][i].item()
            self.torque_limits[i] = dof_props["effort"][i].item()
            self.torque_clipping[i] = self.cfg["control"]["torque_clipping"][i]

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)

        self.default_dof_stiffness = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_damping = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            found_armature = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.default_dof_stiffness[i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.default_dof_damping[i] = self.cfg["control"]["damping"][name]
                    found = True
                    break
            for name in self.cfg["control"]["armature"].keys():
                if name in self.dof_names[i]:
                    dof_props["armature"][i] = self.cfg["control"]["armature"][name]
                    found_armature = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
            if not found_armature:
                raise ValueError(f"Armature of joint {self.dof_names[i]} were not defined")


        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"]["dof_stiffness"])
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"]["dof_damping"])
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"]["dof_friction"])

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        penalized_sole_names = []
        for name in self.cfg["rewards"]["penalize_sole_contacts"]:
            penalized_sole_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])
        if  self.base_indice == -1:
            raise Exception("Origin Index Unknown!")

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.penalized_sole_indices = torch.zeros(len(penalized_sole_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_sole_names)):
            self.penalized_sole_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_sole_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            if indices == -1:
                raise Exception("Foot Name Index Unknown!")
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        self.origin_indices = torch.zeros(2, dtype=torch.long, device=self.device)
        for i in range(len(asset_cfg["origin_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["origin_names"][i])
            if indices == -1:
                raise Exception("Origin Index Unknown!")
            self.origin_indices[i] = indices

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

        kicker_count = 0
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset if np.fmod(i, 2) == 0 else robot_asset_box_feet, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
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

        if not self.cfg["basic"]["headless"]:
            self.terrain.draw_terrain_friction(self.envs[0], self.gym, self.viewer)

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
                    props[j].com.x, self.cfg["randomization"]["base_com_x"], return_noise=True
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
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
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
        self.num_base_obs = self.cfg["env"]["num_base_observations"]
        self.num_last_obs = self.cfg["env"]["num_last_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.num_history = self.cfg["env"]["history_size"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.num_obs = self.num_base_obs * self.num_history + self.num_last_obs

        self.obs_buf = torch.zeros(self.num_envs, self.num_history, self.num_base_obs, dtype=torch.float, device=self.device)
        self.obs = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}
        
        self.waist_shift = 1 if self.cfg["algorithm"]["use_waist"] else 0

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
        self.root_states_all = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.root_states_all[:self.num_envs]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.prev_dof_pos = torch.zeros_like(self.dof_pos)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.custom_dof_vel = torch.zeros_like(self.dof_vel)
        self.filtered_custom_dof_vel = torch.zeros_like(self.dof_vel)
        self.dof_pos_offset = torch.zeros_like(self.dof_pos)

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
        self.kick_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.actions_raw = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_raw_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.actions_gait = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
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
        self.last_feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.last_feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        self.still_envs = torch.empty(0, dtype=torch.long, device=self.device)

        self.all_envs = torch.arange(self.num_envs, device=self.device)
        self.not_still_envs = self.all_envs[~torch.isin(self.all_envs, self.still_envs)]

        self.slow_envs = torch.empty(0, dtype=torch.long, device=self.device)
        self.max_lin_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.terminate_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.gait_factor_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.tracking_walk_direction = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.penalize_ground_pressure = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.filtered_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

        # Ball tensors
        self.ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.direction = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.range = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.zero_obs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ones_obs = torch.ones(self.num_envs, dtype=torch.float, device=self.device)

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
        return self.obs, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.obs_buf[env_ids] = 0
        self.obs[env_ids] = 0

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0
        self.terminate_counter[env_ids] = 0
        self.feet_contact[env_ids] = False
        self.last_feet_contact[env_ids] = False
        self.penalize_ground_pressure[env_ids] = True
        self.filtered_height[env_ids] = 0

        self.dof_stiffness[env_ids, :] = self.default_dof_stiffness
        self.dof_damping[env_ids, :] = self.default_dof_damping

        self.dof_stiffness[env_ids] = apply_randomization(self.dof_stiffness[env_ids], self.cfg["randomization"]["dof_stiffness"])
        self.dof_damping[env_ids] = apply_randomization(self.dof_damping[env_ids], self.cfg["randomization"]["dof_damping"])

        # Ankles
        cols = torch.tensor([-1, -2, -7, -8], device=self.device)
        self.dof_stiffness.index_put_(
            (env_ids.unsqueeze(1), cols),
            self.default_dof_stiffness[cols]
        )
        self.dof_stiffness.index_put_(
            (env_ids.unsqueeze(1), cols),
            apply_randomization(self.dof_stiffness[env_ids][:, [-1, -2, -7, -8]], self.cfg["randomization"]["dof_ankle_stiffness"])
        )
        self.dof_damping.index_put_(
            (env_ids.unsqueeze(1), cols),
            self.default_dof_damping[cols]
        )
        self.dof_damping.index_put_(
            (env_ids.unsqueeze(1), cols),
            apply_randomization(self.dof_damping[env_ids][:, [-1, -2, -7, -8]], self.cfg["randomization"]["dof_ankle_damping"])
        )

        # Hip roll
        cols = torch.tensor([-5, -11], device=self.device)
        self.dof_stiffness.index_put_(
            (env_ids.unsqueeze(1), cols),
            self.default_dof_stiffness[cols]
        )
        self.dof_stiffness.index_put_(
            (env_ids.unsqueeze(1), cols),
            apply_randomization(self.dof_stiffness[env_ids][:, [-5, -11]], self.cfg["randomization"]["dof_hiproll_stiffness"])
        )
        self.dof_damping.index_put_(
            (env_ids.unsqueeze(1), cols),
            self.default_dof_damping[cols]
        )
        self.dof_damping.index_put_(
            (env_ids.unsqueeze(1), cols),
            apply_randomization(self.dof_damping[env_ids][:, [-5, -11]], self.cfg["randomization"]["dof_hiproll_damping"])
        )

        # Apply offsets to better learn handling not perfect calibrated robots
        pitch_joints = [0, 3, 4, 6, 9, 10]
        other_joints = [1, 2, 5, 7, 8, 11]
        if self.cfg["algorithm"]["use_waist"]:
            pitch_joints = [j + 1 for j in pitch_joints]
            other_joints = [j + 1 for j in other_joints]
            other_joints.append(0)

        pitch_cols = torch.tensor(pitch_joints, device=self.device)
        other_cols = torch.tensor(other_joints, device=self.device)
        self.dof_pos_offset[env_ids] = 0

        self.dof_pos_offset.index_put_(
            (env_ids.unsqueeze(1), pitch_cols),
            apply_randomization(self.dof_pos_offset[env_ids][:, pitch_cols], self.cfg["randomization"]["dof_pos_offset_pitch"])
        )
        self.dof_pos_offset.index_put_(
            (env_ids.unsqueeze(1), other_cols),
            apply_randomization(self.dof_pos_offset[env_ids][:, other_cols], self.cfg["randomization"]["dof_pos_offset_other"])
        )

        self.delay_steps[env_ids] = torch.randint(4, 7, (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.prev_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.0
        self.custom_dof_vel[env_ids] = 0.0
        self.filtered_custom_dof_vel[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_actions[env_ids] = torch.cat((self.dof_pos[env_ids], torch.zeros(len(env_ids), 1, dtype=torch.float, device=self.device)), dim=-1)
        self.last_raw_actions[env_ids] = torch.cat((self.dof_pos[env_ids], torch.zeros(len(env_ids), 1, dtype=torch.float, device=self.device)), dim=-1)
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
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )

        self.root_states[env_ids, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )
        self.root_states[env_ids, 9] = apply_randomization(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_z"),
        )

        self.root_states[env_ids, 10:13] = apply_randomization(
            torch.zeros(len(env_ids), 3, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_ang_vel_xyz"),
        )

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))

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
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
            self._refresh_feet_state()

    def _resample_commands(self):
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.cfg["commands"]["lin_vel_x"][0], self.cfg["commands"]["lin_vel_x"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.cfg["commands"]["lin_vel_y"][0], self.cfg["commands"]["lin_vel_y"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

        # Sample still envs
        if self.cfg["commands"]["still_proportion"] * self.num_envs - len(self.still_envs) > 2:
            new_still_envs = env_ids[torch.randperm(len(env_ids))[: int(math.ceil((self.cfg["commands"]["still_proportion"] + self.cfg["commands"]["proportion_correction"]) * len(env_ids)))]]
        else:
            new_still_envs = env_ids[torch.randperm(len(env_ids))[: int(self.cfg["commands"]["still_proportion"] * len(env_ids))]]
        mask_still = ~torch.isin(self.still_envs, env_ids)
        self.still_envs = torch.cat([self.still_envs[mask_still], new_still_envs])
        self.commands[new_still_envs, :] = 0.0
        self.gait_frequency[new_still_envs] = 0.0
        
        # Sample walking in place envs
        not_standing_env_ids = env_ids[~torch.isin(env_ids, self.still_envs)]
        if self.cfg["commands"]["slow_proportion"] * self.num_envs - len(self.slow_envs) > 2:
            new_slow_envs = not_standing_env_ids[torch.randperm(len(not_standing_env_ids))[: int(math.ceil((self.cfg["commands"]["slow_proportion"] + self.cfg["commands"]["proportion_correction"]) * len(not_standing_env_ids)))]]
        else:
            new_slow_envs = not_standing_env_ids[torch.randperm(len(not_standing_env_ids))[: int(self.cfg["commands"]["slow_proportion"] * len(not_standing_env_ids))]]
        mask_slow = ~torch.isin(self.slow_envs, not_standing_env_ids)
        self.slow_envs = torch.cat([self.slow_envs[mask_slow], new_slow_envs])
        
        target_norms = 0.001 + (self.cfg["commands"]["slow_walk_range"] - 0.001) * torch.rand(len(new_slow_envs), 1, device=self.commands.device)
        self.commands[new_slow_envs] = self.commands[new_slow_envs] * (
            target_norms / (self.commands[new_slow_envs].norm(dim=-1, keepdim=True) + 1e-8)
        )

        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

        self.not_still_envs = self.all_envs[~torch.isin(self.all_envs, self.still_envs)]

    def step(self, actions):
        # pre physics step
        self.actions_raw[:] = actions
        self.actions[:, :-1] = torch.clip(actions[:, :-1], -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        self.actions[:,-1] = torch.clip(actions[:,-1], self.cfg["normalization"]["action_frequence_limit"][0], self.cfg["normalization"]["action_frequence_limit"][1])
        self.actions_gait[:] = self.actions[:,-1]
        self.actions_gait += self.cfg["rewards"]["target_frequence"]
        self.actions_gait[self.still_envs] = 0
        dof_targets = torch.clip(self.default_dof_pos + self.dof_pos_offset + self.cfg["control"]["action_scale"] * self.actions[:,:self.num_actions-1], min=self.dof_pos_limits[:,0], max=self.dof_pos_limits[:,1])

        # perform physics step
        self.torques.zero_()

        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.filtered_custom_dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = dof_torques - friction
            dof_torques_clipped = torch.clip(dof_torques, min=-self.torque_clipping, max=self.torque_clipping)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques_clipped))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

            self.custom_dof_vel[:] = (self.dof_pos - self.prev_dof_pos) / self.cfg["sim"]["dt"]
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
        ratio = torch.clip(torch.norm(self.commands[:], dim=1) / self.cfg["normalization"]["filter_weight_speed_range"], min=0.0, max=1.0)
        filter_weight = (ratio * self.cfg["normalization"]["filter_weight"] + (1.0 - ratio) * self.cfg["normalization"]["filter_weight_low_speed"]).unsqueeze(-1)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * filter_weight + self.filtered_lin_vel[:] * (
            1.0 - filter_weight
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * filter_weight + self.filtered_ang_vel[:] * (
            1.0 - filter_weight
        )
        self.feet_vel[:] = self.body_states[:, self.feet_indices, 7:10]
        self.feet_vel_filtered[:] = self.feet_vel[:] * 0.3 + self.feet_vel_filtered[:] * 0.7
        self._refresh_feet_state()

        self.filtered_height[:] = self.filtered_height * 0.8 + self.terrain.terrain_heights(self.base_pos) * 0.2

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.actions_gait, 1.0)

        tracking_ratio = torch.max(torch.abs(self.commands[:, 0]), torch.abs(self.commands[:, 1])).clip(max=1.0)
        self.tracking_walk_direction[:] = tracking_ratio * torch.exp(-torch.square((torch.atan2(self.base_lin_vel[:, 1], self.base_lin_vel[:, 0]) - torch.atan2(self.commands[:, 1], self.commands[:, 0]) + torch.pi) % (2 * torch.pi) - torch.pi) / 0.5) + (1.0 - tracking_ratio)

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
        self.last_raw_actions[:] = self.actions_raw
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.last_feet_yaw[:] = self.feet_yaw

        self.penalize_ground_pressure[:, 0] = (self.last_feet_contact[:, 0] & ~self.feet_contact[:, 0]) | (self.penalize_ground_pressure[:, 0] & (self.contact_forces[:, self.penalized_sole_indices[0], 2] < 1.0))
        self.penalize_ground_pressure[:, 1] = (self.last_feet_contact[:, 1] & ~self.feet_contact[:, 1]) | (self.penalize_ground_pressure[:, 1] & (self.contact_forces[:, self.penalized_sole_indices[1], 2] < 1.0))

        self.last_feet_contact[:] = self.feet_contact
        self.last_kick_pose_distance[:] = self.current_kick_pose_distance

        return self.obs, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            dummy_vel = torch.zeros_like(self.root_states[:, 7:10])
            dummy_vel[:, :2] = apply_randomization(dummy_vel[:, :2], self.cfg["randomization"].get("kick_lin_vel") if self.kick_counter % 3 < 2 else self.cfg["randomization"].get("kick_lin_vel_strong"))
            dummy_vel[self.still_envs, :2] *= 0.5
            dummy_vel[:, 2] = apply_randomization(dummy_vel[:, 2], self.cfg["randomization"].get("kick_lin_vel_z"))
            dummy_angle_vel = torch.zeros_like(self.root_states[:, 10:13])
            dummy_angle_vel[:] = apply_randomization(dummy_angle_vel, self.cfg["randomization"].get("kick_ang_vel"))
            self.root_states[:, 7:10] += dummy_vel
            self.filtered_lin_vel[:] += dummy_vel
            self.root_states[:, 10:13] += dummy_angle_vel
            self.filtered_ang_vel[:] += dummy_angle_vel
            self.last_root_vel[:, :3] += dummy_vel
            self.last_root_vel[:, 3:6] += dummy_angle_vel
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
            self.kick_counter += 1

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

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.terminate_counter += torch.where((self.terminate_counter > 0) | (self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]), 1, 0)
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
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.0)

    def _compute_observations(self):
        """Computes observations"""
        commands_scale = torch.tensor(
            [self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["lin_vel"], self.cfg["normalization"]["ang_vel"]],
            device=self.device,
        )

        self.obs_buf[:] = self.obs_buf.roll(shifts=-1, dims=1)

        noise_gravity = torch.clone(self.projected_gravity)
        noise_ang_vel = torch.clone(self.base_ang_vel)
        noise_dof_vel = torch.clone(self.custom_dof_vel)
        noise_dof_pos = torch.clone(self.dof_pos - self.default_dof_pos - self.dof_pos_offset)

        noise_gravity[self.not_still_envs] = apply_randomization(noise_gravity[self.not_still_envs], self.cfg["noise"]["gravity"])
        noise_gravity[self.still_envs] = apply_randomization(noise_gravity[self.still_envs], self.cfg["noise"]["gravity_stand"])
        noise_ang_vel[self.not_still_envs] = apply_randomization(noise_ang_vel[self.not_still_envs], self.cfg["noise"]["ang_vel"])
        noise_ang_vel[self.still_envs] = apply_randomization(noise_ang_vel[self.still_envs], self.cfg["noise"]["ang_vel_stand"])
        noise_dof_vel[self.not_still_envs] = apply_randomization(noise_dof_vel[self.not_still_envs], self.cfg["noise"]["dof_vel"])
        noise_dof_vel[self.still_envs] = apply_randomization(noise_dof_vel[self.still_envs], self.cfg["noise"]["dof_vel_stand"])
        noise_dof_pos[self.not_still_envs] = apply_randomization(noise_dof_pos[self.not_still_envs], self.cfg["noise"]["dof_pos"])
        noise_dof_pos[self.still_envs] = apply_randomization(noise_dof_pos[self.still_envs], self.cfg["noise"]["dof_pos_stand"])

        self.obs_buf[:, -1] = torch.cat(
            (
                noise_gravity * self.cfg["normalization"]["gravity"],
                noise_ang_vel * self.cfg["normalization"]["ang_vel"],
                noise_dof_pos * self.cfg["normalization"]["dof_pos"],
                self.actions[:, :-1],
                self.zero_obs.unsqueeze(-1), # Ball x
                self.zero_obs.unsqueeze(-1), # Ball x
            ),
            dim=-1,
        )

        # Add more and more noise for older data while standing. Improves standing, otherwise K1 like to oscillate slowly
        self.obs_buf[self.still_envs, :-1, :3] = apply_randomization(self.obs_buf[self.still_envs, :-1, :3], self.cfg["noise"]["gravity_stand"])
        self.obs_buf[self.still_envs, :-1, 3:6] = apply_randomization(self.obs_buf[self.still_envs, :-1, 3:6], self.cfg["noise"]["ang_vel_stand"])
        self.obs_buf[self.still_envs, :-1, 6:6+self.num_actions-1] = apply_randomization(self.obs_buf[self.still_envs, :-1, 6:6+self.num_actions-1], self.cfg["noise"]["dof_pos_stand"])
        self.obs_buf[self.still_envs, :-1, 6+self.num_actions-1:6+2*(self.num_actions-1)] = apply_randomization(self.obs_buf[self.still_envs, :-1, 6+self.num_actions-1:6+2*(self.num_actions-1)], self.cfg["noise"]["dof_pos"])

        self.obs[:] = torch.cat(
            (
                self.obs_buf.flatten(1),
                self.commands * commands_scale,
                (torch.cos(2 * torch.pi * self.gait_process) * (self.actions_gait > 1.0e-8).float()).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process) * (self.actions_gait > 1.0e-8).float()).unsqueeze(-1),
                noise_dof_vel * self.cfg["normalization"]["dof_vel"],
                self.actions[:, -1].unsqueeze(-1),
                self.zero_obs.unsqueeze(-1), # Ball flags
                self.zero_obs.unsqueeze(-1),
                self.zero_obs.unsqueeze(-1),
                self.direction,
                self.range.unsqueeze(-1) * self.cfg["normalization"]["ball_kick_range"],
            ),
            dim=-1,
        )

        self.privileged_obs_buf[:] = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
                self.zero_obs.unsqueeze(-1),
                self.zero_obs.unsqueeze(-1),
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------ reward functions----------------
    def _reward_terminate(self):
        return (self.terminate_counter > 0).float()

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        ts_s = self.cfg["rewards"]["tracking_sigma_speed"]
        ts = self.cfg["rewards"]["tracking_sigma"]
        tr_s = self.cfg["rewards"]["tracking_reward_x_speed"]
        reward_range = self.cfg["rewards"]["tracking_reward_x_factor"]
        speed_norm = torch.norm(self.commands[:], dim=1)
        speed_axis = torch.abs(self.commands[:, 0])
        sigma = torch.where(speed_norm < ts_s[0],
                            ts[0] + speed_norm / ts_s[0] * (ts[1] - ts[0]),                                                         # increase accuracy for small steps
                            ts[1] + torch.clip((speed_axis - ts_s[0]) / (ts_s[1] - ts_s[0]), min=0.0, max=1.0) * (ts[2] - ts[1]))   # decrease accuracy for large steps
        reward_scaling = torch.where(speed_norm < tr_s[0],
                                     reward_range[0] + (reward_range[1] - reward_range[0]) * speed_norm / tr_s[0],
                                     reward_range[1] + (reward_range[2] - reward_range[1]) * torch.square((speed_norm - tr_s[0]) / (tr_s[1] - tr_s[0])).clip(min=0.0, max=1.0))
        sigma[self.still_envs] = self.cfg["rewards"]["tracking_sigma_stand"]                              # Prevent overbalancing during standing
        shift_factor = 0.1 + 0.2 * torch.abs(self.commands[:, 0]).clip(min=0.0, max=1.0)
        direction_shift_factor = 0.1 + 0.2 * torch.abs(self.commands[:, 0]).clip(min=0.0, max=1.0)
        direction_factor = (self.tracking_walk_direction - direction_shift_factor) / (1.0 - direction_shift_factor)
        reward = (torch.exp(-torch.square(self.commands[:, 0] - self.filtered_lin_vel[:, 0]) / sigma) - shift_factor) / (1.0 - shift_factor)
        reward[self.not_still_envs] *= (torch.where(reward < 0.0, reward_scaling.clip(min=1.0) + (1.0 - direction_factor), reward_scaling * direction_factor))[self.not_still_envs]
        reward[self.still_envs] *= reward_scaling[self.still_envs]
        reward[self.still_envs] *= self.cfg["rewards"]["tracking_stand_factor"][0]
        return reward

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        ts_s = self.cfg["rewards"]["tracking_sigma_speed"]
        ts = self.cfg["rewards"]["tracking_sigma"]
        tr_s = self.cfg["rewards"]["tracking_reward_y_speed"]
        reward_range = self.cfg["rewards"]["tracking_reward_y_factor"]
        speed_norm = torch.norm(self.commands[:], dim=1)
        speed_axis = torch.abs(self.commands[:, 1])
        sigma = torch.where(speed_norm < ts_s[0],
                            ts[0] + speed_norm / ts_s[0] * (ts[1] - ts[0]),                                                         # increase accuracy for small steps
                            ts[1] + torch.clip((speed_axis - ts_s[0]) / (ts_s[1] - ts_s[0]), min=0.0, max=1.0) * (ts[2] - ts[1]))   # decrease accuracy for large steps
        reward_scaling = torch.where(speed_norm < tr_s[0],
                                     reward_range[0] + (reward_range[1] - reward_range[0]) * speed_norm / tr_s[0],
                                     reward_range[1] + (reward_range[2] - reward_range[1]) * torch.square((speed_norm - tr_s[0]) / (tr_s[1] - tr_s[0])).clip(min=0.0, max=1.0))
        sigma[self.still_envs] = self.cfg["rewards"]["tracking_sigma_stand"]                                                        # Prevent overbalancing during standing
        shift_factor = 0.1 + (torch.abs(self.commands[:, 1]) - 1.0).clip(min=0.0, max=1.0) * 0.2
        direction_shift_factor = 0.1 + (torch.abs(self.commands[:, 1]) - 1.0).clip(min=0.0, max=1.0) * 0.2
        direction_factor = (self.tracking_walk_direction - direction_shift_factor) / (1.0 - direction_shift_factor)
        reward = (torch.exp(-torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) / sigma) - shift_factor) / (1.0 - shift_factor)
        reward[self.not_still_envs] *= (torch.where(reward < 0.0, reward_scaling.clip(min=1.0) + (1.0 - direction_factor), reward_scaling * direction_factor))[self.not_still_envs]
        reward[self.still_envs] *= reward_scaling[self.still_envs]
        reward[self.still_envs] *= self.cfg["rewards"]["tracking_stand_factor"][1]
        return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ts_s = self.cfg["rewards"]["tracking_sigma_speed"]
        ts = self.cfg["rewards"]["tracking_yaw_sigma"]
        tr_s = self.cfg["rewards"]["tracking_reward_yaw_speed"]
        reward_range = self.cfg["rewards"]["tracking_reward_yaw_factor"]
        speed_norm = torch.norm(self.commands[:], dim=1)
        speed_axis = torch.abs(self.commands[:, 2])
        sigma = torch.where(speed_norm < ts_s[0],
                            ts[0] + speed_norm / ts_s[0] * (ts[1] - ts[0]),                                                         # increase accuracy for small steps
                            ts[1] + torch.clip((speed_axis - ts_s[0]) / (ts_s[1] - ts_s[0]), min=0.0, max=1.0) * (ts[2] - ts[1]))   # decrease accuracy for large steps
        reward_scaling = torch.where(speed_norm < tr_s[0],
                                     reward_range[0] + (reward_range[1] - reward_range[0]) * speed_norm / tr_s[0],
                                     reward_range[1] + (reward_range[2] - reward_range[1]) * torch.square((speed_norm - tr_s[0]) / (tr_s[1] - tr_s[0])).clip(min=0.0, max=1.0))
        sigma[self.still_envs] = self.cfg["rewards"]["tracking_sigma_stand"]                                                        # Prevent overbalancing during standing
        shift_factor = 0.1 + (torch.abs(self.commands[:, 2]) - 1.0).clip(min=0.0, max=1.0) * 0.2
        direction_shift_factor = 0.1 + (torch.abs(self.commands[:, 2]) - 1.0).clip(min=0.0, max=1.0) * 0.2
        direction_factor = (self.tracking_walk_direction - direction_shift_factor) / (1.0 - direction_shift_factor)
        reward = (torch.exp(-torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2]) / sigma) - shift_factor) / (1.0 - shift_factor)
        reward[self.not_still_envs] *= (torch.where(reward < 0.0, reward_scaling.clip(min=1.0) + (1.0 - direction_factor), reward_scaling * direction_factor))[self.not_still_envs]
        reward[self.still_envs] *= reward_scaling[self.still_envs]
        reward[self.still_envs] *= self.cfg["rewards"]["tracking_stand_factor"][2]
        return reward

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.filtered_height
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

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
        return torch.sum((((self.default_dof_pos + self.actions_raw[:,:self.num_actions-1]) < lower) | ((self.default_dof_pos + self.actions_raw[:,:self.num_actions-1]) > upper)).float(), dim=-1)

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
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

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
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_slip_z_rot(self):
        # Penalize feet z-rotation when contact
        return (
            torch.sum(
                torch.square(((self.last_feet_yaw - self.feet_yaw) / self.dt).unsqueeze(-1)).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square(self.feet_vel[:, :, 2]), dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_yaw_diff(self):
        return torch.square((self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi)
        return torch.square((get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_distance(self):
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )

        ratio = (torch.norm(self.commands[:,:], dim=-1) / 0.1).clip(min=0.0, max=1.0)
        reward = ratio * torch.clip(self.cfg["rewards"]["feet_distance_ref"] - feet_distance, min=0.0, max=0.1) + (1.0 - ratio) * torch.clip(feet_distance - self.cfg["rewards"] ["feet_distance_ref"] * 0.8, min=0.0, max=0.1) * 3.0
        return reward

    def _reward_feet_swing(self):
        if self.cfg["rewards"]["use_swing_reward_scaling"]:
            speed_ratio = 1.0 - 0.75 * ((torch.norm(self.commands, dim=-1) - 0.2) / 0.8).clip(min=0.0, max=1.0)
        else:
            speed_ratio = 1.0
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.actions_gait > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.actions_gait > 1.0e-8)
        return ((left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()) * speed_ratio

    def _reward_hip_roll(self):
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        reward[self.still_envs] = torch.abs(self.dof_pos[self.still_envs, self.waist_shift+1]) + torch.abs(self.dof_pos[self.still_envs, self.waist_shift+7])
        return reward

    def _reward_ground_pressure(self):
        ground_pressure = (~self.last_feet_contact[:, 0] & self.feet_contact[:, 0]).float() * torch.clamp(self.feet_vel_filtered[:, 0, 2], max=0.0) + (~self.last_feet_contact[:, 1] & self.feet_contact[:, 1]).float() * torch.clamp(self.feet_vel_filtered[:, 1, 2], max=0.0)
        return -ground_pressure
        #return torch.sum(self.contact_forces[:, self.penalized_sole_indices, 2].clip(min=0.0) * self.penalize_ground_pressure.float(), dim=-1)

    def _reward_gait_phase_factor(self):
        return torch.abs(self.actions_raw[:,-1] - torch.clamp(self.actions_raw[:,-1], min=self.cfg["normalization"]["action_frequence_limit"][0], max=self.cfg["normalization"]["action_frequence_limit"][1]))

    def _reward_feet_height(self):
        # Force feet height for swing sole only for low walk speeds
        speed_ratio = 1.0 - (torch.norm(self.commands, dim=-1) / 0.5).clip(max=1.0)
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.actions_gait > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.actions_gait > 1.0e-8)
        left_error = (~left_swing).float() + left_swing.float() * torch.exp(-torch.square(torch.clip(self.feet_pos[:, 0, 2] - self.feet_pos[:, 1, 2] - self.cfg["rewards"]["feet_height_ref"], max=0.0)) / 0.0005)
        right_error = (~right_swing).float() + right_swing.float() * torch.exp(-torch.square(torch.clip(self.feet_pos[:, 1, 2] - self.feet_pos[:, 0, 2] - self.cfg["rewards"]["feet_height_ref"], max=0.0)) / 0.0005)
        return (2 - left_error - right_error) * speed_ratio
        
    def _reward_waist(self):
        if not self.cfg["algorithm"]["use_waist"]:
            print("Waist is not supported! Remove reward function!")
            raise Exception("Waist is not supported! Remove reward function!")
        return torch.square(self.dof_pos[:, 0]) + torch.square(self.actions_raw[:, 0])

    def _reward_waist_action(self):
        if not self.cfg["algorithm"]["use_waist"]:
            print("Waist is not supported! Remove reward function!")
            raise Exception("Waist is not supported! Remove reward function!")
        return torch.abs(self.last_raw_actions[:,0] - self.actions_raw[:,0])

