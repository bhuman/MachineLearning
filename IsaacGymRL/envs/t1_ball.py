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


class T1_Ball(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]

        self.num_kickers = self.num_envs

        asset_cfg = self.cfg["asset"]
        robot_assets = []
        for file in asset_cfg["files"]:
            robot_assets.append(self._get_robot_asset(file))

        dof_props = self.gym.get_asset_dof_properties(robot_assets[0])
        dof_props["damping"].fill(0.0)
        dof_props["friction"].fill(0.0)
        dof_props["armature"].fill(0.0)

        self.num_dofs = self.gym.get_asset_dof_count(robot_assets[0])
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_assets[0])
        self.dof_names = self.gym.get_asset_dof_names(robot_assets[0])

        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_clipping = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_high_limit = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            self.dof_vel_limits[i] = dof_props["velocity"][i].item()
            self.torque_limits[i] = dof_props["effort"][i].item()
            self.torque_clipping[i] = self.cfg["control"]["torque_clipping"][i]
            self.torque_high_limit[i] = self.cfg["control"]["torque_prefered"][i]

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

        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_assets[0])
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_assets[0], asset_cfg["base_name"])
        if  self.base_indice == -1:
            raise Exception("Origin Index Unknown!")

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_assets[0], penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_assets[0], termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_assets[0])
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_assets[0], asset_cfg["foot_names"][i])
            if indices == -1:
                raise Exception("Foot Name Index Unknown!")
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        for other_index in range(1, len(robot_assets)):
            self.feet_indices2 = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
            self.foot_shape_indices2 = []
            for i in range(len(asset_cfg["foot_names"])):
                indices = self.gym.find_asset_rigid_body_index(robot_assets[other_index], asset_cfg["foot_names"][i])
                if indices == -1:
                    raise Exception("Foot Name Index Unknown!")
                self.feet_indices2[i] = indices
                self.foot_shape_indices2 += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

            if (self.feet_indices != self.feet_indices2).any() or self.foot_shape_indices != self.foot_shape_indices2:
                raise Exception(f'Assets {other_index} have different foot shape indices!')

        self.origin_indices = torch.zeros(2, dtype=torch.long, device=self.device)
        for i in range(len(asset_cfg["origin_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_assets[0], asset_cfg["origin_names"][i])
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
        self.ball_handles = []            # len == num_kickers (actor handle for each ball)
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.env_is_kicker = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.env_to_ball = -torch.ones(self.num_envs, dtype=torch.long, device=self.device)  # mapping env->ball idx or -1

        self.heavy_ball = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.heavy_ball[:] = False

        kicker_count = 0
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_assets[i % len(robot_assets)], start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
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

        for i in range(self.num_envs):
            # position ball relative to robot
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)
            start_pose.p.z = 1
            ball_asset = self._create_ball_asset(radius=apply_randomization(self.cfg["asset"]["ball"]["radius"], self.cfg["randomization"].get("ball_radius")))
            ball_handle = self.gym.create_actor(env_handle, ball_asset, start_pose, "ball", i, True, 0)
            try:
                ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
                for b in range(len(ball_body_props)):
                    ball_body_props[b].mass = apply_randomization(self.cfg["asset"]["ball"]["mass"], self.cfg["randomization"].get("ball_mass"))
                    if np.random.random() < self.cfg["rewards"]["ball_parameters"]["heavy_ball"]:
                        ball_body_props[b].mass *= 100
                        self.heavy_ball[i] = True

                self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, recomputeInertia=True)
            except Exception:
                print("meh")
                pass

            # shape props: restitution/friction
            try:
                ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, ball_handle)
                for s in range(len(ball_shape_props)):
                    ball_shape_props[s].restitution = 0.1
                    ball_shape_props[s].friction = 1
                    ball_shape_props[s].rolling_friction = 0.3
                    ball_shape_props[s].torsion_friction = 0.1
                    ball_shape_props[s].thickness = 0.01
                    ball_shape_props[s].contact_offset = 0.02
                    ball_shape_props[s].rest_offset = 0.0
                self.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shape_props)
            except Exception as e:
                print(e)
                print("meh")
                pass

            self.env_to_ball[i] = kicker_count
            self.ball_handles.append(ball_handle)
            kicker_count += 1

            self.envs.append(env_handle)

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

    def _create_ball_asset(self, radius):
        """
        Erzeugt einmalig ein Ball-Asset (primitive).
        Achtung: je nach IsaacGym-Version kann die helper-API anders heißen.
        Falls create_sphere nicht vorhanden ist, kannst du alternativ ein kleines Box-Asset erstellen.
        """
        ball_options = gymapi.AssetOptions()
        ball_options.fix_base_link = False
        ball_options.density = 200
        ball_options.angular_damping = 0.5
        ball_options.linear_damping = 0.75
        ball_options.max_angular_velocity = 1000.0
        ball_options.max_linear_velocity = 20.0
        ball_options.disable_gravity = False
        ball_options.replace_cylinder_with_capsule = False
        ball_options.thickness = 0.01

        # versuchen, ein sphärisches Primitive-Asset zu erstellen
        try:
         ball_asset = self.gym.create_sphere(self.sim, radius, ball_options)  # manche builds: create_sphere_asset/create_sphere
        except Exception:
         # Fallback: sehr kleines Box-Primitive, falls Sphere-Helper nicht verfügbar
         # hx,hy,hz halbe Kantenlängen
         hx = radius; hy = radius; hz = radius
         ball_asset = self.gym.create_box(self.sim, hx, hy, hz, ball_options)

        # setze dof/body Informationen falls nötig (meist 1 rigid body)
        return ball_asset

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com_x"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com_y"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com_z"), return_noise=True
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
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_ball = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
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
        self.ball_root_states = self.root_states_all[self.num_envs:]
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
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.actions_raw = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.actions_gait = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.delay_ball_pos = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_forces_ball = torch.zeros(self.num_envs, 1, 3, dtype=torch.float, device=self.device)
        self.pushing_torques_ball = torch.zeros(self.num_envs, 1, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
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
        self.last_relative_ball_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.last_ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.filtered_ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.buffered_global_ball_size = 4
        self.buffered_global_ball_pos = torch.zeros(self.num_envs, self.buffered_global_ball_size, 3, dtype=torch.float, device=self.device)
        self.kick_leg_was_obvious = torch.zeros(self.num_envs, dtype=torch.int, device=self.device) # 1 for left, 0 unclear, -1 right
        self.ball_walk_target_overshoot = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.ball_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.last_ball_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_vel_before_kick = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.direction = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.direction_angle = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.direction_angle_current = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.robot_direction_ref = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.range = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_end_target = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.ball_kick_pose = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_close_to_kick_pose = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.more_harsh_walk_speed_condition = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.origin_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.inside_kick_pose_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.ball_noise_range = torch.zeros(self.num_envs, 2, 2, dtype=torch.float, device=self.device)

        self.zero_obs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ones_obs = torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        self.ball_reset_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Helper tensors
        self.ball_kicked_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_kick_direction_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_velocity_norm = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_velocity_norm_at_kick = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_still_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.kick_pose_translation_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.allow_ball_kick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.kick_sole_yaw_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.last_world_pose = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.current_world_pose = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.feet_height_over_ground = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.ball_freeze_at_kick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_pos_at_kick = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
        self.ball_request_resetted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.odo_buffer = torch.zeros(self.num_envs, 10, 3, dtype=torch.float, device=self.device)
        self.odo_noise_factor = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

        # Additional kick flags
        self.flag_is_strong_kick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.flag_is_inaccurate_kick = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.flag_allow_deviation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.kick_infos_left = torch.zeros(2, 10000, dtype=torch.float, device=self.device)
        self.kick_infos_right = torch.zeros(2, 10000, dtype=torch.float, device=self.device)
        self.kick_infos_idx_left = 0
        self.kick_infos_idx_right = 0

        self.left_full = False
        self.right_full = False
        self.request_changed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.changed_ball_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Note, this is currently only configured for the K1!
        self.ankle_polygon = torch.tensor(
            [
                [ 0.1,    0.375],
                [ 0.375,  0.0],
                [ 0.1,   -0.375],
                [-0.67,  -0.375],
                [-0.875,  0.0],
                [-0.67,   0.375],
            ],
            device=self.device,
            dtype=torch.float32,
        )

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, which will be called to compute the total reward.
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
        self._resample_kick_ball(ids)
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0
        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_kick_type"][0] / self.dt),
            int(self.cfg["commands"]["resampling_kick_type"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

        self.terminate_counter[env_ids] = 0
        self.ball_reset_counter[env_ids] = 0
        self.feet_swing_counter[env_ids, :] = 0

        self.dof_stiffness[env_ids, :] = self.default_dof_stiffness
        self.dof_damping[env_ids, :] = self.default_dof_damping
        self.dof_stiffness[env_ids] = apply_randomization(self.dof_stiffness[env_ids], self.cfg["randomization"]["dof_stiffness"])
        self.dof_damping[env_ids] = apply_randomization(self.dof_damping[env_ids], self.cfg["randomization"]["dof_damping"])

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
        roll_joints = [1, 5, 7, 11]
        other_joints = [2, 8]
        if self.cfg["algorithm"]["use_waist"]:
            pitch_joints = [j + 1 for j in pitch_joints]
            roll_joints = [j + 1 for j in roll_joints]
            other_joints = [j + 1 for j in other_joints]
            other_joints.append(0)

        pitch_cols = torch.tensor(pitch_joints, device=self.device)
        roll_cols = torch.tensor(roll_joints, device=self.device)
        other_cols = torch.tensor(other_joints, device=self.device)
        self.dof_pos_offset[env_ids] = 0

        self.dof_pos_offset.index_put_(
            (env_ids.unsqueeze(1), pitch_cols),
            apply_randomization(self.dof_pos_offset[env_ids][:, pitch_cols], self.cfg["randomization"]["dof_pos_offset_pitch"])
        )
        self.dof_pos_offset.index_put_(
            (env_ids.unsqueeze(1), roll_cols),
            apply_randomization(self.dof_pos_offset[env_ids][:, roll_cols], self.cfg["randomization"]["dof_pos_offset_roll"])
        )
        self.dof_pos_offset.index_put_(
            (env_ids.unsqueeze(1), other_cols),
            apply_randomization(self.dof_pos_offset[env_ids][:, other_cols], self.cfg["randomization"]["dof_pos_offset_other"])
        )

        self._reset_ball_idx(env_ids)

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.delay_ball_pos[env_ids] = torch.randint(0, 3, (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_ball_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._resample_input_flags(env_ids)
        self.ball_close_to_kick_pose[env_ids] = 1
        self.ball_kicked_counter[env_ids] = 0
        self.ball_vel_before_kick[env_ids] = 0
        self.ball_still_counter[env_ids] = 0
        self.inside_kick_pose_counter[env_ids] = 0
        self.changed_ball_vel[env_ids] = False

        reset_ball_root_states_ids = env_ids[torch_rand_float(
                0, 1, (len(env_ids), 1), device=self.device
            ).squeeze(1) > self.cfg["commands"]["keep_old_ball_when_reset"]]

        self._reset_ball_root_states(reset_ball_root_states_ids)
        self._resample_kick_ball(env_ids)
        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.episode_length_buf[env_ids] = 0
        self.cmd_resample_time[env_ids] = 0
        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_kick_type"][0] / self.dt),
            int(self.cfg["commands"]["resampling_kick_type"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )
        self.allow_ball_kick[env_ids] = 0
        self.ball_walk_target_overshoot[env_ids] = 0
        self.kick_leg_was_obvious[env_ids] = 0
        self.kick_sole_yaw_ratio[env_ids] = 0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
        self.odo_buffer[env_ids, :] = 0
        self.ball_freeze_at_kick[env_ids] = False

        self.odo_noise_factor[env_ids, :] = 0
        self.odo_noise_factor[env_ids, 0] = apply_randomization(self.odo_noise_factor[env_ids, 0], self.cfg["noise"]["ball_pos_odo_noise_x"])
        self.odo_noise_factor[env_ids, 1] = apply_randomization(self.odo_noise_factor[env_ids, 1], self.cfg["noise"]["ball_pos_odo_noise_y"])

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return

        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.prev_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.0
        self.custom_dof_vel[env_ids] = 0.0
        self.filtered_custom_dof_vel[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_actions[env_ids] = torch.cat((self.dof_pos[env_ids], torch.zeros(len(env_ids), 1, dtype=torch.float, device=self.device)), dim=-1)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_ball_root_states(self, env_ids):
        if len(env_ids) == 0:
            return

        self.request_changed[env_ids] = 0
        ball_vel_mask = self.ball_root_states[env_ids, 0] >= self.terrain.env_width * 0.5
        ball_vel_ids = env_ids[ball_vel_mask]
        ball_still_mask = self.ball_root_states[env_ids, 0] < self.terrain.env_width * 0.5
        ball_still_ids = env_ids[ball_still_mask]
        self.ball_root_states[env_ids, :3] = 0
        self.ball_root_states[ball_still_ids, 0] = apply_randomization(self.ball_root_states[ball_still_ids, 0], self.cfg["randomization"].get("init_ball_pos_x"))
        self.ball_root_states[ball_vel_ids, 0] = apply_randomization(self.ball_root_states[ball_vel_ids, 0], self.cfg["randomization"].get("init_ball_pos_far_x"))
        x_ball = self.ball_root_states[env_ids, 0]
        y_ball = self.ball_root_states[env_ids, 1]
        rot = torch.randint(
            -180,
            180,
            (len(env_ids),),
            device=self.device,
        ) / 180.0 * torch.pi
        rot = (rot + torch.pi) % (2 * torch.pi) - torch.pi
        self.ball_root_states[env_ids, 0] = x_ball * torch.cos(rot) - y_ball * torch.sin(rot)
        self.ball_root_states[env_ids, 1] = x_ball * torch.sin(rot) + y_ball * torch.cos(rot)
        self.ball_root_states[env_ids, :3] = self.root_states[env_ids, :3] + quat_rotate(self.root_states[env_ids, 3:7], self.ball_root_states[env_ids, :3])
        self.ball_root_states[env_ids, 2] = self.terrain.terrain_heights(self.ball_root_states[env_ids, :2]) + self.cfg["asset"]["ball"]["radius"] + self.cfg["randomization"]["ball_radius"]["range"][1] + 0.01
        self.ball_root_states[env_ids, 7:13] = 0

        # Add velocity only to second half of evns
        self.ball_root_states[ball_vel_ids, 7:9] = apply_randomization(self.ball_root_states[ball_vel_ids, 7:9], self.cfg["randomization"].get("init_ball_vel"))

        # Small percentage to force the ball to roll straight towards the robot
        ball_towards_robot_ids = ball_vel_ids[torch_rand_float(
                0, 1, (len(ball_vel_ids), 1), device=self.device
            ).squeeze(1) < 0.05]
        ball_to_robot_vector = self.root_states[ball_towards_robot_ids, :2] - self.ball_root_states[ball_towards_robot_ids, :2]

        # Make sure to prevent division by 0
        self.ball_root_states[ball_towards_robot_ids, 7:9] = torch.where((torch.norm(ball_to_robot_vector, dim=-1) > 0).unsqueeze(-1), ball_to_robot_vector / torch.norm(ball_to_robot_vector, dim=-1).unsqueeze(-1) * torch.norm(self.ball_root_states[ball_towards_robot_ids, 7:9], dim=-1).unsqueeze(-1), self.zero_obs[ball_towards_robot_ids].unsqueeze(-1).expand(-1, 2))

        self.last_ball_pos[env_ids] = self.ball_root_states[env_ids, :3]
        self.last_ball_vel[env_ids] = self.ball_root_states[env_ids, 7:10]

        # 10% of envs shall get a still ball very close to robot, to learn better walking around it
        ball_close_to_robot_first_half_ids = ball_still_ids[torch_rand_float(
                0, 1, (len(ball_still_ids), 1), device=self.device
            ).squeeze(1) < self.cfg["commands"]["force_ball_close"]]
        ball_close_to_robot_second_half_ids = ball_vel_ids[torch_rand_float(
                0, 1, (len(ball_vel_ids), 1), device=self.device
            ).squeeze(1) < self.cfg["commands"]["force_ball_close"]]

        if len(ball_close_to_robot_first_half_ids) > 0:
            self.ball_root_states[ball_close_to_robot_first_half_ids, 7:9] = 0
            ball_close_vec_fh = self.ball_root_states[ball_close_to_robot_first_half_ids, :2] - self.root_states[ball_close_to_robot_first_half_ids, :2]
            self.ball_root_states[ball_close_to_robot_first_half_ids, :2] = self.root_states[ball_close_to_robot_first_half_ids, :2] + ball_close_vec_fh / torch.norm(ball_close_vec_fh, dim=-1).unsqueeze(-1) * self.cfg["randomization"]["init_ball_pos_x"]["range"][0]

        if len(ball_close_to_robot_second_half_ids) > 0:
            self.ball_root_states[ball_close_to_robot_second_half_ids, 7:9] = 0
            ball_close_vec_sh = self.ball_root_states[ball_close_to_robot_second_half_ids, :2] - self.root_states[ball_close_to_robot_second_half_ids, :2]
            self.ball_root_states[ball_close_to_robot_second_half_ids, :2] = self.root_states[ball_close_to_robot_second_half_ids, :2] + ball_close_vec_sh / torch.norm(ball_close_vec_sh, dim=-1).unsqueeze(-1) * self.cfg["randomization"]["init_ball_pos_far_x"]["range"][0]

        # Ball got teleported, fill poses
        self.buffered_global_ball_pos[env_ids, :, :] = self.ball_root_states[env_ids, :3].unsqueeze(1).expand(-1, self.buffered_global_ball_pos.shape[1], -1)

    def _reset_root_states(self, env_ids):
        # Robot
        if len(env_ids) == 0:
            return

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

        self._reset_ball_root_states(env_ids)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return

        # Ball
        out_x_min_ball = self.ball_root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max_ball = self.ball_root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min_ball = self.ball_root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max_ball = self.ball_root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        out_z_max_ball = self.ball_root_states[:, 2] < 0
        self.ball_root_states[out_x_min_ball, 0] += self.terrain.env_width + self.terrain.border_size
        self.ball_root_states[out_x_max_ball, 0] -= self.terrain.env_width + self.terrain.border_size
        self.ball_root_states[out_y_min_ball, 1] += self.terrain.env_length + self.terrain.border_size
        self.ball_root_states[out_y_max_ball, 1] -= self.terrain.env_length + self.terrain.border_size
        self.ball_root_states[out_z_max_ball, 2] = 0.3

        # Ball got teleported, fill poses
        self.buffered_global_ball_pos[out_x_min_ball, 0] += self.terrain.env_width + self.terrain.border_size
        self.buffered_global_ball_pos[out_x_max_ball, 0] -= self.terrain.env_width + self.terrain.border_size
        self.buffered_global_ball_pos[out_y_min_ball, 1] += self.terrain.env_length + self.terrain.border_size
        self.buffered_global_ball_pos[out_y_max_ball, 1] -= self.terrain.env_length + self.terrain.border_size

        self.ball_end_target[out_x_min_ball, 0] += self.terrain.env_width + self.terrain.border_size
        self.ball_end_target[out_x_max_ball, 0] -= self.terrain.env_width + self.terrain.border_size
        self.ball_end_target[out_y_min_ball, 1] += self.terrain.env_length + self.terrain.border_size
        self.ball_end_target[out_y_max_ball, 1] -= self.terrain.env_length + self.terrain.border_size

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

        self.last_world_pose[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.last_world_pose[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.last_world_pose[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.last_world_pose[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        self.current_world_pose[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.current_world_pose[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.current_world_pose[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.current_world_pose[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size

        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any() or out_x_min_ball.any() or out_x_max_ball.any() or out_y_min_ball.any() or out_y_max_ball.any() or out_z_max_ball.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
            self._refresh_feet_state()

    def _resample_kick_ball(self, env_ids):
        # Randomize for kicking environments
        if len(env_ids) == 0:
            return

        _, _, yaw = get_euler_xyz(self.root_states[env_ids, 3:7])

        self.origin_pos[env_ids,:] = 0
        for i in self.feet_indices:
            self.origin_pos[env_ids] += self.body_states[env_ids, i, 0:3] + quat_rotate(self.body_states[env_ids, i, 3:7], to_torch(self.cfg["asset"]["feet_sole_pos"], device=self.device).unsqueeze(0).expand(len(env_ids), -1))
        self.origin_pos[env_ids] /= len(self.feet_indices)

        self.current_world_pose[env_ids, :2] = self.origin_pos[env_ids, :2]
        self.current_world_pose[env_ids, 2] = yaw
        self.last_world_pose[env_ids] = self.current_world_pose[env_ids]

        self.ball_pos[env_ids,:] = 0
        self.ball_vel[env_ids,:] = 0
        self.direction_angle[env_ids] = 0
        self.range[env_ids] = 0

        self.ball_pos[env_ids, :] = self.ball_root_states[env_ids, :3] - self.origin_pos[env_ids, :3]
        self.ball_pos[env_ids] = quat_rotate_inverse(self.base_quat_z[env_ids], self.ball_pos[env_ids, :])
        self.ball_pos[env_ids, 2] = 0

        self.last_relative_ball_pos[env_ids, :] = self.ball_pos[env_ids, :2]

        self.ball_vel[env_ids, :] = quat_rotate_inverse(self.base_quat_z[env_ids], self.ball_root_states[env_ids, 7:10])
        self.ball_vel[env_ids, 2] = 0

        self.direction_angle[env_ids] = torch_rand_float(
                -1, 1, (len(env_ids), 1), device=self.device
            ).squeeze(1)

        self.robot_direction_ref[env_ids] = yaw
        self.direction_angle_current[env_ids] = self.direction_angle[env_ids] * torch.pi - (yaw - self.robot_direction_ref[env_ids])
        self.direction_angle_current[env_ids] = (self.direction_angle_current[env_ids] + torch.pi) % (2 * torch.pi) - torch.pi

        self.range[env_ids[~self.flag_is_strong_kick[env_ids]]] = apply_randomization(self.range[env_ids[~self.flag_is_strong_kick[env_ids]]], self.cfg["ball"]["kick_normal"]["range"])
        self.range[env_ids[self.flag_is_strong_kick[env_ids]]] = self.cfg["ball"]["kick_strong"]["range"]

        global_direction = self.robot_direction_ref[env_ids] + self.direction_angle[env_ids] * torch.pi
        global_direction = (global_direction + torch.pi) % (2 * torch.pi) - torch.pi
        sin_dir = torch.sin(global_direction)
        cos_dir = torch.cos(global_direction)
        self.ball_end_target[env_ids, 0] = self.ball_root_states[env_ids, 0] + cos_dir * self.range[env_ids]
        self.ball_end_target[env_ids, 1] = self.ball_root_states[env_ids, 1] + sin_dir * self.range[env_ids]
        self.ball_kick_pose[env_ids, 0] = self.ball_root_states[env_ids, 0] + cos_dir * -0.35
        self.ball_kick_pose[env_ids, 1] = self.ball_root_states[env_ids, 1] + sin_dir * -0.35
        self.current_kick_pose_distance[env_ids] = torch.norm(self.ball_kick_pose[env_ids, :2] - self.base_pos[env_ids, 0:2], dim=1)
        self.current_kick_pose_distance[env_ids] = torch.norm(self.ball_root_states[env_ids, 0:2] - self.base_pos[env_ids, 0:2], dim=1)
        self.last_kick_pose_distance[env_ids] = self.current_kick_pose_distance[env_ids]

        self.ball_close_to_kick_pose[env_ids] = 1
        self.ball_kicked_counter[env_ids] = 0
        self.inside_kick_pose_counter[env_ids] = 0
        self.ball_still_counter[env_ids] = 0
        self.ball_vel_before_kick[env_ids] = 0

        ball_in_kick_angle = torch.clone(self.ball_pos[env_ids])
        ball_in_kick_angle[:,0] = torch.cos(-self.direction_angle_current[env_ids]) * self.ball_pos[env_ids,0] - torch.sin(-self.direction_angle_current[env_ids]) * self.ball_pos[env_ids,1]
        ball_in_kick_angle[:,1] = torch.cos(-self.direction_angle_current[env_ids]) * self.ball_pos[env_ids,1] + torch.sin(-self.direction_angle_current[env_ids]) * self.ball_pos[env_ids,0]
        self.filtered_ball_pos[env_ids] = ball_in_kick_angle

        self.last_ball_pos[env_ids] = self.ball_root_states[env_ids, :3]
        self.last_ball_vel[env_ids] = self.ball_root_states[env_ids, 7:10]
        
        self.ball_request_resetted[env_ids] = False

    def _resample_input_flags(self, env_ids):
        self.flag_is_strong_kick[env_ids] = torch_rand_float(
                0, 1, (len(env_ids), 1), device=self.device
            ).squeeze(1) < 0.5
        self.flag_is_inaccurate_kick[env_ids] = torch.where(torch_rand_float(
                0, 1, (len(env_ids), 1), device=self.device
            ).squeeze(1) < 0.3, 1.0, 0.0)
        self.flag_allow_deviation[env_ids] = torch_rand_float(
                0, 1, (len(env_ids), 1), device=self.device
            ).squeeze(1) < 0.5

        self.flag_allow_deviation[self.heavy_ball] = True

        if self.cfg["rewards"]["ball_parameters"]["disable_deviation_flag"]:
            self.flag_allow_deviation[:] = 0
        if self.cfg["rewards"]["ball_parameters"]["disable_extra_deviation_flag"]:
            self.flag_is_inaccurate_kick[:] = 0

    def _resample_commands(self):
        l_dis, r_dis = self._helper_ball_sole_distance()
        min_distance = torch.minimum(l_dis, r_dis)
        env_ids = ((self.episode_length_buf == self.cmd_resample_time) & (min_distance > self.cfg["asset"]["ball"]["radius"] + self.cfg["randomization"]["ball_radius"]["range"][1] + 0.04)).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_kick_type"][0] / self.dt),
            int(self.cfg["commands"]["resampling_kick_type"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

    def _update_kick_range(self, env_ids):
        if len(env_ids) == 0:
            return
        self.range[env_ids] = 0
        self.range[env_ids[~self.flag_is_strong_kick[env_ids]]] = apply_randomization(self.range[env_ids[~self.flag_is_strong_kick[env_ids]]], self.cfg["ball"]["kick_normal"]["range"])
        self.range[env_ids[self.flag_is_strong_kick[env_ids]]] = self.cfg["ball"]["kick_strong"]["range"]

        global_direction = self.robot_direction_ref[env_ids] + self.direction_angle[env_ids] * torch.pi
        global_direction = (global_direction + torch.pi) % (2 * torch.pi) - torch.pi
        sin_dir = torch.sin(global_direction)
        cos_dir = torch.cos(global_direction)
        self.ball_end_target[env_ids, 0] = self.ball_root_states[env_ids, 0] + cos_dir * self.range[env_ids]
        self.ball_end_target[env_ids, 1] = self.ball_root_states[env_ids, 1] + sin_dir * self.range[env_ids]

    def step(self, actions):
        # pre physics step
        self.actions_raw[:] = actions
        self.actions_gait[:] = self.actions[:,-1]
        self.actions_gait[:] = torch.clip(self.actions_gait[:], self.cfg["normalization"]["action_frequence_limit"][0], self.cfg["normalization"]["action_frequence_limit"][1])
        self.actions_gait += self.cfg["rewards"]["target_frequence"]
        self.actions_gait[self.still_envs] = 0
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = torch.clip(self.default_dof_pos + self.dof_pos_offset + self.cfg["control"]["action_scale"] * self.actions[:,:self.num_actions-1], min=self.dof_pos_limits[:,0], max=self.dof_pos_limits[:,1])

        # Code to workaround the parallel structure. Not used for policies used at competitions
        #envs_left_clip = ~self.is_inside_ankle_polygon(dof_targets[:, -8], dof_targets[:, -7])
        #envs_right_clip = ~self.is_inside_ankle_polygon(dof_targets[:, -2], dof_targets[:, -1])

        #if envs_left_clip.any():
        #    dof_targets[envs_left_clip, -8], dof_targets[envs_left_clip, -7] = self.ankle_polygon_intersection(dof_targets[envs_left_clip, -8], dof_targets[envs_left_clip, -7])
        #if envs_right_clip.any():
        #    dof_targets[envs_right_clip, -2], dof_targets[envs_right_clip, -1] = self.ankle_polygon_intersection(dof_targets[envs_right_clip, -2], dof_targets[envs_right_clip, -1])

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
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.actions_gait, 1.0)

        self.buffered_global_ball_pos[:] = self.buffered_global_ball_pos.roll(shifts=-1, dims=1)
        self.buffered_global_ball_pos[:, -1] = self.ball_root_states[:, :3]
        self.request_changed[:] += 1

        # Handle all ball related stuff
        self._update_kick_tensors()

        l_dis, r_dis = self._helper_ball_sole_distance()
        min_distance = torch.minimum(l_dis, r_dis)
        ball_request_reset_ids_checks = ((self.ball_close_to_kick_pose < 1.0) & (min_distance < 0.5) & (min_distance > 0.25) & (self.ball_kicked_counter == 0)).nonzero(as_tuple=False).flatten()
        ball_request_reset_mask = torch_rand_float(
                0, 1, (len(ball_request_reset_ids_checks), 1), device=self.device
            ).squeeze(1) < self.cfg["commands"]["reset_ball_request"]
        ball_request_reset_ids = ball_request_reset_ids_checks[ball_request_reset_mask]
        self._resample_kick_ball(ball_request_reset_ids)
        self.request_changed[ball_request_reset_ids] = 0

        self._kick_robots()
        self._push_robots()
        self._update_odometrie_buffer()
        self._check_termination()
        self._compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self._reset_idx(env_ids)

        reset_ball_ids = self.reset_ball.nonzero(as_tuple=False).flatten()
        self._reset_ball_idx(reset_ball_ids)

        self._teleport_robot()
        self._resample_commands()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.last_feet_yaw[:] = self.feet_yaw
        self.last_feet_contact[:] = self.feet_contact
        self.last_ball_pos[:] = self.ball_root_states[:, :3]
        self.last_ball_vel[:] = self.ball_root_states[:, 7:10]
        self.last_kick_pose_distance[:] = self.current_kick_pose_distance
        self.last_relative_ball_pos[:] = self.ball_pos[:, :2]
        self.last_world_pose[:] = self.current_world_pose

        # Only for evaluation
        #self._add_kick_info()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _update_kick_tensors(self):
        _, _, yaw = get_euler_xyz(self.root_states[:, 3:7])

        # 1) First check for ball kicked
        l_dis, r_dis = self._helper_ball_sole_distance()
        min_distance = torch.minimum(l_dis, r_dis)
        ball_acc = torch.norm(self.ball_root_states[:, 7:9] - self.last_ball_vel[:, :2], dim=-1)

        isRightKickSole = l_dis >= r_dis
        sole_movement = torch.where((~isRightKickSole).unsqueeze(-1), self.feet_pos[:, 0, :2] - self.last_feet_pos[:, 0, :2], self.feet_pos[:, 1, :2] - self.last_feet_pos[:, 1, :2])
        sole_to_ball = torch.where((~isRightKickSole).unsqueeze(-1), self.ball_root_states[:, :2] - self.feet_pos[:, 0, :2], self.ball_root_states[:, :2] - self.feet_pos[:, 1, :2])
        kick_sole_direction = torch.atan2(sole_movement[:,1], sole_movement[:,0])
        sole_movement_speed = torch.norm(sole_movement, dim=-1) / self.dt
        sole_to_ball_direction = torch.atan2(sole_to_ball[:,1], sole_to_ball[:,0])
        self.allow_ball_kick[:] = torch.where(((torch.abs(((kick_sole_direction - sole_to_ball_direction + torch.pi) % (2 * torch.pi)) - torch.pi) < torch.pi / 2.0) & (self.episode_length_buf > 5) & (sole_movement_speed > 0.1) & (self.feet_height_over_ground.gather(1, isRightKickSole.long().unsqueeze(1)).squeeze(1) > 0.003) | (self.episode_length_buf > 50)), True, False)

        ball_still_allow_kick = (self.ball_still_counter > 10) & (ball_acc > 0.01) # If ball is still long enough, every kick is counted
        ball_moved_kick = (ball_acc > 0.1) & self.allow_ball_kick # Moved towards ball -> your fault that it is counted as a kick
        ball_kicked_mask = (ball_still_allow_kick | ball_moved_kick | self.heavy_ball) & (min_distance < self.cfg["asset"]["ball"]["radius"] + self.cfg["randomization"]["ball_radius"]["range"][1] + 0.04) # ball kicked event
        self.ball_kicked_counter[:] += torch.where((self.ball_kicked_counter[:] > 0) | ball_kicked_mask, 1, 0)

        # 2) save global ball pos at kick start
        self.ball_pos_at_kick[:] = torch.where((self.ball_kicked_counter == 0).unsqueeze(-1), self.ball_root_states[:, :3], self.ball_pos_at_kick)

        # 3) determine current relative ball position
        # 3.1) Get relative coordinate system. It is between both sole origins
        self.origin_pos[:,:] = 0
        for i in self.feet_indices:
            self.origin_pos += self.body_states[:, i, 0:3] + quat_rotate(self.body_states[:, i, 3:7], to_torch(self.cfg["asset"]["feet_sole_pos"], device=self.device).unsqueeze(0).expand(self.num_envs, -1))
        self.origin_pos /= len(self.feet_indices)

        self.current_world_pose[:, :2] = self.origin_pos[:, :2]
        self.current_world_pose[:, 2] = yaw

        self.ball_velocity_norm = torch.norm(self.ball_root_states[:, 7:10], dim=-1)

        # 3.2) Determin which ball positions should freeze. We want to freeze the ball during a kick, because often on the real robot we do not see detect the ball during the kick
        random_ball_freeze_candidates = self.ball_kicked_counter == 1
        random_ball_freeze_id_candidates = random_ball_freeze_candidates.nonzero(as_tuple=False).flatten()
        self.ball_freeze_at_kick[random_ball_freeze_id_candidates] |= torch_rand_float(
                0, 1, (len(random_ball_freeze_id_candidates), 1), device=self.device
            ).squeeze(1) < self.cfg["commands"]["ball_freeze_at_kick"]

        # 3.3) Update ball position and velocity for the observation
        idx = torch.arange(self.buffered_global_ball_pos.size(0))
        self.ball_pos[:, :] = torch.where(self.ball_freeze_at_kick.unsqueeze(-1), self.ball_pos_at_kick, self.buffered_global_ball_pos[idx, self.delay_ball_pos, :3]) - self.origin_pos
        self.ball_pos[:, :] = quat_rotate_inverse(self.base_quat_z[:], self.ball_pos[:, :])
        self.ball_pos[:, 2] = 0
        self.ball_vel[:, :] = quat_rotate_inverse(self.base_quat_z[:], self.ball_root_states[:, 7:10])
        self.ball_vel[:, 2] = 0

        # 4) Update kick direction. If ball is currently rolling, the direction and range shall show torwards the target
        ball_to_target = self.ball_end_target[:] - self.ball_root_states[:, :2]
        global_direction = torch.atan2(ball_to_target[:,1], ball_to_target[:,0])
        self.direction_angle[:] = global_direction - self.robot_direction_ref[:]
        self.direction_angle[:] = (self.direction_angle[:] + torch.pi) % (2 * torch.pi) - torch.pi
        self.direction_angle[:] /= torch.pi
        self.range[:] = torch.norm(ball_to_target[:, :2], dim=-1).clip(max=self.cfg["ball"]["kick_strong"]["range"])
        self.direction_angle_current[:] = self.direction_angle * torch.pi - (yaw - self.robot_direction_ref)
        self.direction_angle_current[:] = (self.direction_angle_current + torch.pi) % (2 * torch.pi) - torch.pi

        # 5) Save some info and slow down ball
        ball_distance = torch.norm(self.ball_pos[:, :2], dim=-1)
        self.ball_still_counter[:] = torch.where(self.ball_kicked_counter > 0, self.ball_still_counter, torch.where(self.ball_velocity_norm < 0.0001, self.ball_still_counter[:] + 1, self.zero_obs))
        self.ball_velocity_norm_at_kick[:] = torch.where(self.ball_kicked_counter <= 1, self.ball_velocity_norm, torch.max(self.ball_velocity_norm_at_kick, self.ball_velocity_norm))
        self.ball_vel_before_kick[:] = torch.where(self.ball_kicked_counter > 0, self.ball_vel_before_kick, self.ball_velocity_norm)
        self.ball_root_states[:, 7:10] *= torch.where((self.ball_kicked_counter == 0) & (self.ball_velocity_norm > 1.5), 0.97 - 0.2 * ((ball_distance - 2).clip(min=0.0)), 1.0).unsqueeze(-1)

        # 6) Determine relative ball position to robot, but relative in the kick direction
        sin_dir = torch.sin(global_direction)
        cos_dir = torch.cos(global_direction)

        use_direction_angle_current_with_offset = (self.direction_angle_current + torch.pi) % (2 * torch.pi) - torch.pi
        waist_angle = self.dof_pos[:, 0] if self.cfg["algorithm"]["use_waist"] else self.zero_obs
        kick_pose_rotation_ratio = torch.clip((torch.abs(use_direction_angle_current_with_offset + waist_angle) - self.cfg["rewards"]["ball_parameters"]["ball_rotation_offset"]) / 0.4, min=0.0, max=1.0)
        ball_in_kick_angle = torch.clone(self.ball_pos)
        ball_in_kick_angle[:,0] = torch.cos(-use_direction_angle_current_with_offset) * self.ball_pos[:,0] - torch.sin(-use_direction_angle_current_with_offset) * self.ball_pos[:,1]
        ball_in_kick_angle[:,1] = torch.cos(-use_direction_angle_current_with_offset) * self.ball_pos[:,1] + torch.sin(-use_direction_angle_current_with_offset) * self.ball_pos[:,0]

        # 7) Checks for later for reward function to punish "overshooting" to the left/right
        old_kick_leg_was_obvious = torch.clone(self.kick_leg_was_obvious)
        self.kick_leg_was_obvious[:] = torch.where((self.kick_leg_was_obvious * ball_in_kick_angle[:,1] > 0) | (torch.abs(ball_in_kick_angle[:,1]) > 0.05), torch.sign(ball_in_kick_angle[:,1]), 0)
        self.ball_walk_target_overshoot[:] = torch.where((old_kick_leg_was_obvious != self.kick_leg_was_obvious) & (self.kick_leg_was_obvious == 0) & (self.episode_length_buf > 25), 10, self.ball_walk_target_overshoot[:] - 1)

        # 8) Calculations for the kick pose. Also used to scale kick reward
        ball_y_dis_ratio = ((torch.abs(ball_in_kick_angle[:, 1]) - self.cfg["rewards"]["ball_parameters"]["ball_y_dis_ratio"][0] * self.cfg["rewards"]["ball_parameters"]["kick_pose_y_off"]) / (self.cfg["rewards"]["ball_parameters"]["ball_y_dis_ratio"][1] * self.cfg["asset"]["ball"]["radius"])).clip(min=0.0, max=1.0)
        self.kick_pose_translation_ratio = torch.clip((torch.abs(torch.atan2(ball_in_kick_angle[:,1], ball_in_kick_angle[:,0])) - 0.3) / 1.2, min=0.0, max=1.0)
        kick_pose_distance_ratio = ((ball_distance - self.cfg["rewards"]["ball_parameters"]["ball_distance_offset"]) / 0.4).clip(min=0.0, max=1.0)
        min_kick_pose_ratio = torch.max(ball_y_dis_ratio, torch.max(torch.max(kick_pose_distance_ratio, kick_pose_rotation_ratio), self.kick_pose_translation_ratio))
        self.ball_close_to_kick_pose[:] = (-1.0 + 2.0 * min_kick_pose_ratio).clip(min=self.ball_close_to_kick_pose[:] - self.dt*2, max =self.ball_close_to_kick_pose[:] + self.dt*2)
        self.more_harsh_walk_speed_condition[:] = (torch.where(torch.max(torch.max(kick_pose_distance_ratio, kick_pose_rotation_ratio), self.kick_pose_translation_ratio) < 1, 1, self.more_harsh_walk_speed_condition[:] - 0.1)).clip(min=0.0)

        kick_pose_counter_increase_value = torch.where(self.ball_kicked_counter > 0, 0, self.cfg["rewards"]["ball_parameters"]["kick_pose_increase_value"])
        self.inside_kick_pose_counter[:] = torch.where((self.ball_close_to_kick_pose < 1) | (self.ball_kicked_counter > 1), (self.inside_kick_pose_counter + kick_pose_counter_increase_value).clip(max=1.0), 0)

        lowpass_factor = torch.where(min_kick_pose_ratio == 1.0, 0.9, 0.2).unsqueeze(-1)
        self.filtered_ball_pos[:] = (1.0 - lowpass_factor) * self.filtered_ball_pos + lowpass_factor * ball_in_kick_angle

        kp_x = self.cfg["rewards"]["ball_parameters"]["kick_pose_x"]

        self.ball_kick_pose[:,0] = self.ball_root_states[:, 0] + cos_dir * (kp_x[0] + (kp_x[1] - kp_x[0]) * (1.0 * torch.clip(self.ball_close_to_kick_pose[:], min=0.0)))
        self.ball_kick_pose[:,1] = self.ball_root_states[:, 1] + sin_dir * (kp_x[0] + (kp_x[1] - kp_x[0]) * (1.0 * torch.clip(self.ball_close_to_kick_pose[:], min=0.0)))
        self.current_kick_pose_distance[:] = torch.norm(self.ball_kick_pose[:, :2] - self.base_pos[:, :2], dim=1)

        # 9.1) Randomly apply some ball velocity again
        apply_ball_force_candidates = (self.ball_kicked_counter == 0) & (~self.changed_ball_vel) & (min_distance > 0.35) & (min_distance < 0.55)
        apply_ball_force_candidates_ids = apply_ball_force_candidates.nonzero(as_tuple=False).flatten()
        apply_ball_force_candidates_mask = torch_rand_float(
                0, 1, (len(apply_ball_force_candidates_ids), 1), device=self.device
            ).squeeze(1) < self.cfg["commands"]["add_ball_vel"]
        apply_ball_force_ids = apply_ball_force_candidates_ids[apply_ball_force_candidates_mask]
        self.ball_vel[apply_ball_force_ids, 0] = torch.rand(len(apply_ball_force_ids), device=self.device) * self.cfg["commands"]["add_ball_vel_max_velocity"]
        self.ball_vel[apply_ball_force_ids, 1] = (torch.rand(len(apply_ball_force_ids), device=self.device) * self.cfg["commands"]["add_ball_vel_max_velocity"] - 0.5) * 2.0
        self.ball_root_states[apply_ball_force_ids, 7:9] = quat_rotate(self.base_quat_z[apply_ball_force_ids], self.ball_vel[apply_ball_force_ids, :])[:, :2]
        self.changed_ball_vel[apply_ball_force_ids] = True
        
        # 9.2) Randomly apply some ball velocity again, but this time in kick direction
        apply_ball_force_candidates = (self.ball_kicked_counter == 0) & (~self.changed_ball_vel) & (min_distance > 0.35) & (min_distance < 0.55) & (self.kick_pose_translation_ratio < 1) & (self.ball_pos[:, 0] > 0)
        apply_ball_force_candidates_ids = apply_ball_force_candidates.nonzero(as_tuple=False).flatten()
        apply_ball_force_candidates_mask = torch_rand_float(
                0, 1, (len(apply_ball_force_candidates_ids), 1), device=self.device
            ).squeeze(1) < self.cfg["commands"]["add_ball_vel_in_direction"]
        apply_ball_force_ids_2 = apply_ball_force_candidates_ids[apply_ball_force_candidates_mask]
        
        velocity_strength = (torch.rand(len(apply_ball_force_ids_2), device=self.device) + 1.0) / 2.0
        vel_direction = self.direction_angle_current[apply_ball_force_ids_2].clone()
        vel_direction[:] = apply_randomization(vel_direction, self.cfg["commands"]["ball_vel_in_direction_range"])
        self.ball_vel[apply_ball_force_ids_2, 0] = torch.cos(vel_direction) * self.cfg["commands"]["add_ball_vel_in_direction_max_velocity"] * velocity_strength
        self.ball_vel[apply_ball_force_ids_2, 1] = torch.sin(vel_direction) * self.cfg["commands"]["add_ball_vel_in_direction_max_velocity"] * velocity_strength
        self.ball_root_states[apply_ball_force_ids_2, 7:9] = quat_rotate(self.base_quat_z[apply_ball_force_ids_2], self.ball_vel[apply_ball_force_ids_2, :])[:, :2]
        self.changed_ball_vel[apply_ball_force_ids_2] = True

        # 9.3) Overwrite ball state
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))

        # 10) Determine kick sole yaw rotation
        _, _, feet_yaw_left = get_euler_xyz(self.feet_quat[:, 0])
        _, _, feet_yaw_right = get_euler_xyz(self.feet_quat[:, 1])
        target_yaw = torch.where(l_dis < r_dis, 1, -1) * self.cfg["rewards"]["ball_parameters"]["target_sole_yaw_angle"]
        kick_sole_yaw_diff = (target_yaw - (torch.where(l_dis < r_dis, feet_yaw_left, feet_yaw_right) - global_direction) + torch.pi) % (2 * torch.pi) - torch.pi
        kick_sole_yaw_ratio = torch.exp(-torch.square(kick_sole_yaw_diff) / self.cfg["rewards"]["ball_parameters"]["target_sole_yaw_sigma"])
        self.kick_sole_yaw_ratio[:] = torch.where(self.ball_kicked_counter <= 1, kick_sole_yaw_ratio, torch.min(kick_sole_yaw_ratio, self.kick_sole_yaw_ratio))

        # Drawings
        if not self.cfg["basic"]["headless"]:
            copy_ball = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
            sin_dir_other = torch.sin(global_direction)
            cos_dir_other = torch.cos(global_direction)
            copy_ball[:] = self.ball_root_states[:, :3]
            copy_ball[:,0] += cos_dir_other * self.range[:]
            copy_ball[:,1] += sin_dir_other * self.range[:]
            color_kick_pose = gymapi.Vec3(0.0, 0.0, 1.0)  # Blue
            color_green = gymapi.Vec3(0.0, 1.0, 0.0)  # Green


            walk_target = self._helper_walk_target()
            walk_target = quat_rotate(self.base_quat_z[:], walk_target) + self.base_pos

            # Now draw
            self.gym.clear_lines(self.viewer)

            #self.terrain.draw_terrain_friction(self.envs[0], self.gym, self.viewer)
            for env_id in range(0, self.num_envs):
                env_id = int(env_id)
                color = gymapi.Vec3(1.0, 0.0, 0.0) if self.flag_is_strong_kick[env_id] else gymapi.Vec3(1.0, 1.0, 0.0)  # Red for strong, yellow for normal
                start = gymapi.Vec3(*self.ball_root_states[env_id, :3].tolist())
                end   = gymapi.Vec3(*copy_ball[env_id].tolist())
                x, y, z = self.ball_kick_pose[env_id].tolist()
                kick_pose_point = gymapi.Vec3(x,y,z)
                kick_pose_point2 = gymapi.Vec3(x,y,z + 0.5)
                # Linie direkt zeichnen
                gymutil.draw_line(start, end, color, self.gym, self.viewer, self.envs[env_id])
                gymutil.draw_line(kick_pose_point, kick_pose_point2, color_kick_pose, self.gym, self.viewer, self.envs[env_id])

                gymutil.draw_line(kick_pose_point, gymapi.Vec3(walk_target[env_id][0], walk_target[env_id][1], 0.1), color_green, self.gym, self.viewer, self.envs[env_id])

    def _add_kick_info(self):
        '''
        This was the helper function to evaluate the kicks in IsaacGym
        '''

        l_dis, r_dis = self._helper_ball_sole_distance()

        kicked_mask_left = (self.ball_kicked_counter == 3) & (l_dis < r_dis)
        kicked_mask_right = (self.ball_kicked_counter == 3) & (l_dis > r_dis)
        angle_diff = (torch.atan2(self.ball_vel[:,1], self.ball_vel[:,0]) - self.direction_angle_current[:] + torch.pi) % (2 * torch.pi) - torch.pi

        vel_left = self.ball_velocity_norm_at_kick[kicked_mask_left]
        angle_left = angle_diff[kicked_mask_left]

        num_new_left = vel_left.shape[0]
        last_left_idx = self.kick_infos_idx_left
        # left
        if num_new_left > 0:
            end_idx = self.kick_infos_idx_left + num_new_left

            if end_idx <= self.kick_infos_left.shape[1]:
                self.kick_infos_left[0, self.kick_infos_idx_left:end_idx] = vel_left
                self.kick_infos_left[1, self.kick_infos_idx_left:end_idx] = angle_left
            else:
                # Wrap-around (Ringbuffer)
                first_part = self.kick_infos_left.shape[1] - self.kick_infos_idx_left
                second_part = num_new_left - first_part

                self.kick_infos_left[0, self.kick_infos_idx_left:] = vel_left[:first_part]
                self.kick_infos_left[1, self.kick_infos_idx_left:] = angle_left[:first_part]

                self.kick_infos_left[0, :second_part] = vel_left[first_part:]
                self.kick_infos_left[1, :second_part] = angle_left[first_part:]

            self.kick_infos_idx_left = end_idx % self.kick_infos_left.shape[1]

        # Right
        vel_right = self.ball_velocity_norm_at_kick[kicked_mask_right]
        angle_right = angle_diff[kicked_mask_right]

        num_new_right = vel_right.shape[0]
        last_right_idx = self.kick_infos_idx_right
        # left
        if num_new_right > 0:
            end_idx = self.kick_infos_idx_right + num_new_right

            if end_idx <= self.kick_infos_right.shape[1]:
                self.kick_infos_right[0, self.kick_infos_idx_right:end_idx] = vel_right
                self.kick_infos_right[1, self.kick_infos_idx_right:end_idx] = angle_right
            else:
                # Wrap-around (Ringbuffer)
                first_part = self.kick_infos_right.shape[1] - self.kick_infos_idx_right
                second_part = num_new_right - first_part

                self.kick_infos_right[0, self.kick_infos_idx_right:] = vel_right[:first_part]
                self.kick_infos_right[1, self.kick_infos_idx_right:] = angle_right[:first_part]

                self.kick_infos_right[0, :second_part] = vel_right[first_part:]
                self.kick_infos_right[1, :second_part] = angle_right[first_part:]

            self.kick_infos_idx_right = end_idx % self.kick_infos_right.shape[1]

        self.left_full = self.left_full or last_left_idx > self.kick_infos_idx_left
        self.right_full = self.right_full or last_right_idx > self.kick_infos_idx_right

        if self.left_full and self.right_full:
            vel_data = self.kick_infos_left[0]
            angle_data = self.kick_infos_left[1]

            vel_mean = vel_data.mean()
            vel_std = vel_data.std()

            angle_mean = angle_data.mean()
            angle_std = angle_data.std()
            print("left")
            print(vel_mean)
            print(vel_std)
            print(angle_mean * 180.0 / 3.141)
            print(angle_std * 180.0 / 3.141)
            print("----")
            vel_data = self.kick_infos_right[0]
            angle_data = self.kick_infos_right[1]

            vel_mean = vel_data.mean()
            vel_std = vel_data.std()

            angle_mean = angle_data.mean()
            angle_std = angle_data.std()
            print("right")
            print(vel_mean)
            print(vel_std)
            print(angle_mean * 180.0 / 3.141)
            print(angle_std * 180.0 / 3.141)
            breakpoint()


    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 7:9] = apply_randomization(self.root_states[:, 7:9], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 9] = apply_randomization(self.root_states[:, 9], self.cfg["randomization"].get("kick_lin_vel_z"))
            self.root_states[:, 10:13] = apply_randomization(self.root_states[:, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, 1] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, 1]),
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

        forces_all = torch.cat([
            self.pushing_forces.view(-1, 3),
            self.pushing_forces_ball.view(-1, 3)
        ], dim=0)

        torques_all = torch.cat([
            self.pushing_torques.view(-1, 3),
            self.pushing_torques_ball.view(-1, 3)
        ], dim=0)

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces_all),
            gymtorch.unwrap_tensor(torques_all),
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

        height_over_ground = feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos)
        height_over_ground = height_over_ground.view(
            self.num_envs,
            2,
            4
        )
        self.feet_height_over_ground[:] = height_over_ground.min(dim=2).values

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_ball[:] = False
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.terminate_counter += torch.where((self.terminate_counter > 0) | (self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]), 1, 0)
        self.reset_buf |= self.terminate_counter > self.cfg["rewards"]["terminate_time"] / self.dt
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)

        kicked_threshold = to_torch(self.cfg["rewards"]["ball_parameters"]["time_after_kick"], device=self.device).unsqueeze(0).expand(self.num_envs).clone()
        kicked_threshold[self.heavy_ball] *= 3
        self.reset_ball |= self.ball_kicked_counter >= kicked_threshold

        # Randomly reset ball with a tiny probability
        dist1, dist2 = self._helper_ball_sole_distance()
        min_dist = torch.minimum(dist1, dist2)

        # First get mask of which ball would even reset
        reset_ball_threshold = to_torch(self.cfg["commands"]["random_ball_reset"][0], device=self.device).unsqueeze(0).expand(self.num_envs).clone()
        reset_ball_threshold[self.flag_allow_deviation] = self.cfg["commands"]["random_ball_reset"][1]
        reset_ball_threshold[self.flag_allow_deviation & self.flag_is_strong_kick] = self.cfg["commands"]["random_ball_reset"][2]
        reset_ball_candidates = torch_rand_float(
                0, 1, (self.num_envs, 1), device=self.device
            ).squeeze(1) < reset_ball_threshold
        # Check extra conditions
        random_reset_mask = (
                                         reset_ball_candidates
                                      & (self.ball_reset_counter < self.cfg["rewards"]["episode_number_of_kicks"])
                                      & (
                                            ((self.ball_close_to_kick_pose < 1.0)
                                             & (min_dist < self.cfg["asset"]["feet_edge_pos"][0][0] + self.cfg["asset"]["ball"]["radius"]))
                                            | (self.ball_kicked_counter > 1)
                                        )
                                  )
        # Apply reset flag
        self.reset_ball[:] |= random_reset_mask

        random_distance_reset_candidates = (
                                              (self.ball_reset_counter < self.cfg["rewards"]["episode_number_of_kicks"])
                                            & (min_dist < self.cfg["commands"]["distance_reset"][1])
                                            & (min_dist > self.cfg["commands"]["distance_reset"][0])
                                           )
        random_distance_reset_id_candidates = random_distance_reset_candidates.nonzero(as_tuple=False).flatten()
        self.reset_ball[random_distance_reset_id_candidates] |= torch_rand_float(
                0, 1, (len(random_distance_reset_id_candidates), 1), device=self.device
            ).squeeze(1) < self.cfg["commands"]["random_ball_distance_reset"]

        self.ball_reset_counter[self.reset_ball] += torch.where(self.ball_kicked_counter[self.reset_ball] > 0, 1, 0)

        self.time_out_buf |= (self.ball_reset_counter > self.cfg["rewards"]["episode_number_of_kicks"]) & self.reset_ball
        self.reset_buf |= self.time_out_buf
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

        self.reset_ball[self.reset_buf[:]] = False

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0

        if "ball_kick_direction" in self.reward_names:
            self.ball_kick_direction_reward[:] = self._helper_ball_kick_direction()

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew

        '''sorted_rewards = sorted(
            self.extras["rew_terms"].items(),
            key=lambda x: x[1].mean().item(),
            reverse=True
        )
        print(sorted_rewards)
        print(self.rew_buf)
        print("---")'''

        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _update_odometrie_buffer(self):
        last_left_sole = self._helper_rotate_2d(self.last_feet_pos[:, 0, :2] - self.last_world_pose[:, :2], -self.last_world_pose[:, 2])
        last_right_sole = self._helper_rotate_2d(self.last_feet_pos[:, 1, :2] - self.last_world_pose[:, :2], -self.last_world_pose[:, 2])

        current_left_sole = self._helper_rotate_2d(self.feet_pos[:, 0, :2] - self.current_world_pose[:, :2], -self.current_world_pose[:, 2])
        current_right_sole = self._helper_rotate_2d(self.feet_pos[:, 1, :2] - self.current_world_pose[:, :2], -self.current_world_pose[:, 2])

        odo_trans = torch.where((self.feet_height_over_ground[:, 0] > self.feet_height_over_ground[:, 1]).unsqueeze(-1), (last_left_sole - last_right_sole) - (current_left_sole - current_right_sole), (current_right_sole - current_left_sole) - (last_right_sole - last_left_sole)) * 0.5

        odo_rot = self.current_world_pose[:, 2] - self.last_world_pose[:, 2]

        self.odo_buffer[:] = self.odo_buffer.roll(shifts=-1, dims=1)
        self.odo_buffer[:, -1] = torch.cat((torch.abs(odo_trans), torch.abs(odo_rot.unsqueeze(-1))), dim=-1)

    def _compute_observations(self):
        """Computes observations"""
        dummy_ball_pos = torch.clone(self.ball_pos)
        ball_noise = torch.zeros_like(dummy_ball_pos)
        ball_odo_noise = torch.zeros_like(dummy_ball_pos)

        ball_distance_x_ratio = ((torch.abs(self.ball_pos[:, 0]) - self.cfg["noise"]["ball_pos_noise_scale_range"][0]) / (self.cfg["noise"]["ball_pos_noise_scale_range"][1] - self.cfg["noise"]["ball_pos_noise_scale_range"][0])).clip(min=0.0, max=1.0)
        ball_distance_y_ratio = ((torch.abs(self.ball_pos[:, 1]) - self.cfg["noise"]["ball_pos_noise_scale_range"][0]) / (self.cfg["noise"]["ball_pos_noise_scale_range"][1] - self.cfg["noise"]["ball_pos_noise_scale_range"][0])).clip(min=0.0, max=1.0)

        # Copy, overwrite actual max noise value, set base (0) to 0
        self.ball_noise_range.zero_()
        scale_noise_x = self.cfg["noise"]["ball_pos_noise_x"]["range"]
        scale_noise_y = self.cfg["noise"]["ball_pos_noise_y"]["range"]
        self.ball_noise_range[:, 0, 1] = (scale_noise_x[1] - scale_noise_x[0]) * ball_distance_x_ratio + scale_noise_x[0]
        self.ball_noise_range[:, 1, 1] = (scale_noise_y[1] - scale_noise_y[0]) * ball_distance_y_ratio + scale_noise_y[0]
        self.ball_noise_range[:, 0, 0] = scale_noise_x[0]
        self.ball_noise_range[:, 1, 0] = scale_noise_y[0]

        ball_noise[:, :2] = apply_randomization(ball_noise[:, :2], self.cfg["noise"]["ball_pos_noise_dummy"]) * self.ball_noise_range[:, :, 1]
        ball_noise[:, :2] += (self.odo_noise_factor * self.odo_buffer.sum(dim=1)[:, :2]) # noise based on the last movement
        ball_noise[:, :2] += 0.2 * (self.odo_noise_factor[:,1] * self.odo_buffer.sum(dim=1)[:, 2]).unsqueeze(-1) # noise based on the last movement

        dummy_ball_pos[:, :3] += ball_noise[:, :3]
        dummy_last_ball_pos = torch.clone(self.last_relative_ball_pos)
        dummy_last_ball_pos[:, :2] += ball_noise[:, :2]

        noise_direction_angle = apply_randomization(self.direction_angle_current, self.cfg["noise"]["ball_direction_noise"])
        self.direction[:,0] = torch.sin(noise_direction_angle)
        self.direction[:,1] = torch.cos(noise_direction_angle)

        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"]["gravity"]) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"]["ang_vel"]) * self.cfg["normalization"]["ang_vel"],
                torch.cat(
                    (
                        dummy_ball_pos[:, :2] * self.cfg["normalization"]["ball_pos"],
                        (noise_direction_angle / torch.pi).unsqueeze(-1),
                    ),
                    dim=-1,
                ),
                (torch.cos(2 * torch.pi * self.gait_process) * (self.actions_gait > 1.0e-8).float()).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process) * (self.actions_gait > 1.0e-8).float()).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.dof_pos_offset - self.default_dof_pos, self.cfg["noise"]["dof_pos"]) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.custom_dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )

        expected_vel = 1.4142135623 * torch.sqrt(self.cfg["rewards"]["ball_parameters"]["ball_friction"] * self.range)
        self.obs_buf = torch.cat(
            (
                self.obs_buf,
                self.flag_is_strong_kick.float().unsqueeze(-1),
                self.flag_is_inaccurate_kick.unsqueeze(-1),
                self.flag_allow_deviation.float().unsqueeze(-1),
                apply_randomization(apply_randomization(self.ball_vel[:, 0], self.cfg["noise"]["ball_vel_noise"]), self.cfg["noise"]["ball_vel_scale_noise_x"]).unsqueeze(-1) * self.cfg["normalization"]["ball_vel"],
                apply_randomization(apply_randomization(self.ball_vel[:, 1], self.cfg["noise"]["ball_vel_noise"]), self.cfg["noise"]["ball_vel_scale_noise_y"]).unsqueeze(-1) * self.cfg["normalization"]["ball_vel"],
                self.zero_obs.unsqueeze(-1),
                self.direction,
                apply_randomization(expected_vel, self.cfg["noise"]["ball_range_noise"]).unsqueeze(-1) * self.cfg["normalization"]["ball_kick_range"],
                dummy_last_ball_pos * self.cfg["normalization"]["ball_pos"],
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
                self.inside_kick_pose_counter.unsqueeze(-1),
                self.ball_close_to_kick_pose.unsqueeze(-1),
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    def _helper_rotate_2d(self, pose_2d, yaw_rot):
        new_pose_2d = torch.zeros_like(pose_2d)
        new_pose_2d[:, 0] = torch.cos(yaw_rot) * pose_2d[:, 0] - torch.sin(yaw_rot) * pose_2d[:, 1]
        new_pose_2d[:, 1] = torch.sin(yaw_rot) * pose_2d[:, 0] + torch.cos(yaw_rot) * pose_2d[:, 1]
        return new_pose_2d

    # ------------ reward functions----------------
    def _reward_terminate(self):
        return (self.terminate_counter > 0).float()

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
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
        ratio = self.ones_obs.clone()
        ratio[self.ball_kicked_counter > 0] = 2
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1) * ratio

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
        return torch.sum((((self.default_dof_pos + self.actions[:,:self.num_actions-1]) < lower) | ((self.default_dof_pos + self.actions[:,:self.num_actions-1]) > upper)).float(), dim=-1)

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

    def _reward_torque_high(self):
        # Penalize torques above a given threshold
        return torch.sum((torch.abs(self.torques) - self.torque_high_limit).clip(min=0.0), dim=-1)
        
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
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

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
        return torch.clip(self.cfg["rewards"]["feet_distance_ref"] - feet_distance, min=0.0, max=0.1)

    def _reward_feet_swing(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.actions_gait > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.actions_gait > 1.0e-8)
        self.feet_swing_counter[:, 0] += (left_swing & ~self.feet_contact[:, 0]).float() + (~(left_swing & ~self.feet_contact[:, 0])).float() * -0.2
        self.feet_swing_counter[:, 1] += (right_swing & ~self.feet_contact[:, 1]).float() + (~(right_swing & ~self.feet_contact[:, 1])).float() * -0.2
        self.feet_swing_counter[:, 0] = torch.clamp(self.feet_swing_counter[:, 0], min=0.0)
        self.feet_swing_counter[:, 1] = torch.clamp(self.feet_swing_counter[:, 1], min=0.0)
        return (left_swing & ~self.feet_contact[:, 0] & (self.feet_swing_counter[:, 0] < 7.1)).float() + (right_swing & ~self.feet_contact[:, 1] & (self.feet_swing_counter[:, 1] < 7.1)).float()
        
    def _reward_ground_pressure(self):
        ground_pressure = (~self.last_feet_contact[:, 0] & self.feet_contact[:, 0]).float() * torch.clamp(self.feet_vel_filtered[:, 0, 2] + 0.2, max=0.0) + (~self.last_feet_contact[:, 1] & self.feet_contact[:, 1]).float() * torch.clamp(self.feet_vel_filtered[:, 1, 2] + 0.2, max=0.0)
        return -ground_pressure

    def _reward_gait_phase_factor(self):
        return torch.square(self.actions[:,-1] - torch.clamp(self.actions[:,-1], min=self.cfg["normalization"]["action_frequence_limit"][0], max=self.cfg["normalization"]["action_frequence_limit"][1]))

    def _reward_feet_height(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.actions_gait > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.actions_gait > 1.0e-8)
        left_error = (~left_swing).float() + left_swing.float() * torch.exp(-torch.square(torch.clip(self.feet_pos[:, 0, 2] - self.feet_pos[:, 1, 2] - self.cfg["rewards"]["feet_height_ref"], max=0.0)) / 0.0005)
        right_error = (~right_swing).float() + right_swing.float() * torch.exp(-torch.square(torch.clip(self.feet_pos[:, 1, 2] - self.feet_pos[:, 0, 2] - self.cfg["rewards"]["feet_height_ref"], max=0.0)) / 0.0005)
        return 2 - left_error - right_error
        
    def _reward_waist(self):
        if not self.cfg["algorithm"]["use_waist"]:
            print("Waist is not supported! Remove reward function!")
            raise Exception("Waist is not supported! Remove reward function!")
        return torch.square(self.dof_pos[:, 0]) + torch.square(self.actions_raw[:, 0])

    def _reward_waist_action(self):
        if not self.cfg["algorithm"]["use_waist"]:
            print("Waist is not supported! Remove reward function!")
            raise Exception("Waist is not supported! Remove reward function!")
        return torch.abs(self.last_actions[:,0] - self.actions[:,0])

##### Ball Reward ####

    def _helper_walk_target(self):
        # Determine kick pose
        ball_kick_pose_in_robot = quat_rotate_inverse(self.base_quat_z[:], self.ball_kick_pose[:, :] - self.base_pos[:, :3])

        extra_offset = ((self.kick_pose_translation_ratio - 0.5) / 0.5).clip(min=0.0) * (self.cfg["asset"]["ball"]["radius"] + 0.1)
        x_offset = -torch.sin(self.direction_angle_current[:]) * (self.cfg["rewards"]["ball_parameters"]["kick_pose_y_off"] + extra_offset)
        y_offset = torch.cos(self.direction_angle_current[:]) * (self.cfg["rewards"]["ball_parameters"]["kick_pose_y_off"] + extra_offset)

        ball_in_robot_left = torch.clone(ball_kick_pose_in_robot[:, :])
        ball_in_robot_right = torch.clone(ball_kick_pose_in_robot[:, :])
        ball_in_robot_left[:, 0] += x_offset
        ball_in_robot_left[:, 1] += y_offset
        ball_in_robot_right[:, 0] -= x_offset
        ball_in_robot_right[:, 1] -= y_offset

        walk_target = torch.where((self.filtered_ball_pos[:, 1] < 0).unsqueeze(-1), ball_in_robot_left, ball_in_robot_right)

        # Determine ball avoidance pose
        global_direction = self.robot_direction_ref[:] + self.direction_angle[:] * torch.pi
        direction_quat = quat_from_euler_xyz(
            torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            global_direction,
        )

        ball_in_robot = quat_rotate_inverse(direction_quat, self.ball_root_states[:, :3] - self.base_pos[:, :3])
        ratio = torch.where(self.ball_close_to_kick_pose < 1, 0, (-ball_in_robot[:, 0] / 0.3).clip(min=0.0, max=1.0))
        ball_in_robot[:, 0] += (torch.abs(ball_in_robot[:, 1]) / 0.3).clip(min=0.0, max=1.0) * -0.2 + (torch.max(-ball_in_robot[:, 0], 1.0 - (torch.abs(ball_in_robot[:, 1]) / 0.3))).clip(min=0.0, max=1.0) * 0.3
        ball_in_robot[:, 1] += torch.where(self.filtered_ball_pos[:, 1] < 0, 1, -1) * self.cfg["rewards"]["ball_parameters"]["ball_avoidance_range"]

        side_target = quat_rotate_inverse(self.base_quat_z[:], quat_rotate(direction_quat, ball_in_robot))
        return walk_target * (1.0 - ratio).unsqueeze(-1) + side_target * ratio.unsqueeze(-1)

    def _helper_ball_walk_angle_diff(self):
        walk_target = self._helper_walk_target()
        target_angle = torch.atan2(walk_target[:, 1], walk_target[:, 0])
        walk_angle = torch.atan2(self.filtered_lin_vel[:, 1], self.filtered_lin_vel[:, 0])
        return (target_angle - walk_angle + torch.pi) % (2 * torch.pi) - torch.pi

    def _helper_ball_walk_towards_ball(self):
        angle_diff = self._helper_ball_walk_angle_diff()
        sigma = self.cfg["rewards"]["ball_parameters"]["ball_walk_direction_sigma"] #torch.where(self.more_harsh_walk_speed_condition > 0.01, 0.2, self.cfg["rewards"]["ball_parameters"]["ball_walk_direction_sigma"])
        reward_far = (torch.exp(-torch.square(angle_diff) / sigma) - 0.5) * 2.0
        reward_far = torch.where(reward_far < 0, -1, reward_far)
        return reward_far

    def _helper_ball_vision_cone(self):
        ball_in_robot = quat_rotate_inverse(self.base_quat_z[:], self.ball_root_states[:, :3] - self.base_pos[:, :3])
        return torch.exp(-torch.square(torch.atan2(ball_in_robot[:,1], ball_in_robot[:,0])) / self.cfg["rewards"]["ball_parameters"]["ball_walk_vision_sigma"])

    def _reward_ball_vision_cone(self):
        cfg = self.cfg["rewards"]["ball_parameters"]
        reward = (self._helper_ball_vision_cone() - 0.5 ) * 2.0
        return reward

    def _helper_ball_kick_direction(self):
        cfg = self.cfg["rewards"]["ball_parameters"]
        clip_factor = torch.where(self.ball_kicked_counter - cfg["ball_kicked_start_counter"] > cfg["ball_kicked_clipped_counter"], 0, 1)
        time_ratio = 1.0 - torch.clip((self.ball_kicked_counter - cfg["ball_kicked_start_counter"]) / cfg["ball_kicked_max_counter"], min=0.0, max=1.0)
        ratio = torch.where(self.ball_kicked_counter > 0, 1, 0)
        angle_diff = (self.direction_angle_current[:] - torch.atan2(self.ball_vel[:,1], self.ball_vel[:,0]) + torch.pi) % (2 * torch.pi) - torch.pi
        still_counter_ratio = (self.ball_still_counter / 25).clip(min=0.0, max=1.0)

        kick_direction_sigma_range = to_torch(cfg["kick_direction_sigma_range"], device=self.device).unsqueeze(0).expand(self.num_envs).clone()
        kick_direction_sigma_range[self.flag_is_inaccurate_kick > 0.5] = cfg["kick_direction_sigma_just_hit"]

        kick_pose_counter_factor = torch.square(self.inside_kick_pose_counter)
        sole_yaw_factor = (self.kick_sole_yaw_ratio * 0.5 + 0.5)
        kick_pose_counter_factor[self.flag_allow_deviation] = 1
        sole_yaw_factor[self.flag_allow_deviation] = 1

        if self.cfg["rewards"]["ball_parameters"]["disable_sole_yaw_direction_penalty"]:
            sole_yaw_factor[:] = 1

        reward = ((torch.exp(-torch.square(angle_diff) / kick_direction_sigma_range)) - 0.5) * 2.0 * ratio * time_ratio * clip_factor

        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        height_ratio = ((base_height - cfg["base_height_scaling"][0]) / (cfg["base_height_scaling"][1] - cfg["base_height_scaling"][0])).clip(min=0.0, max=1.0)

        ball_vel_negative_scale = 1.0 - 0.9 * self.ball_vel_before_kick.clip(max=2.0) / 2.0

        negative_scale = 0.1 + 0.9 * ((self.request_changed - 0.25 / self.dt) / (0.5 / self.dt)).clip(min=0.0, max=1.0)
        #negative_scale = 1
        reward[:] = torch.where(reward < 0, reward * cfg["kick_direction_negative_factor"] * ball_vel_negative_scale * negative_scale, reward * height_ratio * kick_pose_counter_factor * sole_yaw_factor)
        return reward

    def _reward_ball_kick_direction(self):
        return self.ball_kick_direction_reward

    def _reward_ball_kick_velocity(self):
        cfg = self.cfg["rewards"]["ball_parameters"]
        # Calculation from B-Human BallPhysics
        ballFriction = self.cfg["rewards"]["ball_parameters"]["ball_friction"]
        expected_vel = 1.4142135623 * torch.sqrt(ballFriction * self.range[:])
        vel_diff = expected_vel - self.ball_velocity_norm_at_kick
        reward = torch.exp(-torch.square(vel_diff) / self.cfg["rewards"]["ball_parameters"]["ball_velocity_range_sigma_scale"]) * torch.where(self.ball_kicked_counter[:] > self.cfg["rewards"]["ball_parameters"]["ball_kicked_start_counter"], 1.0, 0)
        reward[:] = torch.where(self.ball_kick_direction_reward < 0, self.ball_kick_direction_reward * 0.1 * self.ball_velocity_norm_at_kick, reward * self.ball_kick_direction_reward)

        return reward

    def _reward_ball_kick_velocity_strong(self):
        cfg = self.cfg["rewards"]["ball_parameters"]
        reward = self.ball_velocity_norm_at_kick * torch.where(self.ball_kick_direction_reward < 0, self.ball_kick_direction_reward * 0.1, self.ball_kick_direction_reward)
        reward[~self.flag_is_strong_kick] = 0
        return reward

    def _reward_ball_walk_speed(self):
        # when kicked, no positive reward allowed
        allow_positive_reward = torch.where(self.ball_kick_direction_reward < 0, 0, 1)

        # Get current speeds
        lin_vel_clipped = self.filtered_lin_vel[:, :3].clone()
        lin_vel_clipped[:, 2] = self.filtered_ang_vel[:, 2]
        # Scale down to 0 from 3.5-4.0 radian. Scale it negative above
        too_much_rotation_ratio = 1.0 - ((torch.abs(lin_vel_clipped[:, 2]) - 4.0) / 0.5).clip(min=0.0)

        # Get factor to scale speeds based on ball vision cone and whether robot is rotating into the correct direction
        ball_vision_factor = self._reward_ball_vision_cone()
        min_rotation_negative_factor = torch.min(-((torch.abs(lin_vel_clipped[:, 2]) - 0.3).clip(min=0.0, max=1.0)),ball_vision_factor.clip(max=0.0))
        ball_vision_factor_scaling = 1.0 - ((ball_vision_factor + 0.5) / 0.25).clip(min=1.0, max=0.0)
        max_rotation_positive_factor = ball_vision_factor_scaling + (1.0 - ball_vision_factor_scaling) * (0.25 + 0.75 * (torch.norm(self.ball_pos, dim=-1) - 0.4 / 0.3).clip(min=0.0, max=1.0))
        rotation_sign_factor = torch.where(lin_vel_clipped[:, 2] * self.ball_pos[:, 1] >= 0.0, allow_positive_reward * max_rotation_positive_factor, min_rotation_negative_factor)

        # When walking in the wrong direction, punish walk speed
        walk_direction_factor = self._helper_ball_walk_towards_ball()
        walk_direction_factor[:] *= torch.where(walk_direction_factor < 0, (ball_vision_factor_scaling * 3.0 + 1.0) * self.cfg["rewards"]["ball_parameters"]["negative_walk_speed_factor"], 1)
        walk_direction_factor[:] = torch.where(self.ball_close_to_kick_pose < 1, walk_direction_factor.clip(min=-1.0), walk_direction_factor * allow_positive_reward)

        # Apply scales
        lin_vel_clipped[:, 0] = torch.abs(lin_vel_clipped[:, 0])
        lin_vel_clipped[:, 1] = torch.abs(lin_vel_clipped[:, 1])
        lin_vel_clipped[:, 2] = torch.abs(lin_vel_clipped[:, 2]) * rotation_sign_factor
        # If rotation is already negative, keep current value
        lin_vel_clipped[:, 2] *= torch.where(lin_vel_clipped[:, 2] < 0, 1.0, too_much_rotation_ratio)

        # Don't allow both soles near the ball
        #lin_vel_clipped[:, :] *= (1.0 - self._reward_ball_single_feet_avoidance()).unsqueeze(-1)
        # Once kick pose shifts into ball, clip reward for forward velocites. Policy shall focus on kick and not walk
        ball_distance = torch.norm(self.ball_pos[:, :2], dim=-1)
        ratio = torch.where((self.ball_close_to_kick_pose < 1.0) & (ball_distance < 0.6), 0.0, 1.0)
        min_speed = self.cfg["rewards"]["ball_parameters"]["clip_max_speed_near_ball"]
        lin_vel_clipped[:, 0] = torch.where(walk_direction_factor < 0, lin_vel_clipped[:, 0], torch.abs(lin_vel_clipped[:, 0]).clip(max=min_speed * (1.0 - ratio) + 3 * ratio))
        lin_vel_clipped[:, 1] = torch.where(walk_direction_factor < 0, lin_vel_clipped[:, 1], torch.abs(lin_vel_clipped[:, 1]).clip(max=min_speed * (1.0 - ratio) + 3 * ratio))
        # Only clip positive rotation reward values. Negatives are also penalized and should not get penalized
        lin_vel_clipped[:, 2] = torch.where(lin_vel_clipped[:, 2] < 0, lin_vel_clipped[:, 2], torch.abs(lin_vel_clipped[:, 2]).clip(max=3 * ((ratio - 0.5) * 2).clip(min=0.0)))

        # Get final reward
        reward = (torch.norm(lin_vel_clipped[:, :2], dim=-1) * walk_direction_factor + lin_vel_clipped[:, 2])

        return reward

    def _helper_ball_sole_distance(self):
        feet_pos = torch.clone(self.body_states[:, self.feet_indices, 0:3])
        feet_quat = torch.clone(self.body_states[:, self.feet_indices, 3:7])
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)

        feet_edge_pos = (expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))).reshape(self.num_envs, 2, feet_edge_relative_pos.shape[2], 3)
        ball_distances = torch.norm(
            feet_edge_pos[..., :2] - self.ball_root_states[:, :2].unsqueeze(1).unsqueeze(2),
            dim=-1
        )

        left_distance, _ = torch.min(ball_distances[:, 0, :], dim=-1)
        right_distance, _ = torch.min(ball_distances[:, 1, :], dim=-1)

        return left_distance, right_distance

    def _reward_ball_walk_target_overshoot(self):
        return (self.ball_walk_target_overshoot.clip(min=0.0) / 10.0) * (1.0 - ((self.ball_velocity_norm - 0.3) / 0.7).clip(min=0.0, max=1.0)) * (1.0 - (self.ball_kicked_counter > 0).float())

    def _reward_ball_sole_yaw(self):
        kick_pose_counter_factor = torch.square(self.inside_kick_pose_counter)
        kick_pose_counter_factor[self.flag_allow_deviation] = 1
        return kick_pose_counter_factor * self.kick_sole_yaw_ratio * torch.where(self.ball_kick_direction_reward > 0, 1, 0)

    def is_inside_ankle_polygon(
        self,
        pitch: torch.Tensor,
        roll: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pitch: (Num_Envs, 1)
            roll:  (Num_Envs, 1)

        Returns:
            inside: (Num_Envs, 1) bool
        """

        point = torch.cat((pitch.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)  # (N, 2)

        p1 = self.ankle_polygon
        p2 = torch.roll(self.ankle_polygon, shifts=-1, dims=0)

        edge = p2 - p1
        rel = point.unsqueeze(1) - p1.unsqueeze(0)

        # 2D cross product: edge x rel
        cross = (
            edge[:, 0].unsqueeze(0) * rel[:, :, 1]
            - edge[:, 1].unsqueeze(0) * rel[:, :, 0]
        )

        # Polygon is clock-wise:
        # Point must be right side of every edge.
        inside = (cross <= 0).all(dim=1, keepdim=True)

        return inside.squeeze(-1)

    def ankle_polygon_intersection(
        self,
        pitch: torch.Tensor,
        roll: torch.Tensor,
    ):
        """
        Calculates the intersection of

            (x, y) = t * (pitch, roll), t >= 0

        with the polygon.

        Args:
            pitch: (Num_Envs, 1)
            roll:  (Num_Envs, 1)

        Returns:
            intersection_pitch: (Num_Envs, 1)
            intersection_roll:  (Num_Envs, 1)
        """

        direction = torch.cat((pitch.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)  # (N, 2)

        p1 = self.ankle_polygon
        p2 = torch.roll(self.ankle_polygon, shifts=-1, dims=0)

        edge = p2 - p1

        # ---------------------------------------------------------
        # Intersection:
        #
        # t * direction = p1 + u * edge
        #
        # t = cross(p1, edge) / cross(direction, edge)
        # ---------------------------------------------------------

        direction = direction.unsqueeze(1)  # (N, 1, 2)
        p1 = p1.unsqueeze(0)                # (1, 6, 2)
        edge = edge.unsqueeze(0)            # (1, 6, 2)

        def cross2d(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        denominator = cross2d(direction, edge)
        numerator = cross2d(p1, edge)

        eps = torch.finfo(pitch.dtype).eps

        valid_denominator = torch.abs(denominator) > eps

        denominator_safe = torch.where(
            valid_denominator,
            denominator,
            torch.ones_like(denominator),
        )

        t = numerator / denominator_safe
        u = cross2d(p1, direction) / denominator_safe

        valid = (
            valid_denominator
            & (t >= 0.0)
            & (u >= 0.0)
            & (u <= 1.0)
        )

        t = torch.where(
            valid,
            t,
            torch.full_like(t, float("inf")),
        )

        t_min = t.min(dim=1, keepdim=True).values

        intersection = direction[:, 0, :] * t_min

        return (
            intersection[:, 0:1].squeeze(-1),
            intersection[:, 1:2].squeeze(-1),
        )

    def _reward_ankle_not_allowed(self):
        envs_left_clip = ~self.is_inside_ankle_polygon(self.dof_pos[:, -8], self.dof_pos[:, -7])
        envs_right_clip = ~self.is_inside_ankle_polygon(self.dof_pos[:, -2], self.dof_pos[:, -1])

        reward = self.zero_obs.clone()
        if envs_left_clip.any():
            left_new_pitch, left_new_roll = self.ankle_polygon_intersection(self.dof_pos[envs_left_clip, -8], self.dof_pos[envs_left_clip, -7])
            reward[envs_left_clip] += torch.abs(left_new_pitch - self.dof_pos[envs_left_clip, -8]) + torch.abs(left_new_roll - self.dof_pos[envs_left_clip, -7])
        if envs_right_clip.any():
            right_new_pitch, right_new_roll = self.ankle_polygon_intersection(self.dof_pos[envs_right_clip, -2], self.dof_pos[envs_right_clip, -1])
            reward[envs_right_clip] += torch.abs(right_new_pitch - self.dof_pos[envs_right_clip, -2]) + torch.abs(right_new_roll - self.dof_pos[envs_right_clip, -1])

        return reward

