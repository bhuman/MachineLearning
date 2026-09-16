import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio
from envs import *
import torch
import torch.nn.functional as F
from utils.model import *
from utils.buffer import ExperienceBuffer
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder


class Runner_History:

    def __init__(self, test=False):
        self.test = test
        # prepare the environment
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)

        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.model = ActorCritic(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._load()

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs*2, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        if not self.test:
            self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self.optimizer.load_state_dict(model_dict["optimizer"])
        except Exception as e:
            print(f"Failed to load optimizer: {e}")

    def train(self):
        self.recorder = Recorder(self.cfg)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        for it in range(self.cfg["basic"]["max_iterations"]):
            # within horizon_length, env.step() is called with same act
            for n in range(self.cfg["runner"]["horizon_length"]):
                if self.cfg["algorithm"]["learn_ball"]:
                    mirror_obs = self._mirror_obs_batch_ball(obs)
                else:
                    mirror_obs = self._mirror_obs_batch_walk(obs)
                mirror_privileged_obs = self._mirror_priv_obs(privileged_obs)

                self.buffer.update_data("obses", n, torch.cat((obs, mirror_obs), dim=0))
                self.buffer.update_data("privileged_obses", n, torch.cat((privileged_obs, mirror_privileged_obs), dim=0))
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.sample()
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                
                mirror_act = self._mirror_action_batch(act)
                self.buffer.update_data("actions", n, torch.cat((act, mirror_act), dim=0))
                self.buffer.update_data("rewards", n, torch.cat((rew, rew), dim=0))
                self.buffer.update_data("dones", n, torch.cat((done, done), dim=0))
                self.buffer.update_data("time_outs", n, torch.cat((infos["time_outs"].to(self.device), infos["time_outs"].to(self.device)), dim=0))
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

            with torch.no_grad():
                old_dist = self.model.act(self.buffer["obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)
                old_actions_log_prob[:, self.env.num_envs:] = old_actions_log_prob[:, :self.env.num_envs]

            mean_value_loss = 0
            mean_actor_loss = 0
            mean_bound_loss = 0
            mean_entropy = 0
            mean_sym_loss = 0
            
            if self.cfg["algorithm"]["learn_ball"]:
                mirror_obs = self._mirror_obs_batch_ball(obs)
            else:
                mirror_obs = self._mirror_obs_batch_walk(obs)
            mirror_privileged_obs = self._mirror_priv_obs(privileged_obs)
            
            cat_obs = torch.cat((obs, mirror_obs), dim=0)
            cat_priv_obs = torch.cat((privileged_obs, mirror_privileged_obs), dim=0)
            
            last_values = self.model.est_value(cat_obs, cat_priv_obs)

            for n in range(self.cfg["runner"]["mini_epochs"]):
                values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                with torch.no_grad():
                    self.buffer["rewards"][self.buffer["time_outs"]] = values[self.buffer["time_outs"]]
                    advantages = discount_values(
                        self.buffer["rewards"],
                        self.buffer["dones"] | self.buffer["time_outs"],
                        values,
                        last_values,
                        self.cfg["algorithm"]["gamma"],
                        self.cfg["algorithm"]["lam"],
                    )
                    returns = values + advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                value_loss = F.mse_loss(values, returns)

                dist = self.model.act(self.buffer["obses"])

                actions_log_prob = dist.log_prob(self.buffer["actions"]).sum(dim=-1)
                min_log_prob = torch.min(actions_log_prob[:, :self.env.num_envs]).clip(max=-20)
                max_log_prob = torch.max(actions_log_prob[:, :self.env.num_envs]).clip(min=20)
                actions_log_prob = torch.clamp(actions_log_prob, min=min_log_prob, max=max_log_prob)
                actor_loss = surrogate_loss(old_actions_log_prob, actions_log_prob, advantages, e_clip=self.cfg["algorithm"]["e_clip"])
                bound_loss = torch.clip(dist.loc - self.cfg["normalization"]["clip_actions"], min=0.0).square().mean() + torch.clip(dist.loc + self.cfg["normalization"]["clip_actions"], max=0.0).square().mean()

                sym_loss = F.mse_loss(
                    dist.loc[:, :self.env.num_envs],
                    self._mirror_action_batch(dist.loc[:, self.env.num_envs:])
                )

                entropy = dist.entropy().sum(dim=-1)

                # make sure the entropy is not too small
                min_entropy = self.cfg["algorithm"]["entropy_range"]["min"]
                max_entropy = self.cfg["algorithm"]["entropy_range"]["max"]
                loss_entropy = torch.mean((torch.clamp(entropy.mean(), min=min_entropy, max=max_entropy) - entropy.mean())**2)
                loss = (
                    value_loss
                    + actor_loss
                    + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                    + 0.01 * loss_entropy
                    + 0.01 * sym_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                if torch.isnan(self.model.logstd).any():
                    self.model.reset_logstd()
                    print("reset logstd")

                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(dist.scale / old_dist.scale)
                        + 0.5 * (torch.square(old_dist.scale) + torch.square(dist.loc - old_dist.loc)) / torch.square(dist.scale)
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                        self.learning_rate = max(float(self.cfg["algorithm"]["learning_rate_range"]["min"]), self.learning_rate / 1.5)
                    elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                        self.learning_rate = min(float(self.cfg["algorithm"]["learning_rate_range"]["max"]), self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_bound_loss += bound_loss.item()
                mean_entropy += entropy.mean()
                mean_sym_loss += sym_loss.mean()
            mean_value_loss /= self.cfg["runner"]["mini_epochs"]
            mean_actor_loss /= self.cfg["runner"]["mini_epochs"]
            mean_bound_loss /= self.cfg["runner"]["mini_epochs"]
            mean_sym_loss /= self.cfg["runner"]["mini_epochs"]
            mean_entropy /= self.cfg["runner"]["mini_epochs"]
            self.recorder.record_statistics(
                {
                    "value_loss": mean_value_loss,
                    "actor_loss": mean_actor_loss,
                    "bound_loss": mean_bound_loss,
                    "sym_loss": mean_sym_loss,
                    "entropy": mean_entropy,
                    "kl_mean": kl_mean,
                    "lr": self.learning_rate,
                },
                it,
            )

            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    it + 1,
                )
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))

    def _mirror_obs_batch_walk(self, obs_batch):
        """
        obs_batch: tensor shape (..., obs_dim)
        returns mirrored_obs_batch with same shape
          [ projected_gravity(3),
            base_ang_vel(3),
            commands(3),
            gait_cos(1),
            gait_sin(1),
            dof_pos_diff(12 [+1 waist]),
            dof_vel(12 [+1 waist]),
            actions(12 legs + 1 extra [+1 waist]),
          ]
        """

        x = obs_batch
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])  # (N, obs_dim)


        # copy input
        out = flat.clone()

        for history_index in range(0, self.env.num_history):

            # sizes
            idx = history_index * self.env.num_base_obs
            pg_idx = slice(idx, idx+3); idx += 3
            bag_idx = slice(idx, idx+3); idx += 3

            if self.cfg["algorithm"]["use_waist"]:
                dof_pos_waist_idx = idx; idx += 1
            dof_pos_idx = slice(idx, idx+12); idx += 12

            if self.cfg["algorithm"]["use_waist"]:
                action_waist_idx = idx; idx += 1
            actions_idx = slice(idx, idx+12); idx += 12  # 12 legs

            # 1) projected_gravity: mirror y (index 1)
            out[:, pg_idx] = flat[:, pg_idx] * torch.tensor([1., -1., 1.], device=flat.device)

            # 2) base_ang_vel: mirror x and z
            out[:, bag_idx] = flat[:, bag_idx] * torch.tensor([-1., 1., -1.], device=flat.device)

            # 5) dof_pos_diff
            dp = flat[:, dof_pos_idx].reshape(-1, 2, 6)
            dp_swapped = dp[:, [1, 0], :]
            flip_mask = torch.ones_like(dp_swapped)
            flip_mask[:, :, [1, 2, 5]] = -1.0
            out[:, dof_pos_idx] = (dp_swapped * flip_mask).reshape(-1, 12)

            # 5.1) waist
            if self.cfg["algorithm"]["use_waist"]:
                out[:, dof_pos_waist_idx] = -flat[:, dof_pos_waist_idx]

            # 7) actions (legs + extra)
            acts = flat[:, actions_idx]              # (N,13)
            legs = acts[:, :].reshape(-1, 2, 6)
            legs_swapped = legs[:, [1, 0], :]
            out[:, actions_idx] = (legs_swapped * flip_mask).reshape(-1, 12)

            # 7.1) waist
            if self.cfg["algorithm"]["use_waist"]:
                out[:, action_waist_idx] = -flat[:, action_waist_idx]

        # Most recent information
        # sizes
        idx = self.env.num_base_obs * self.env.num_history
        #pg_idx = slice(idx, idx+3); idx += 3
        #bag_idx = slice(idx, idx+3); idx += 3
        cmd_idx = slice(idx, idx+3); idx += 3
        gait_cos_idx = idx; idx += 1
        gait_sin_idx = idx; idx += 1
        
        #if self.cfg["algorithm"]["use_waist"]:
        #    dof_pos_waist_idx = idx; idx += 1
        #dof_pos_idx = slice(idx, idx+12); idx += 12
        
        if self.cfg["algorithm"]["use_waist"]:
            dof_vel_waist_idx = idx; idx += 1
        dof_vel_idx = slice(idx, idx+12); idx += 12
        
        #if self.cfg["algorithm"]["use_waist"]:
        #    action_waist_idx = idx; idx += 1
        #actions_idx = slice(idx, idx+12); idx += 12  # 12 legs

        # 1) projected_gravity: mirror y (index 1)
        #out[:, pg_idx] = flat[:, pg_idx] * torch.tensor([1., -1., 1.], device=flat.device)

        # 2) base_ang_vel: mirror x and z
        #out[:, bag_idx] = flat[:, bag_idx] * torch.tensor([-1., 1., -1.], device=flat.device)

        # 3) commands: mirror sideways (y) and yaw (z)
        out[:, cmd_idx] = flat[:, cmd_idx] * torch.tensor([1., -1., -1.], device=flat.device)

        # 4) gait cos/sin -> shift by π == flip signs
        out[:, gait_cos_idx] = -flat[:, gait_cos_idx]
        out[:, gait_sin_idx] = -flat[:, gait_sin_idx]

        # 5) dof_pos_diff
        #dp = flat[:, dof_pos_idx].reshape(-1, 2, 6)
        #dp_swapped = dp[:, [1, 0], :]
        #flip_mask = torch.ones_like(dp_swapped)
        #flip_mask[:, :, [1, 2, 5]] = -1.0
        #out[:, dof_pos_idx] = (dp_swapped * flip_mask).reshape(-1, 12)

        # 5.1) waist
        #if self.cfg["algorithm"]["use_waist"]:
        #    out[:, dof_pos_waist_idx] = -flat[:, dof_pos_waist_idx]

        # 6) dof_vel
        dv = flat[:, dof_vel_idx].reshape(-1, 2, 6)
        dv_swapped = dv[:, [1, 0], :]
        flip_mask = torch.ones_like(dv_swapped)
        flip_mask[:, :, [1, 2, 5]] = -1.0
        out[:, dof_vel_idx] = (dv_swapped * flip_mask).reshape(-1, 12)

        # 6.1) waist
        if self.cfg["algorithm"]["use_waist"]:
            out[:, dof_vel_waist_idx] = -flat[:, dof_vel_waist_idx]

        # 7) actions
        #acts = flat[:, actions_idx]              # (N,12)
        #legs = acts[:, :].reshape(-1, 2, 6)
        #legs_swapped = legs[:, [1, 0], :]
        #mirrored_legs = (legs_swapped * flip_mask).reshape(-1, 12)
        #out[:, actions_idx] = torch.cat([mirrored_legs], dim=-1)

        # 7.1) waist
        #if self.cfg["algorithm"]["use_waist"]:
        #    out[:, action_waist_idx] = -flat[:, action_waist_idx]

        return out.reshape(orig_shape).clone()

    def _mirror_obs_batch_ball(self, obs_batch):
        """
        obs_batch: tensor shape (..., obs_dim)
        returns mirrored_obs_batch with same shape
          [ projected_gravity(3),
            base_ang_vel(3),
            ball_3d(3),
            gait_cos(1),
            gait_sin(1),
            dof_pos_diff(12 [+1 waist]),
            dof_vel(12 [+1 waist]),
            actions(12 legs + 1 extra [+1 waist]),
            flags(3),
            ball_vel(3),
            direction(2), sin, cos
            range(1),
            ball_old_x,
            ball_old_y,
            ...
          ]
        """

        x = obs_batch
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])  # (N, obs_dim)

        # copy input
        out = flat.clone()

        for history_index in range(0, self.env.num_history):
            # sizes
            idx = history_index * self.env.num_base_obs
            pg_idx = slice(idx, idx+3); idx += 3
            bag_idx = slice(idx, idx+3); idx += 3
            
            if self.cfg["algorithm"]["use_waist"]:
                dof_pos_waist_idx = idx; idx += 1
            dof_pos_idx = slice(idx, idx+12); idx += 12
            
            if self.cfg["algorithm"]["use_waist"]:
                action_waist_idx = idx; idx += 1
            actions_idx = slice(idx, idx+12); idx += 12  # 12 legs

            ball_2d_idx = slice(idx, idx+2); idx += 2

            # 1) projected_gravity: mirror y (index 1)
            out[:, pg_idx] = flat[:, pg_idx] * torch.tensor([1., -1., 1.], device=flat.device)

            # 2) base_ang_vel: mirror x and z
            out[:, bag_idx] = flat[:, bag_idx] * torch.tensor([-1., 1., -1.], device=flat.device)

            # 5) dof_pos_diff
            dp = flat[:, dof_pos_idx].reshape(-1, 2, 6)
            dp_swapped = dp[:, [1, 0], :]
            flip_mask = torch.ones_like(dp_swapped)
            flip_mask[:, :, [1, 2, 5]] = -1.0
            out[:, dof_pos_idx] = (dp_swapped * flip_mask).reshape(-1, 12)

            # 5.1) waist
            if self.cfg["algorithm"]["use_waist"]:
                out[:, dof_pos_waist_idx] = -flat[:, dof_pos_waist_idx]

            # 7) actions (legs)
            acts = flat[:, actions_idx]              # (N,12)
            legs = acts[:, :].reshape(-1, 2, 6)
            legs_swapped = legs[:, [1, 0], :]
            out[:, actions_idx] = (legs_swapped * flip_mask).reshape(-1, 12)

            # 7.1) waist
            if self.cfg["algorithm"]["use_waist"]:
                out[:, action_waist_idx] = -flat[:, action_waist_idx]

            # 3) ball 2d pos: mirror sideways (y)
            out[:, ball_2d_idx] = flat[:, ball_2d_idx] * torch.tensor([1., -1.], device=flat.device)

        # Most recent information
        idx = self.env.num_base_obs * self.env.num_history
        #pg_idx = slice(idx, idx+3); idx += 3
        #bag_idx = slice(idx, idx+3); idx += 3
        dummy_cmd = slice(idx, idx+3); idx += 3
        gait_cos_idx = idx; idx += 1
        gait_sin_idx = idx; idx += 1
        
        #if self.cfg["algorithm"]["use_waist"]:
        #    dof_pos_waist_idx = idx; idx += 1
        #dof_pos_idx = slice(idx, idx+12); idx += 12
        
        if self.cfg["algorithm"]["use_waist"]:
            dof_vel_waist_idx = idx; idx += 1
        dof_vel_idx = slice(idx, idx+12); idx += 12
        
        #if self.cfg["algorithm"]["use_waist"]:
        #    action_waist_idx = idx; idx += 1
        #actions_idx = slice(idx, idx+13); idx += 13  # 12 legs + 1 extra
        dummy_phase = idx; idx += 1

        #ball_2d_idx = slice(idx, idx+2); idx += 2
        flags = slice(idx, idx+3); idx += 3
        #ball_vel = slice(idx, idx+3); idx += 3
        direction_sin = slice(idx, idx+1); idx += 1
        direction_cos = slice(idx, idx+1); idx += 1
        ball_range = slice(idx, idx+1); idx += 1

        # 1) projected_gravity: mirror y (index 1)
        #out[:, pg_idx] = flat[:, pg_idx] * torch.tensor([1., -1., 1.], device=flat.device)

        # 2) base_ang_vel: mirror x and z
        #out[:, bag_idx] = flat[:, bag_idx] * torch.tensor([-1., 1., -1.], device=flat.device)

        # 3) ball 3d pos: mirror sideways (y) and yaw (z)
        #out[:, ball_3d_idx] = flat[:, ball_3d_idx] * torch.tensor([1., -1., -1.], device=flat.device)

        # 4) gait cos/sin -> shift by π == flip signs
        out[:, gait_cos_idx] = -flat[:, gait_cos_idx]
        out[:, gait_sin_idx] = -flat[:, gait_sin_idx]

        # 5) dof_pos_diff
        #dp = flat[:, dof_pos_idx].reshape(-1, 2, 6)
        #dp_swapped = dp[:, [1, 0], :]
        #flip_mask = torch.ones_like(dp_swapped)
        #flip_mask[:, :, [1, 2, 5]] = -1.0
        #out[:, dof_pos_idx] = (dp_swapped * flip_mask).reshape(-1, 12)

        # 5.1) waist
        #if self.cfg["algorithm"]["use_waist"]:
        #    out[:, dof_pos_waist_idx] = -flat[:, dof_pos_waist_idx]

        # 6) dof_vel
        dv = flat[:, dof_vel_idx].reshape(-1, 2, 6)
        dv_swapped = dv[:, [1, 0], :]
        flip_mask = torch.ones_like(dv_swapped)
        flip_mask[:, :, [1, 2, 5]] = -1.0
        out[:, dof_vel_idx] = (dv_swapped * flip_mask).reshape(-1, 12)

        # 6.1) waist
        if self.cfg["algorithm"]["use_waist"]:
            out[:, dof_vel_waist_idx] = -flat[:, dof_vel_waist_idx]

        # 7) actions (legs + extra)
        #acts = flat[:, actions_idx]              # (N,13)
        #legs = acts[:, :12].reshape(-1, 2, 6)
        #extra = acts[:, 12:13].clone()             # (N,1)
        #legs_swapped = legs[:, [1, 0], :]
        #mirrored_legs = (legs_swapped * flip_mask).reshape(-1, 12)
        #out[:, actions_idx] = torch.cat([mirrored_legs, extra], dim=-1)

        # 7.1) waist
        #if self.cfg["algorithm"]["use_waist"]:
        #    out[:, action_waist_idx] = -flat[:, action_waist_idx]

        # 8) Flip ball_vel and direction
        
        # 3) ball 3d pos: mirror sideways (y) and yaw (z)
        #out[:, ball_2d_idx] = flat[:, ball_2d_idx] * torch.tensor([1., -1.], device=flat.device)
        #out[:, flags] = flat[:, flags]
        #out[:, ball_vel] = flat[:, ball_vel] * torch.tensor([1., -1., 1.], device=flat.device)
        raw_angle = torch.asin(flat[:, direction_sin])
        out[:, direction_sin] = torch.sin(-raw_angle)
        out[:, direction_cos] = torch.cos(-raw_angle)
        #out[:, ball_old_2d] = flat[:, ball_old_2d] * torch.tensor([1., -1.], device=flat.device)

        return out.reshape(orig_shape).clone()

    def _mirror_priv_obs(self, obs_batch):
        """
        obs_batch: tensor shape (..., obs_dim)
        returns mirrored_obs_batch with same shape
          [ base_mass_scaled(4),
            base_lin_vel(3),
            height(1),
            pushing_force(3),
            pushing_torque(3),
            ...,
          ]
        """
        x = obs_batch
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])  # (N, obs_dim)

        # copy input
        out = flat.clone()

        # sizes
        idx = 0

        bms_idx = slice(idx, idx+4); idx += 4
        blv_idx = slice(idx, idx+3); idx += 3
        height_idx = slice(idx, idx+1); idx += 1
        pf_idx = slice(idx, idx+3); idx += 3
        pt_idx = slice(idx, idx+3); idx += 3


        # 1) Base Mass
        out[:, bms_idx] = flat[:, bms_idx] * torch.tensor([1., -1., 1., 1.], device=flat.device)

        # 2) Base Lin Vel
        out[:, blv_idx] = flat[:, blv_idx] * torch.tensor([1., -1., 1.], device=flat.device)

        # 3) Skip Height

        # 4) Pushing Force
        out[:, pf_idx] = flat[:, pf_idx] * torch.tensor([1., -1., 1.], device=flat.device)

        # 5) Pushing Torque
        out[:, pt_idx] = flat[:, pt_idx] * torch.tensor([-1., 1., -1.], device=flat.device)

        return out.reshape(orig_shape).clone()

    def _mirror_action_batch(self, action_batch):
        """
        Mirrors actions with 2 legs (6 DOFs each) + optional waist + 1 extra action at the end.
        Input: (..., 13|14)
        - [0]        : waist (optional, sign-flipped if present)
        - [0..5]     : left leg
        - [6..11]    : right leg
        - [12]       : extra action (unchanged)
        """
        orig_shape = action_batch.shape
        flat = action_batch.reshape(-1, orig_shape[-1])

        # --- optional waist ---
        start_idx = 0
        waist = None
        if self.cfg["algorithm"]["use_waist"]:
            start_idx = 1
            waist = -flat[:, :1].clone()   # immer neuer Tensor

        # --- Beine ---
        legs = flat[:, start_idx:start_idx+12].reshape(-1, 2, 6)   # (batch, 2, 6)

        # swap left/right
        legs_swapped = legs[:, [1, 0], :]

        # flip mask für roll/yaw/ankleRoll (1, 2, 5)
        flip_local = [1, 2, 5]
        flip_mask = torch.ones_like(legs_swapped)
        flip_mask[:, :, flip_local] = -1.0

        legs_flipped = legs_swapped * flip_mask
        mirrored_legs = legs_flipped.reshape(-1, 12)

        # --- Rest (extra action) ---
        extra = flat[:, start_idx+12:].clone()

        # --- final concat ---
        if self.cfg["algorithm"]["use_waist"]:
            out = torch.cat([waist, mirrored_legs, extra], dim=-1)
        else:
            out = torch.cat([mirrored_legs, extra], dim=-1)

        return out.reshape(orig_shape).clone()

    def play(self):
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        while True:
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.loc
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as self.writer:
                        for frame in self.env.camera_frames:
                            self.writer.append_data(frame)
                    if self.interrupt:
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True
