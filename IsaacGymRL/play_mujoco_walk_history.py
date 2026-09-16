import os
import sys
import glob
import yaml
import select
import argparse
import numpy as np
import torch
import mujoco, mujoco.viewer
from utils.model import *

import time

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_last_observations"]+cfg["env"]["num_base_observations"]*cfg["env"]["history_size"], cfg["env"]["num_privileged_obs"])
    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(model_dict["model"])

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    
    num_of_unknown_joints = 0 # unknown joints are assumed to come first followed by all known joints
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            if cfg["control"]["allow_unknown_joints"]:
                dof_stiffness[i] = cfg["control"]["stiffness"]["Default"]
                dof_damping[i] = cfg["control"]["damping"]["Default"]
                num_of_unknown_joints += 1
            else:
                raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")

    num_of_unknown_joints = 0 # unknown joints are assumed to come first followed by all known joints
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
                break
        if not found:
            if cfg["control"]["allow_unknown_joints"]:
                dof_stiffness[i] = cfg["control"]["stiffness"]["Default"]
                dof_damping[i] = cfg["control"]["damping"]["Default"]
                num_of_unknown_joints += 1
            else:
                raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")

    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_pos = mj_data.qpos.astype(np.float32)[7:]
    pref_dof_pos = np.zeros(dof_pos.shape, dtype=np.float32)
    pref_dof_pos[:] = dof_pos
    filtered_vel = np.zeros(dof_pos.shape, dtype=np.float32)
    joint_range = mj_model.jnt_range
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = gait_process = 0.0
    lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
    it = 0
    waist_shift = 1 if cfg["algorithm"]["use_waist"] else 0
    
    obs_history = np.zeros((10,cfg["env"]["num_base_observations"]), dtype=np.float32)
    obs_all = np.zeros((cfg["env"]["num_last_observations"]+cfg["env"]["num_base_observations"]*cfg["env"]["history_size"]), dtype=np.float32)
    with_arms = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print(f"Set command (x, y, yaw): ")
        
        try:
            key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "upper_body_init")
            dof_targets[:num_of_unknown_joints] = mj_model.key_qpos[key_id][7:7+num_of_unknown_joints]
        except:
            print("No Initial Values Provided")

        while viewer.is_running():
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                            gait_frequency = 0
                        else:
                            gait_frequency = np.average(cfg["commands"]["gait_frequency"])
                        print(
                            f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw): ",
                            end="",
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numeric values.\nSet command (x, y, yaw): ", end="")

            dof_pos = mj_data.qpos.astype(np.float32)[7:]
            dof_vel = mj_data.qvel.astype(np.float32)[6:]
            quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            current_vel = (dof_pos[:] - pref_dof_pos[:]) / cfg["sim"]["dt"]
            filtered_vel[:] = current_vel * 0.22 + filtered_vel * 0.78
            if it % cfg["control"]["decimation"] == 0:
                obs_history[:] = np.roll(obs_history, shift=-1, axis=0)
                obs = obs_history[-1]
                obs[:] = 0
                obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
                obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
                obs[6:18+waist_shift] = (dof_pos[num_of_unknown_joints:] - default_dof_pos[num_of_unknown_joints:]) * cfg["normalization"]["dof_pos"]
                obs[18+waist_shift:30+2*waist_shift] = actions[:-1]

                obs_history_flat = torch.tensor(obs_history.flatten())
                obs_all[:] = 0
                history_length = obs_history_flat.shape[0]
                obs_all[:history_length] = obs_history_flat
                #obs_all[history_length+0:3+history_length] = projected_gravity * cfg["normalization"]["gravity"]
                #obs_all[3+history_length:6+history_length] = base_ang_vel * cfg["normalization"]["ang_vel"]
                obs_all[0+history_length] = lin_vel_x * cfg["normalization"]["lin_vel"]
                obs_all[1+history_length] = lin_vel_y * cfg["normalization"]["lin_vel"]
                obs_all[2+history_length] = ang_vel_yaw * cfg["normalization"]["ang_vel"]
                obs_all[3+history_length] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs_all[4+history_length] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                #obs_all[11+history_length:23+waist_shift+history_length] = (dof_pos[num_of_unknown_joints:] - default_dof_pos[num_of_unknown_joints:]) * cfg["normalization"]["dof_pos"]
                obs_all[5+history_length:17+1*waist_shift+history_length] = current_vel[num_of_unknown_joints:] * cfg["normalization"]["dof_vel"]
                obs_all[17+1*waist_shift+history_length] = actions[-1]
                idx = 18+1*waist_shift+history_length
                obs_all[idx:idx+6] = 0
                #obs_all[idx+8] = 0
                #obs_all[idx+9] = 0
                #obs_all[idx+10] = 0
                dist = model.act(torch.tensor(obs_all).unsqueeze(0))
                actions[:] = dist.loc.detach().numpy()
                actions[:] = np.clip(actions[:cfg["env"]["num_actions"]], -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                actions[-1] = np.clip(actions[-1], cfg["normalization"]["action_frequence_limit"][0], cfg["normalization"]["action_frequence_limit"][1])

                dof_targets[num_of_unknown_joints:] = default_dof_pos[num_of_unknown_joints:] + cfg["control"]["action_scale"] * actions[:cfg["env"]["num_actions"]-1]
                for j in range(0, cfg["env"]["num_actions"]-1):
                  dof_targets[num_of_unknown_joints+j] = np.clip(dof_targets[num_of_unknown_joints+j], joint_range[num_of_unknown_joints+j+1][0], joint_range[num_of_unknown_joints+j+1][1])

            gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * (actions[-1] + cfg["rewards"]["target_frequence"]), 1.0)
            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * filtered_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            pref_dof_pos[:] = mj_data.qpos.astype(np.float32)[7:]
            time.sleep(0.002)
            mujoco.mj_step(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
