# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)
    env.set_eval(False)
    env.reset()
    env.set_commands([0.5,0.0,0.0])
    obs_dict = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner.load("logs/LipsNet/Nov14_09-30-53_1e-5_L2/model_10000.pt")
    

    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    obs_list = []; action_list = []; k_out_list = []; f_out_list = []; jac_norm_list = []
    use_student = True
    # with torch.inference_mode():
    with torch.no_grad():
        for i in range(10*int(env.max_episode_length)):
            
            
            if use_student:
                actions = policy(obs_dict,use_student=use_student)
            else:
                actions = policy(obs_dict,use_student=use_student)
            
            
            obs_list.append(obs_dict['obs'].cpu().detach().numpy()); 
            action_list.append(actions.cpu().detach().numpy())

            # if use_student:
            #     k_out_list.append(k_out.cpu().detach().numpy()[robot_index, joint_index]); 
            #     f_out_list.append(f_out.cpu().detach().numpy()[robot_index, joint_index]); 
            #     jac_norm_list.append(jac_norm.cpu().detach().numpy()[robot_index])

            if i == env.max_episode_length:
                break
            else:
                print(f"{i}/{env.max_episode_length}")
            obs_dict, rewards, dones, infos = env.step(actions.detach())

            if RECORD_FRAMES:
                filepath = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported')
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                filename = os.path.join(filepath, "frames{:03}.png".format(img_idx))
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 

            # if RECORD_FRAMES:
            #     if i % 2:
            #         filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
            #         env.gym.write_viewer_image_to_file(env.viewer, filename)
            #         img_idx += 1 
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    }
                )
                # if use_student:
                #     logger.log_lips(
                #         {
                #             'k_out': k_out[robot_index, joint_index].item(),
                #             'f_out': f_out[robot_index, joint_index].item(),
                #             'jac_norm': jac_norm[robot_index].item(),
                #             'action': actions[robot_index, joint_index].item()
                #         }
                #     )
            elif i==stop_state_log:
                logger.plot_states()
                if use_student:
                    logger.plot_lips()
            if  0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()
        
    # action_arr = np.concatenate(action_list)
    # obs_arr = np.concatenate(obs_list)
    # if use_student:
    #     k_out_arr = np.concatenate(k_out_list)
    #     f_out_arr = np.concatenate(f_out_list)
    #     jac_norm_arr = np.concatenate(jac_norm_list)
    # # 绘制图像
    # import matplotlib.pyplot as plt
    # idx = np.arange(action_arr.shape[0])
    # plt.plot(idx, action_arr)
    # plt.title("action")
    # plt.show()
    # if use_student:
    #     plt.plot(idx, k_out_arr, label="k_out")
    #     plt.plot(idx, f_out_arr, label="f_out")
    #     plt.title("k_out")
    #     plt.legend()
    #     plt.show()
    
    import pdb; pdb.set_trace()
    
    
if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    args = get_args()
    play(args)
