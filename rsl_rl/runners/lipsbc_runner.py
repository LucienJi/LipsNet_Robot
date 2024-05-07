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

import time
import os
from collections import deque
import statistics
from legged_gym.utils.helpers import class_to_dict,NumpyEncoder
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms.lipsbc_ppo import PPOPriLipsNet
from rsl_rl.modules.actor_critic_pri_lipsnet_v2 import ActorCriticPriLipsNet
from rsl_rl.env import VecEnv
import json 

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,  # train所需的所有参数都在这里
                 log_dir=None,
                 device='cpu'):
        self.train_cfg = train_cfg
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs # critic 可以使用与actor不同的obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic= ActorCriticPriLipsNet( self.env.num_obs,
                                                        self.env.num_obs_history,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        self.alg= PPOPriLipsNet(actor_critic, device=self.device,use_lips = self.policy_cfg['use_lips'], **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], 
                              [self.env.num_privileged_obs], [self.env.cfg.env.num_observation_history,self.env.num_obs], [self.env.num_actions])

        
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        self.save_cfg()
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len: #TODO:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        for k,v in obs_dict.items():
            obs_dict[k] = v.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.no_grad():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_dict['obs'], obs_dict['privileged_obs'], obs_dict['obs_history'],use_DAgger=it > self.cfg['start_update_student'])
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    for k,v in obs_dict.items():
                        obs_dict[k] = v.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_dict['obs'], obs_dict['privileged_obs'],obs_dict['obs_history'])
            
            mean_value_loss, mean_surrogate_loss, mean_student_neglogp,mean_adaptation_loss,\
            mean_k_out, mean_jac_norm, mean_f_out, max_k_out, max_jac_norm, max_f_out, min_k_out, min_jac_norm ,min_f_out,mean_k_l2,mean_l2_coef= self.alg.update(
                                                                                        update_teacher=self.cfg['update_teacher'], 
                                                                                        update_student=it > self.cfg['start_update_student'])
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_student_neglogp', locs['mean_student_neglogp'], locs['it'])
        self.writer.add_scalar('Loss/Adaptation_module', locs['mean_adaptation_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_k_l2', locs['mean_k_l2'], locs['it'])
        self.writer.add_scalar('Loss/mean_k_out', locs['mean_k_out'], locs['it'])
        self.writer.add_scalar('Loss/max_k_out', locs['max_k_out'], locs['it'])
        self.writer.add_scalar('Loss/min_k_out', locs['min_k_out'], locs['it'])

        self.writer.add_scalar('Loss/mean_f_out', locs['mean_f_out'], locs['it'])
        self.writer.add_scalar('Loss/min_f_out', locs['min_f_out'], locs['it'])
        self.writer.add_scalar('Loss/max_f_out', locs['max_f_out'], locs['it'])

        self.writer.add_scalar('Loss/mean_jac_norm', locs['mean_jac_norm'], locs['it'])
        self.writer.add_scalar('Loss/max_jac_norm', locs['max_jac_norm'], locs['it'])
        self.writer.add_scalar('Loss/min_jac_norm', locs['min_jac_norm'], locs['it'])

        self.writer.add_scalar('Loss/L2_coef', locs['mean_l2_coef'], locs['it'])
        
        # self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        # self.writer.add_scalar('Loss/learning_rate_actor_f', self.alg.learning_rate_actor_f, locs['it'])
        # self.writer.add_scalar('Loss/learning_rate_actor_k', self.alg.learning_rate_actor_k, locs['it'])
        # self.writer.add_scalar('Loss/learning_rate_critic', self.alg.learning_rate_critic, locs['it'])
        
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Adaptation loss:':>{pad}} {locs['mean_adaptation_loss']:.4f}\n"""
                          f"""{'BC loss:':>{pad}} {locs['mean_student_neglogp']:.4f}\n"""
                          f"""{'mean_k_l2:':>{pad}} {locs['mean_k_l2']:.4f}\n"""
                          f"""{'mean_k_out:':>{pad}} {locs['mean_k_out']:.4f}\n"""
                          f"""{'max_k_out:':>{pad}} {locs['max_k_out']:.4f}\n"""
                          f"""{'min_k_out:':>{pad}} {locs['min_k_out']:.4f}\n"""
                          f"""{'mean_jac_norm:':>{pad}} {locs['mean_jac_norm']:.4f}\n"""
                          f"""{'min_jac_norm:':>{pad}} {locs['min_jac_norm']:.4f}\n"""
                          f"""{'max_jac_norm:':>{pad}} {locs['max_jac_norm']:.4f}\n"""
                          f"""{'mean_f_out:':>{pad}} {locs['mean_f_out']:.4f}\n"""
                          f"""{'max_f_out:':>{pad}} {locs['max_f_out']:.4f}\n"""
                          f"""{'min_f_out:':>{pad}} {locs['min_f_out']:.4f}\n"""
                          f"""{'L2 Coef:':>{pad}} {locs['mean_l2_coef']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Adaptation loss:':>{pad}} {locs['mean_adaptation_loss']:.4f}\n"""
                          f"""{'BC loss:':>{pad}} {locs['mean_student_neglogp']:.4f}\n"""
                          f"""{'mean_k_l2:':>{pad}} {locs['mean_k_l2']:.4f}\n"""
                          f"""{'mean_k_out:':>{pad}} {locs['mean_k_out']:.4f}\n"""
                          f"""{'max_k_out:':>{pad}} {locs['max_k_out']:.4f}\n"""
                          f"""{'min_k_outs:':>{pad}} {locs['min_k_out']:.4f}\n"""
                          f"""{'mean_jac_norm:':>{pad}} {locs['mean_jac_norm']:.4f}\n"""
                          f"""{'min_jac_norm:':>{pad}} {locs['min_jac_norm']:.4f}\n"""
                          f"""{'max_jac_norm:':>{pad}} {locs['max_jac_norm']:.4f}\n"""
                          f"""{'mean_f_out:':>{pad}} {locs['mean_f_out']:.4f}\n"""
                          f"""{'max_f_out:':>{pad}} {locs['max_f_out']:.4f}\n"""
                          f"""{'min_f_out:':>{pad}} {locs['min_f_out']:.4f}\n"""
                          f"""{'L2 Coef:':>{pad}} {locs['mean_l2_coef']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'teacher_optimizer_state_dict': self.alg.teacher_optimizer.state_dict(),
            'student_optimizer_state_dict': self.alg.student_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.teacher_optimizer.load_state_dict(loaded_dict['teacher_optimizer_state_dict'])
            self.alg.student_optimizer.load_state_dict(loaded_dict['student_optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def load_expert(self,path):
        loaded_dict = torch.load(path)
        for k,v in loaded_dict['model_state_dict'].items():
            loaded_dict['model_state_dict'][k] = v.to(self.device)
        self.alg.actor_critic.actor_teacher.load_state_dict(loaded_dict['model_state_dict'],strict=False)
        self.alg.actor_critic.critic.load_state_dict(loaded_dict['model_state_dict'],strict=False)
        self.alg.actor_critic.std.data = loaded_dict['model_state_dict']['std']
        print("###### Teacher loaded ######")
        return loaded_dict['infos']
    
    def save_cfg(self):
        # 保存训练使用的cfg参数
        train_cfg_dict = class_to_dict(self.train_cfg) if type(self.train_cfg) is not dict else self.train_cfg
        train_cfg_json = json.dumps(train_cfg_dict, sort_keys=False, indent=4, separators=(',',':'),cls=NumpyEncoder)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        with open(os.path.join(self.log_dir, "train_cfg.json"), 'w') as f:
            f.write(train_cfg_json)
        
        # 保存环境使用的 cfg 参数
        env_cfg_dict = class_to_dict(self.env.cfg) if type(self.env.cfg) is not dict else self.env.cfg
        env_cfg_json = json.dumps(env_cfg_dict, sort_keys=False, indent=4, separators=(',',':'),cls=NumpyEncoder)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        with open(os.path.join(self.log_dir, "env_cfg.json"), 'w') as f:
            f.write(env_cfg_json)
    

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
