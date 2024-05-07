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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rsl_rl.modules.actor_critic_pri_lipsnet_v2 import ActorCriticPriLipsNet
from rsl_rl.storage import RolloutPriStorage
from .lag import Lagrange

class PPO:
    actor_critic: ActorCriticPriLipsNet
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',                 
                 learning_rate_critic = 1.e-3,
                 learning_rate_student = 1.e-3,
                 adaptation_module_learning_rate = 1.e-3,
                 DAgger_coef = 1.0,
                 ):

        self.device = device
        self.DAgger_coef = DAgger_coef
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.learning_rate_critic = learning_rate_critic
        self.learning_rate_teacher = learning_rate
        self.learning_rate_student = learning_rate_student
        
        self.adaptation_module_learning_rate = adaptation_module_learning_rate 
        # self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # 设置特殊的optimizer
        self.teacher_optimizer = torch.optim.Adam([
                # {'params':self.actor_critic.teacher_adaptation_module.parameters(), 'lr':'inf', 'lr_name':'learning_rate_teacher',},
                {'params':self.actor_critic.actor_teacher.parameters(), 'lr':'inf', 'lr_name':'learning_rate_teacher',},
                {'params':self.actor_critic.critic.parameters(), 'lr':'inf', 'lr_name':'learning_rate_critic',},
                {'params':self.actor_critic.std, 'lr':'inf', 'lr_name':"learning_rate_teacher",  },
            ])
    
        self.student_optimizer = torch.optim.Adam(
                [ 
                    {'params':self.actor_critic.actor_student.parameters(), 'lr':'inf', 'lr_name':'learning_rate_student',},
                    {'params':self.actor_critic.student_std, 'lr':'inf', 'lr_name':"learning_rate_student",},
                ]
            )

        self.update_learning_rate(self.teacher_optimizer)
        self.update_learning_rate(self.student_optimizer)

        for param_group in self.teacher_optimizer.param_groups:
            print(param_group['lr_name'], param_group['lr'])
        print()
        for param_group in self.student_optimizer.param_groups:
            print(param_group['lr_name'], param_group['lr'])
        self.transition = RolloutPriStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        # 设置lipsnet的loss
        
        
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_history_shape, action_shape):
        self.storage = RolloutPriStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.eval()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history, use_DAgger = True):
        # Compute the actions and values
        # use expert policy with probability of self.DAgger_coef
        if use_DAgger:
            use_expert = torch.rand(1).item() < self.DAgger_coef
        else:
            use_expert = True
        if not use_expert:
            self.transition.actions = self.actor_critic.act_student(obs,privileged_obs,obs_history,get_info=False)
        else:
            self.transition.actions = self.actor_critic.act_teacher(obs,privileged_obs,obs_history).detach()
        self.transition.values = self.actor_critic.evaluate( obs, privileged_obs,obs_history).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.obs_historys = obs_history
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs,privileged_obs,obs_history):
        last_values= self.actor_critic.evaluate(last_critic_obs,privileged_obs,obs_history).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update_learning_rate(self,optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = getattr(self, param_group['lr_name'])
    
    def update(self, update_teacher = True,update_student=True):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_student_neglogp = 0
        mean_adaptation_loss = 0 

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, pri_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                if update_teacher:
                    #! RMA 的训练流程  
                    self.actor_critic.act_teacher(obs_batch,pri_obs_batch,obs_history_batch)  
                    actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                    value_batch = self.actor_critic.evaluate(obs_batch, pri_obs_batch,obs_history_batch)
                    mu_batch = self.actor_critic.action_mean
                    sigma_batch = self.actor_critic.action_std
                    entropy_batch = self.actor_critic.entropy
                    # KL
                    if self.desired_kl != None and self.schedule == 'adaptive':
                        with torch.inference_mode():
                            kl = torch.sum(
                                torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                            kl_mean = torch.mean(kl)

                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate_teacher = max(1e-5, self.learning_rate_teacher / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate_teacher = min(1e-2, self.learning_rate_teacher * 1.5)
                    # Surrogate loss
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                    1.0 + self.clip_param)
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                    # Value function loss
                    if self.use_clipped_value_loss:
                        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                        self.clip_param)
                        value_losses = (value_batch - returns_batch).pow(2)
                        value_losses_clipped = (value_clipped - returns_batch).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns_batch - value_batch).pow(2).mean()

                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                    self.teacher_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.teacher_optimizer.step()   
                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()

                if update_student:
                    #! behavior cloning
                    with torch.no_grad():
                        self.actor_critic.act_teacher(obs_batch,pri_obs_batch,obs_history_batch)
                        target_action = self.actor_critic.action_mean.detach() 
                    self.actor_critic.act_student(obs_batch,pri_obs_batch,obs_history_batch )
                    student_actions_log_prob_batch = self.actor_critic.get_actions_log_prob(target_action)
                    log_prob = student_actions_log_prob_batch.mean()
                    loss = -log_prob
                    self.student_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.student_optimizer.step()
                    mean_student_neglogp += loss.item()


        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        mean_adaptation_loss /= num_updates

        mean_student_neglogp /= num_updates
        
    
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_student_neglogp,mean_adaptation_loss
