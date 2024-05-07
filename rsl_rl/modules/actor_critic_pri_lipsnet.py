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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.utils.utils import init_orhtogonal
from .lipsnet import LipsNet

class ActorCriticPriLipsNet(nn.Module):
    is_recurrent = False
    def __init__(self, 
                 num_obs,
                 num_obs_history,
                 num_privileged_obs,
                 num_actions,
                 activation = 'elu',
                 actor_hidden_dims = [512, 256, 128],
                 critic_hidden_dims = [512, 256, 128],
                 adaptation_module_branch_hidden_dims = [256, 128],
                 latent_dim = 16,
                 ## actor-f 函数设计
                 actor_f_hid_dims = [512, 256, 128],
                 actor_f_hid_nonlinear = 'lrelu',
                 actor_f_out_nonlinear = 'identity',
                 ## actor-k 函数设计
                 k_lips = False,
                 actor_global_lips = False,
                 actor_multi_k = True, 
                 actor_k_init = 10,
                 actor_k_hid_dims = [512, 256, 128],
                 actor_k_hid_nonlinear = 'tanh',
                 actor_k_out_nonlinear = 'softplus',
                 actor_eps = 1e-4,
                 actor_squash_action = False,
                 ## others
                 use_lips = True,
                 init_noise_std = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs
        self.num_actions = num_actions 
        self.use_lips = use_lips
        activation = get_activation(activation)
        # Teacher Policy
        ## EnvEncoder
        teacher_adaptation_input = self.num_privileged_obs + self.num_obs 
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(teacher_adaptation_input, adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],latent_dim)) #! latent_dim
            else:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.teacher_adaptation_module = nn.Sequential(*adaptation_module_layers)

        mlp_teacher_input_a = num_obs + latent_dim
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_teacher_input_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_teacher = nn.Sequential(*actor_layers)

        # Student Policy

        mlp_student_input_a = num_obs + self.num_obs_history
        if use_lips:
            if actor_multi_k:
                actor_k_hid_dims += [num_actions]
            else:
                actor_k_hid_dims += [1]
            self.actor_student = LipsNet(f_sizes = [mlp_student_input_a, *actor_f_hid_dims, num_actions],
                  f_hid_nonlinear = get_activation(actor_f_hid_nonlinear), 
                  f_out_nonlinear = get_activation(actor_f_out_nonlinear),
                  global_lips = actor_global_lips,
                  k_init = actor_k_init, 
                  k_sizes = [mlp_student_input_a, *actor_k_hid_dims], 
                  k_hid_nonlinear = get_activation(actor_k_hid_nonlinear), 
                  k_out_nonlinear = get_activation(actor_k_out_nonlinear),
                  eps = actor_eps, 
                  squash_action = actor_squash_action,
                  k_lips=k_lips)
        else:
            actor_layers = []
            actor_layers.append(nn.Linear(mlp_student_input_a, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            self.actor_student = nn.Sequential(*actor_layers)

        # Value function
        mlp_input_dim_c = num_obs + num_privileged_obs 
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.student_std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # self.teacher_adaptation_module.apply(init_orhtogonal)
        # self.actor_student.apply(init_orhtogonal)
        # self.actor_teacher.apply(init_orhtogonal)
        # self.critic.apply(init_orhtogonal)

        print("actor_student: ", self.actor_student)
        print("actor_teacher: ", self.actor_teacher)
        print("critic: ", self.critic)


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        print(f"[warning] neural networks use custom weight initialization.")
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def act_student(self, obs, privileged_obs, obs_history,get_info = False):
        # obs_dict: obs, obs_history, privileged_obs
        obs_history = obs_history.reshape(-1, self.num_obs_history)
        student_input = torch.cat((obs, obs_history), dim=-1)
        if self.use_lips:
            action_mean,k_out, jac_norm, f_out = self.actor_student.forward(student_input,get_info = True)
            self.distribution = Normal(action_mean, action_mean*0. + self.student_std)
            if get_info:
                return self.distribution.sample(), k_out, jac_norm,f_out
            else:
                return self.distribution.sample()
        else:
            action_mean = self.actor_student.forward(student_input)
            self.distribution = Normal(action_mean, action_mean*0. + self.student_std)
            return self.distribution.sample()

    def act_teacher(self, obs, privileged_obs , obs_history):
        # obs_dict: obs, obs_history, privileged_obs
        latent = self.get_teacher_latent(obs,privileged_obs)
        teacher_input = torch.cat((obs, latent), dim=-1)
        action_mean = self.actor_teacher.forward(teacher_input)
        self.distribution = Normal(action_mean, action_mean*0. + self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict:dict, **kwargs):
        obs,pri_obs = obs_dict['obs'], obs_dict['privileged_obs']
        obs_history = obs_dict['obs_history'].reshape(-1, self.num_obs_history)

        if kwargs.get('use_teacher',False):
            latent = self.get_teacher_latent(obs,pri_obs)
            teacher_input = torch.cat((obs, latent), dim=-1)
            actions_mean = self.actor_teacher.forward(teacher_input)
        else:
            student_input = torch.cat((obs, obs_history), dim=-1)
            actions_mean = self.actor_student.forward(student_input)
        return actions_mean
    
    def get_lips_info(self,obs_dict:dict):
        obs,pri_obs = obs_dict['obs'], obs_dict['privileged_obs']
        obs_history = obs_dict['obs_history'].reshape(-1, self.num_obs_history)
        student_input = torch.cat((obs, obs_history), dim=-1)
        actions_mean, k_out, jac_norm, f_out = self.actor_student.forward(student_input,get_info=True)
        return k_out, jac_norm, f_out
        

    def evaluate(self, critic_observations,privileged_obs,obs_history):
        input = torch.cat((critic_observations, privileged_obs), dim=-1)
        value = self.critic(input)
        return value
    def get_teacher_latent(self,obs,privileged_obs):
        input = torch.cat((obs, privileged_obs), dim=-1)
        teacher_latent = self.teacher_adaptation_module(input)
        return teacher_latent

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == 'softplus':
        return nn.Softplus()
    elif act_name == 'identity':
        return nn.Identity()
    else:
        print("invalid activation function!")
        return None
