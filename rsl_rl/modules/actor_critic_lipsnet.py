import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

from .lipsnet import LipsNet

class ActorCriticLipsNet(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        ## actor-f 函数设计
                        actor_f_hid_dims = [512, 256, 128],
                        actor_f_hid_nonlinear = 'lrelu',
                        actor_f_out_nonlinear = 'identity',
                        ## actor-k 函数设计
                        actor_global_lips = False,
                        actor_k_init = 10,
                        actor_k_hid_dims = [512, 256, 128],
                        actor_k_hid_nonlinear = 'tanh',
                        actor_k_out_nonlinear = 'softplus',
                        actor_eps = 1e-4,
                        actor_loss_lambda = 0.001,
                        actor_squash_action = False,
                        ## critic
                        critic_hidden_dims=[256, 256, 256],
                        critic_activation = 'lrelu',
                        ## others
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticLipsNet.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticLipsNet, self).__init__()

        # activation = get_activation(activation)
        # mlp_input_dim_a = num_actor_obs
        # mlp_input_dim_c = num_critic_obs

        # Policy
        # actor_layers = []
        # actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        # actor_layers.append(activation)
        # for l in range(len(actor_hidden_dims)):
        #     if l == len(actor_hidden_dims) - 1:
        #         actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
        #     else:
        #         actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
        #         actor_layers.append(activation)
        # self.actor = nn.Sequential(*actor_layers)

        ## policy network        
        self.actor = LipsNet(f_sizes = [num_actor_obs, *actor_f_hid_dims, num_actions],
                  f_hid_nonlinear = get_activation(actor_f_hid_nonlinear), 
                  f_out_nonlinear = get_activation(actor_f_out_nonlinear),
                  global_lips = actor_global_lips,
                  k_init = actor_k_init, 
                  k_sizes = [num_actor_obs, *actor_k_hid_dims], 
                  k_hid_nonlinear = get_activation(actor_k_hid_nonlinear), 
                  k_out_nonlinear = get_activation(actor_k_out_nonlinear),
                  loss_lambda = actor_loss_lambda, 
                  eps = actor_eps, 
                  squash_action = actor_squash_action)
        
        ## Value network
        critic_layers = []
        critic_activation = get_activation(critic_activation)
        # 第一层
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(critic_activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(critic_activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

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

    # def update_distribution(self, observations, get_k=False):
    #     actor_output = self.actor(observations, get_k=get_k)
    #     if get_k:
    #         mean, k_out = actor_output
    #     else:
    #         mean, k_out = actor_output, None
    #     self.distribution = Normal(mean, mean*0. + self.std)
    #     return mean, k_out  # 原本没有返回值

    # def act_old(self, observations, get_k=False, **kwargs):
    #     mean = self.update_distribution(observations)
    #     return self.distribution.sample()
    
    def act(self, observations, get_info=False, **kwargs):
        mean, k_out, jac_norm = self.actor(observations, get_info=True)
        self.distribution = Normal(mean, mean*0. + self.std)
        if get_info:
            return self.distribution.sample(), k_out, jac_norm
        else:
            return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, get_info=False):
        actions_mean, k_out, jac_norm = self.actor(observations, get_info=True)
        if get_info:
            return actions_mean, k_out, jac_norm
        else:
            return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    str_to_functions = {
        "elu" : nn.ELU(),
        "selu" : nn.SELU(),
        "relu" : nn.ReLU(),
        "lrelu" : nn.LeakyReLU(),
        "tanh" : nn.Tanh(),
        "sigmoid" : nn.Sigmoid(),
        "identity" : nn.Identity(),
        "softplus" : nn.Softplus(),
    }
    func = str_to_functions.get(act_name)
    if func == None:
        raise NotImplementedError(f"invalid activation function! name={act_name}")
    return func
    
    # if act_name == "elu":
    #     return nn.ELU()
    # elif act_name == "selu":
    #     return nn.SELU()
    # elif act_name == "relu":
    #     return nn.ReLU()
    # elif act_name == "crelu":
    #     assert NotImplementedError()
    #     return nn.ReLU()
    # elif act_name == "lrelu":
    #     return nn.LeakyReLU()
    # elif act_name == "tanh":
    #     return nn.Tanh()
    # elif act_name == "":
    #     return nn.Sigmoid()
    # elif act_name == "sigmoid":
    #     pass
    # else:
    #     print(f"[error] invalid activation function! name={act_name}")
    #     return None
