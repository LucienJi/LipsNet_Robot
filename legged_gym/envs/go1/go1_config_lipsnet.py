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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

from legged_gym.envs.base.base_config import BaseConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
# from functorch import jacrev, vmap

class Go1RoughCfgLipsNet( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    
    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0, 1, 0, 0, 0, 0, 0, 0]  # caozhanxiang提供的地形
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_wtw.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        
        #TODO: 这个会clip 负的reward
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100. # forces above this value are penalized
        class scales( LeggedRobotCfg.rewards.scales ):
            # torques = -0.0002
            torques = -0.0002  # -0.0002
            dof_pos_limits = -10.0
            
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0 # -2.0
            ang_vel_xy = -0.05 # -0.05
            orientation = -0.
            dof_vel = -0.
            dof_acc = -5e-7 # -2.5e-7
            base_height = -0. 
            feet_air_time = 1.0 # 1.0
            collision = -1. # -1.0
            feet_stumble = -0.0 
            action_rate = 0.0# -0.01 # -0.01 #TODO: 暂时删除action震荡的penalty
            stand_still = -0.
    
    class commands:
        curriculum = False
        max_curriculum = 3.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-2., 2.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        
class Go1RoughCfgPPOLipsNet(BaseConfig): # 不继承之前的训练配置，防止污染
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy: #( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        # actor_hidden_dims = [512, 256, 128] #TODO: 注意修改网络结构
        # critic_hidden_dims = [512, 256, 128]
        # activation = 'lrelu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        ## actor-f 函数设计
        actor_f_hid_dims = [512, 256, 128]
        actor_f_hid_nonlinear = 'lrelu'
        actor_f_out_nonlinear = 'identity'
        ## actor-k 函数设计
        actor_global_lips = False
        actor_k_init = 50
        actor_k_hid_dims = [512, 256, 128, 1]
        actor_k_hid_nonlinear = 'tanh'
        actor_k_out_nonlinear = 'softplus'
        actor_eps = 1e-4
        actor_loss_lambda = None
        actor_squash_action = False
        ## critic 函数设计
        critic_hidden_dims = [512, 256, 128]
        critic_activation = 'lrelu'
        
        
    class algorithm: #( LeggedRobotCfgPPO.algorithm ):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        #TODO: 重要 k_out norm loss coef 
        lips_loss_coef = 1e-5 #1e-5
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        # 调整学习率
        # learning_rate = 1.e-3 #5.e-4
        learning_rate = None
        learning_rate_actor_f = 1.e-3
        learning_rate_actor_k = 1.e-5
        learning_rate_critic = 1.e-3
        #TODO: schedule 会控制learning rate的变化
        schedule = 'adaptive' # could be adaptive, fixed 
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner: #( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticLipsNet'
        algorithm_class_name = 'PPOLipsNet'
        num_steps_per_env = 24 # per iteration
        max_iterations = 5000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'go1_lipsnet'
        run_name = ''
        
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

  