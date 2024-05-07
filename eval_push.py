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
from legged_gym.utils import  export_policy_as_jit, task_registry, Logger, class_to_dict
from legged_gym.utils.helpers import parse_sim_params
from isaacgym import gymutil
from legged_gym.envs.base.history_wrapper import HistoryWrapper
# from legged_gym.envs.base.noise_wrapper import NoisyWrapper 
from legged_gym.envs.base.push_wrapper import PushConfig, EvalWrapper
from legged_gym.envs.go1.go1_eval_config import Go1Eval
import numpy as np
from legged_gym.utils.helpers import update_cfg_from_args
import torch
import argparse 

def get_eval_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "littledog_CPG", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--play", "action": "store_true", "default": False, "help": "Play learned policy and record frames"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        
        #! For Eval
        {"name": "--model_type",'type':str, "default": "expert"},
        {"name": "--model_path",'type':str, "default": "logs/Expert/Dec06_14-15-22_PushBaseline/model_10000.pt"},
        {"name": "--eval_name",'type':str, "default": "Vel05"},
        {"name": "--cmd_vel",'type':float, "default": 0.5},
        {"name": "--use_teacher",'type':bool, "default": False},
        {"name": "--eval_path", 'type':str, "default": "logs/Eval/ood/3body/v10"},
        {"name": "--push_force", 'type':float, "default": 20},
        {"name": "--random_push", 'type':bool, "default": 'True'},
        {"name": "--push_interval", 'type':int, "default": 100},
        {"name": "--push_duration", 'type':int, "default": 25},

    ]
    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    # args = gymutil.parse_arguments(
    #     description="RL Policy",
    #     custom_parameters=custom_parameters)

    args  =gymutil.parse_custom_arguments(
        parser=parser,
        custom_parameters=custom_parameters
    )

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def eval_push(args,
                     model_type,
                     model_path,
                    push_force = 20, 
                    random_push = True,
                    push_interval = 100,
                    push_duration = 25,
                    target_vel = [0.5,0.0,0.0],
                    use_teacher = False,
                    eval_path = "logs/Eval",
                    eval_name = "eval"):
    env_cfg = Go1Eval()
    if model_type == "Expert":
        from configs.config_expert import Go1RoughCfgPPO 
        from rsl_rl.runners.expert_runner import OnPolicyRunner as Runner
        train_cfg = Go1RoughCfgPPO()
        
    elif model_type == "LipsNet":
        from configs.config_BCLIPS import Go1RoughCfgPPOPriLipsNet
        from rsl_rl.runners.lipsbc_runner import OnPolicyRunner as Runner
        train_cfg = Go1RoughCfgPPOPriLipsNet()
    elif model_type == "BCMLP":
        from configs.config_BCMLP import Go1RoughCfgPPOMLPBC
        from rsl_rl.runners.mlpbc_runner import OnPolicyRunner as Runner
        train_cfg = Go1RoughCfgPPOMLPBC()
    elif model_type == "RMA":
        from configs.config_rma import Go1RoughCfgPPO
        from rsl_rl.runners.rma_runner import OnPolicyRunner as Runner
        train_cfg = Go1RoughCfgPPO()
    
    else:
        raise NotImplementedError
    
    # env_cfg.env.num_envs = 100
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False

    # prepare environment
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)

    env_cfg,_  = update_cfg_from_args(env_cfg,None,args)
    sim_params = {"sim":class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    env = LeggedRobot(sim_params=sim_params,
                                    physics_engine=args.physics_engine,
                                    sim_device=args.sim_device,
                                    headless=args.headless, 
                                    cfg = env_cfg)
    env.set_eval(True)
    env.set_commands(target_vel)
    env = HistoryWrapper(env)
    push_env = EvalWrapper(env,env_cfg,cmd_vel=target_vel,async_mode=True,
                             record = False,move_camera=False,experiment_name="eval")
    
    push_config = PushConfig(
        id = 0,
        body_index_list=[0],
        push_interval=push_interval,
        push_duration=push_duration,
        force_list=[push_force],
    )
    push_env.set_eval_config([push_config])
    # load policy
    train_cfg.runner.resume = False
    _,train_cfg = update_cfg_from_args(None, train_cfg, args)
    train_cfg_dict = class_to_dict(train_cfg)
    
    ppo_runner = Runner(env=env, train_cfg=train_cfg_dict,log_dir=None,device = args.rl_device)
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner.load(model_path)
    policy = ppo_runner.get_inference_policy(device=env.device)

    tmp_eval_name = train_cfg.runner.experiment_name + "-push_force-"+ str(push_force) + "-push_interval-" + str(push_interval) + "-push_duration-"+str(push_duration) + "-" + eval_name
            
    # with torch.inference_mode():
    with torch.no_grad():
        for i in range(int(env.max_episode_length) + 10):
            actions = policy(push_env.obs_dict,use_teacher=use_teacher)
            push_env.step(actions.detach())
    eval_res = push_env.get_result()
    eval_res['name'] = eval_name
    if os.path.exists(eval_path) == False:
        os.makedirs(eval_path)
    eval_file_name = os.path.join(eval_path,tmp_eval_name)
    np.save(eval_file_name, eval_res)
    print("Eval Done")




if __name__ == '__main__':
    args = get_eval_args()
    #! Eval Expert

    eval_push(
        args,
        model_type=args.model_type,
        model_path=args.model_path,
        push_force=args.push_force,
        random_push=args.random_push,
        push_interval=args.push_interval,
        push_duration=args.push_duration,
        target_vel=[args.cmd_vel,0.0,0.0],
        use_teacher=args.use_teacher,
        eval_path = args.eval_path,
        eval_name=args.eval_name
    )

    
    