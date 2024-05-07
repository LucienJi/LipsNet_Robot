from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
import numpy as np 
import os 
import torch
import gym
from collections import defaultdict


class NoisyWrapper():

    def __init__(self, env:LeggedRobot, env_cfg, cmd_vel = [0.5, 0.0,0.0],
                 record = False, move_camera = False,experiment_name = 'NoisePercent'):
        self.env = env
        self.env.set_eval()
        self.eval_config = None
        self.experiment_name = experiment_name
        self.noise_level = 1
        self.noise_type = 'gaussian'
        self.camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        self.camera_vel = np.array([1., 1., 0.])
        self.camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        self.env.set_eval()
        self.cmd_vel = cmd_vel
        with torch.inference_mode():
            self.env.reset()
            self.env.set_commands(cmd_vel)
        self.obs_dict = self.env.get_observations()
        self.record = record 
        self.move_camera = move_camera

        self.step_ct = 0
        self.img_idx = 0
        self.eval_res = defaultdict(list)
    
    def reset(self):
        self.env.set_eval()
        with torch.inference_mode():
            self.env.reset()
            self.env.set_commands(self.cmd_vel)
        self.obs_dict = self.env.get_observations()
        self.eval_res = defaultdict(list)
        
    def set_noise_scale(self,noise_level,noise_type):
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.env.set_noise_scale(noise_level,noise_type)
    
    def set_noise_percent(self,percent,noise_type):
        self.noise_level = percent
        self.noise_type = noise_type
        self.env.set_noise_percent(percent,noise_type)
    
    
    def step(self, action):
         
        self.obs_dict, rewards, dones, infos= self.env.step(action.detach())
        eval_res = self.env.get_noise_eval_result()
        for k,v in eval_res.items():
            self.eval_res[k].append(v)

        self.step_ct += 1    

        if self.record:
            if self.step_ct % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.experiment_name, 'exported', 'frames', f"{self.img_idx}.png")
                self.env.gym.write_viewer_image_to_file(self.env.viewer, filename)
                self.img_idx += 1 
        if self.move_camera:
            self.camera_position += self.camera_vel * self.env.dt
            self.env.set_camera(self.camera_position, self.camera_position + self.camera_direction)
        
        
    def get_result(self):
        for k,v in self.eval_res.items():
            if type(v) == list:
                self.eval_res[k] = np.stack(v, axis=1) # (n_env, n_step)
        first_done = np.argmax(self.eval_res['done'], axis = 1)
        self.eval_res['first_done'] = first_done
        self.eval_res['Fall'] = first_done < 1000
        self.eval_res['noise_level'] = self.noise_level
        self.eval_res['noise_type'] = self.noise_type
        res = self.eval_res.copy()
        self.eval_res.clear()
        return res
        