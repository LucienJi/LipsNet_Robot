#! /bin/bash
python legged_gym/scripts/train.py --task=go1 --experiment_name=go1_flat --max_iterations=5000 --num_envs=4096 --headless --sim_device=cuda:1 --rl_device=cuda:1 --resume