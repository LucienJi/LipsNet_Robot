# LipsRobot

1. /configs:
   1. 分别记录了 baseline 和 lipsnet 的算法超参数
   2. 超参数都分为训练环境参数和算法参数
2. /legged_gym:
   1. 包含了基于 isaacgym 的环境代码
   2. 主要环境代码在 legged_gym/envs/base/legged_robot.py
   3. 测试使用的 noised 和 pushed 环境 wrapper 在 legged_gym/envs/base/noise_wrapper.py 
3. /logs:
   1. 保存训练时使用的环境参数 （env_cfg.json） 和算法参数（train_cfg.json）
   2. 保存了训练时的模型（*.pt）
4. /model_archive:
   1. 用于保存比较好的模型
   2. 用于保存用来进行 behavior cloning 的 teacher 模型
5. /resources:
   1. 仿真相关的文件
   2. 模型相关文件 resources/robots/go1 
6. /rsl_rl 核心算法
   1. /algorithms
      1. 不同的 RL 算法、BC 算法的优化部分
      2. lipsbc_ppo.py： LipsNet 的算法部分
      3. mlpbc_ppo.py: 对照组，使用 MLP 作为策略模块的 BC 算法
      4. ppo.py: 对照组，使用 MLP 作为策略模块的 PPO 算法
      5. rma_ppo.py：对照组，使用 MLP 最为策略模块，同时对历史信息进行编码的算法
   2. /modules
      1. 不同算法的 policy，value function 的实现
      2. actor_critic_pri_lipsnet_v2.py： LipsNet 用到的模块， MLPBC 也用的模块
      3. actor_critic.py： 普通的 PPO 使用的模块
      4. actor_critic_pri.py：rma 使用的模块 （但是目前表现不好）
   3. /runners
      1. 负责和环境交互，初始化 agent 的策略，buffer
      2. 实现 step，收集 reward，transition tuple
      3. rsl_rl/runners/lipsbc_runner.py： LipsNet 的环境交互代码
7. run_*.py:
   1. 运行对应算法的文件
   2. launch 函数中实现：
      1. 初始化对应的环境参数、算法参数；初始化 isaac 环境；初始化 env wrapper；初始化对应的 runner
   3. 对于使用 BC 的算法，这里也用于导入 teacher 模块
