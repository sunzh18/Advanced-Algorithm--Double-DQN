from collections import deque
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from utils_drl2 import Agent
from utils_env2 import MyEnv
from utils_memory import ReplayMemory
from utils_memory import PERMemory


GAMMA = 0.99        # 衰减因子为0.99
GLOBAL_SEED = 0     # 随机数种子
MEM_SIZE = 100_000  # 经验池容量
RENDER = False      # 不保存图片帧
SAVE_PREFIX = "./models"
STACK_SIZE = 4      # 一个状态为4帧图片

EPS_START = 1.      # eps初始为1
EPS_END = 0.1       # 最终值为0.1
EPS_DECAY = 10000   # 衰减率

BATCH_SIZE = 32     # mini_batch大小
POLICY_UPDATE = 4   # 策略网络更新周期
TARGET_UPDATE = 1_000  # 每迭代10000次更新target网络，target网络更新周期
WARM_STEPS = 1_000     # 经验池容量足够后，开始训练
MAX_STEPS = 100_000      # 最大迭代次数

EVALUATE_FREQ = 1_000     # 验证频率

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)    # 生成一个随机数
if not os.path.exists(SAVE_PREFIX):
    os.mkdir(SAVE_PREFIX)

torch.manual_seed(new_seed())   # torch的随机数种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MyEnv(device)     # 环境初始化


#### Training ####
obs_queue: deque = deque(maxlen=5)                               #双向列表，最大长度为5
done = True

if __name__ == "__main__":
    REWARD = []
    global memory
    c = 1
    progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                       ncols=50, leave=False, unit="b")
    while (c <= 1):
        if c == 0:
            agent = Agent(
                env.get_action_dim(),
                device,
                GAMMA,
                new_seed(),
                EPS_START,
                EPS_END,
                EPS_DECAY,
                True,       # DuelingDQN

            )
        else:
            agent = Agent(
                env.get_action_dim(),
                device,
                GAMMA,
                new_seed(),
                EPS_START,
                EPS_END,
                EPS_DECAY,
                False,

            )
        memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)
        Reward = []
        for step in progressive:
            if done:
                observations, _, _ = env.reset()
                for obs in observations:
                    obs_queue.append(obs)

            training = len(memory) > WARM_STEPS
            state = env.make_state(obs_queue).to(device).float()
            action, value_this = agent.run(state, training)  # 当前选择的行动和Q值
            obs, reward, done = env.step(action)
            obs_queue.append(obs)
            # memory.push(env.make_folded_state(obs_queue), action, reward, done)
            # pynvml.nvmlInit()
            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print(meminfo.used)

            memory.push(env.make_folded_state(obs_queue), action, reward, done)

            if step % POLICY_UPDATE == 0 and training:
                agent.learn(memory, BATCH_SIZE, 0)  # PER时学习到的LOSS要来更新优先级

            if step % TARGET_UPDATE == 0:
                agent.sync()

            if step % EVALUATE_FREQ == 0:
                avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
                with open("rewards2.txt", "a") as fp:
                    fp.write(f"{step // EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
                if RENDER:
                    prefix = f"eval_{step // EVALUATE_FREQ:03d}"
                    os.mkdir(prefix)
                    for ind, frame in enumerate(frames):
                        with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                            frame.save(fp, format="png")
                agent.save(os.path.join(
                    SAVE_PREFIX, f"model_{step // EVALUATE_FREQ:03d}"))
                done = True
                Reward.append(avg_reward)
        REWARD.append(Reward)
        c = c + 1

    plt.plot(range(len(REWARD[0])), REWARD[0], 'r', label='DuelingDQN')
    plt.plot(range(len(REWARD[1])), REWARD[1], 'b', label='DQN')
    plt.legend()
    plt.savefig('./result2.jpg')
