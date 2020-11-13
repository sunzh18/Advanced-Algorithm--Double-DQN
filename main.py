from collections import deque  
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import pynvml
from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory
from utils_memory import PERMemory


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
#EPS_DECAY = 1000000
EPS_DECAY = 10000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 1000
WARM_STEPS = 1000
MAX_STEPS = 50000
EVALUATE_FREQ = 1000                                                   #5000步评估一次

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
if not os.path.exists(SAVE_PREFIX):
    os.mkdir(SAVE_PREFIX)

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(1)
env = MyEnv(device)

#memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True

#progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
 #                  ncols=50, leave=False, unit="b")



def choosememory(c):
    if c==0:
        return ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)
    else:
        return PERMemory(STACK_SIZE + 1, MEM_SIZE, device)
    
    
if __name__ == "__main__":   
    REWARD=[]
    global memory
    c=1
    while(c>=0):
        agent = Agent(
            env.get_action_dim(),
            device,
            GAMMA,
            new_seed(),
            EPS_START,
            EPS_END,
            EPS_DECAY,
            "./model_weights_b"####*************
            )
        memory=choosememory(c)
        Reward=[]
        for step in tqdm(range(MAX_STEPS),ncols=50,leave=False,unit="b"):
            if done:
                observations, _, _ = env.reset()
                for obs in observations:
                    obs_queue.append(obs)

            training = len(memory) > WARM_STEPS
            state = env.make_state(obs_queue).to(device).float()
            action,value_this = agent.run(state, training)                           #当前选择的行动和Q值
            obs, reward, done = env.step(action)
            obs_queue.append(obs)
            #memory.push(env.make_folded_state(obs_queue), action, reward, done)
            #pynvml.nvmlInit()
            #handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            #meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            #print(meminfo.used)         
            if c==1:                                                                 #在PER时加入memory时用TDerror来更新优先                 
                state_next = env.make_state(obs_queue).to(device).float()
                value_next = agent.get_target_value(state_next)
                td_error = abs(GAMMA*value_next+reward - value_this)                #用loss的绝对值来作为TDerror
                memory.push(env.make_folded_state(obs_queue), action, reward, done, td_error) 
            else:
                memory.push(env.make_folded_state(obs_queue), action, reward, done)
            if step % POLICY_UPDATE == 0 and training:
                agent.learn(memory, BATCH_SIZE,c)                                    #PER时学习到的LOSS要来更新优先级

            if step % TARGET_UPDATE == 0:
                agent.sync()

            if step % EVALUATE_FREQ == 0:
                avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
                with open("rewards.txt", "a") as fp:
                    fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
                if RENDER:
                    prefix = f"eval_{step//EVALUATE_FREQ:03d}"
                    os.mkdir(prefix)
                    for ind, frame in enumerate(frames):
                        with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                            frame.save(fp, format="png")
                agent.save(os.path.join(
                    SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
                done = True
                Reward.append(avg_reward)
        REWARD.append(Reward)
        c=c-1
        
    plt.plot(range(len(REWARD[1])),REWARD[1],'r',label='Replaymemory')
    plt.plot(range(len(REWARD[0])),REWARD[0],'b',label='PERmemory')
    plt.legend()
    plt.savefig('./result1.jpg')
            