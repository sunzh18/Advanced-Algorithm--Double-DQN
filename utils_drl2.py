from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN
from utils_model import DuelingDQN


class Agent(object):    # 代理

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            isdueling: bool = False,
            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim  # 动作维度
        self.__device = device  # 设备
        self.__gamma = gamma    # 衰减因子

        self.__eps_start = eps_start    # eps-greedy参数的初始值
        self.__eps_final = eps_final    # eps-greedy参数的最终值
        self.__eps_decay = eps_decay    # eps-greedy参数的衰减率

        self.__eps = eps_start
        self.__r = random.Random()                                 #随机浮点数
        self.__r.seed(seed)     # 随机数种子

        ###修改项
        if isdueling:   # 使用DuelingDQN网络
            self.__policy = DuelingDQN(action_dim, device).to(device)  # 值函数网络
            self.__target = DuelingDQN(action_dim, device).to(device)  # target网络
        else:
            self.__policy = DQN(action_dim, device).to(device)  # 值函数网络
            self.__target = DQN(action_dim, device).to(device)  # target网络

        if restore is None:
            if isdueling:
                self.__policy.apply(DuelingDQN.init_weights)  # 初始化权重
            else:
                self.__policy.apply(DQN.init_weights)  # 初始化权重
        ###修改项
        else:
            self.__policy.load_state_dict(torch.load(restore))    # 将restore的数据加载到网络中

        self.__target.load_state_dict(self.__policy.state_dict()) # 将policy参数赋给target，此时两个网络相同
        self.__optimizer = optim.Adam(
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()

    def run(self, state: TensorStack4, training: bool = False) -> int:              #e-greedy选择行动
        """run suggests an action for the given state."""
        if training:        # 线性衰减eps
            self.__eps -= (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps:      # 产生随机数>eps，选择使Q最大的动作
            with torch.no_grad():
                action = self.__policy(state).max(1).indices.item()
        else:
            action = self.__r.randint(0, self.__action_dim - 1)
        value_this = self.__policy(state)[0][action]
        return action, value_this  # 行动和Q值

    def get_target_value(self, state):
        value_next = self.__target(state).max(1).indices.item()
        return value_next

    def learn(self, memory: ReplayMemory, batch_size: int, choice: int) -> float:
        """learn trains the value network via TD-learning."""

        ##修改项
        if (choice == 0):   # 普通memory
            state_batch, action_batch, reward_batch, next_batch, done_batch = memory.sample(batch_size)
        else:       # PERmemory
            state_batch, action_batch, reward_batch, next_batch, done_batch, idx_batch = \
                memory.sample(batch_size)  ####

        values = self.__policy(state_batch.float()).gather(1, action_batch)  # 每一列按action_batch取元素
        values_next = self.__target(next_batch.float()).max(1).values.detach()  # 最大的作为下一个的value
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
                   (1. - done_batch) + reward_batch
        # Loss=values-expected
        if (choice == 0):  #####
            loss = F.smooth_l1_loss(values, expected)
        else:       # PERmemory
            loss_batch = F.smooth_l1_loss(values, expected, reduction='none')  # TD error
            loss = torch.mean(loss_batch, dim=0)
            # loss.requires_grad = True
            memory.update(loss_batch.detach(), idx_batch)
        ##修改项

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():  # 把参数加紧到[-1,1],原地修改
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item()

    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
