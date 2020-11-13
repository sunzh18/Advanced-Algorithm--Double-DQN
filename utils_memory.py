from typing import (
    Tuple,
)

import torch
import numpy as np
import pynvml

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.device = device
        self.__capacity = capacity
        self.size = 0
        self.pos = 0

        self.m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool
    ) -> None:
        self.m_states[self.pos] = folded_state
        self.m_actions[self.pos, 0] = action
        self.m_rewards[self.pos, 0] = reward
        self.m_dones[self.pos, 0] = done
        self.after_push()
       
    def get_pos(self):
        return self.pos
        
    def after_push(self):
        self.pos = (self.pos + 1) % self.__capacity
        self.size = max(self.size, self.pos)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        indices = torch.randint(0, high=self.size, size=(batch_size,))
        b_state = self.m_states[indices, :4].to(self.device).float()
        b_next = self.m_states[indices, 1:].to(self.device).float()
        b_action = self.m_actions[indices].to(self.device)
        b_reward = self.m_rewards[indices].to(self.device).float()
        b_done = self.m_dones[indices].to(self.device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.size
    
    
class PERMemory(object):
    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice):
        self.device = device
        self.__capacity = capacity
        self.size = 0
        self.pos = 0

        self.m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.m_td_errors = torch.zeros((capacity, 1), dtype=torch.float)
        
    def after_push(self):
        self.pos = (self.pos + 1) % self.__capacity
        self.size = max(self.size, self.pos)        
    
    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
            td_error: float,
    ) -> None:
        self.m_states[self.pos] = folded_state
        self.m_actions[self.pos, 0] = action
        self.m_rewards[self.pos, 0] = reward
        self.m_dones[self.pos, 0] = done
        
        self.m_td_errors[self.pos, 0] = float(td_error)+1e-6
        self.after_push()

    def update(self, td_errors, idx_batch):
        for i in range(len(idx_batch)):
            idx = idx_batch[i]
            error = td_errors[i]           
            self.m_td_errors[idx, 0] = float(error)+1e-6
        
    #@overloaded
    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        
        pro = torch.softmax(self.m_td_errors[:self.size], dim=0).squeeze(1).detach().numpy()
        indices = np.random.choice(range(self.size), size=(batch_size,), p=pro)
        #indices = torch.randint(0, high=self.size, size=(batch_size,))
        b_state = self.m_states[indices, :4].to(self.device).float()
        b_next = self.m_states[indices, 1:].to(self.device).float()
        b_action = self.m_actions[indices].to(self.device)
        b_reward = self.m_rewards[indices].to(self.device).float()
        b_done = self.m_dones[indices].to(self.device).float()
        return b_state, b_action, b_reward, b_next, b_done, indices
    def __len__(self) -> int:
        return self.size