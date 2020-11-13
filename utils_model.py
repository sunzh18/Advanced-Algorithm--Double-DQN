import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):       # DQN卷积网络

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()                                            #4*84*84        似乎是四帧
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)   #32*20*20
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)  #64*9*9
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)  #64*7*7
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(512, action_dim)                                #全连接至动作维度
        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        return self.__fc2(x)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")


class DuelingDQN(nn.Module):
    def __init__(self, action_dim, device):
        super(DuelingDQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)  # 32*20*20
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)  # 64*9*9
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)  # 64*7*7
        self.__fc1 = nn.Linear(64 * 7 * 7, 512)     # 状态值函数全连接层
        self.__fc2 = nn.Linear(64 * 7 * 7, 512)     # 动作优势函数全连接层
        self.__fc_state_value = nn.Linear(512, 1)   # 状态值函数
        self.__fc_action_adv = nn.Linear(512, action_dim)   # 动作优势函数
        self.__device = device
        self.action_dim = action_dim
        #self.__fc2 = nn.Linear(512, action_dim)  # 全连接至动作维度

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x1 = F.relu(self.__fc1(x.view(x.size(0), -1)))
        x2 = F.relu(self.__fc2(x.view(x.size(0), -1)))
        
        state_value = F.relu(self.__fc_state_value(x1))  # 状态价值
        action_adv = F.relu(self.__fc_action_adv(x2))    # 动作优势
        arg_action_adv = torch.mean(action_adv, dim = 1).unsqueeze(1)
        #print(state_value.shape, action_adv.shape, arg_action_adv.shape)
        result = (state_value - arg_action_adv).repeat(1, self.action_dim) + action_adv
        return result

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")