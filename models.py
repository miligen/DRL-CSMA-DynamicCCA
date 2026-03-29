import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import *

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

class DQNAgent:
    # 【修改】增加 use_link_quality 参数
    def __init__(self, node_id, use_link_quality=True):
        self.node_id = node_id

        # 【新增】根据是否启用链路质量矩阵，动态决定神经网络的输入维度
        actual_state_dim = STATE_DIM if use_link_quality else K_SENSE_HISTORY

        self.policy_net = DQN(actual_state_dim, ACTION_DIM).to(DEVICE)
        self.target_net = DQN(actual_state_dim, ACTION_DIM).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state, valid_mask):
        eps = EPSILON_END + (EPSILON_START - EPSILON_END) * \
              np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1

        if random.random() > eps:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_vals = self.policy_net(state_t)
                mask_t = torch.BoolTensor(valid_mask).to(DEVICE).unsqueeze(0)
                q_vals[~mask_t] = -1e9
                return q_vals.max(1)[1].item()
        else:
            valid_indices = [i for i, x in enumerate(valid_mask) if x]
            if not valid_indices: return 0
            return random.choice(valid_indices)

    def update(self):
        if len(self.memory) < BATCH_SIZE: return
        batch = self.memory.sample(BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(DEVICE)
        action = torch.LongTensor(action).unsqueeze(1).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        next_state = torch.FloatTensor(np.array(next_state)).to(DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)

        q_curr = self.policy_net(state).gather(1, action)
        q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
        expected_q = reward + (GAMMA_RL * q_next * (1 - done))

        loss = nn.MSELoss()(q_curr, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()