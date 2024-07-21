import os
import cv2
import torch
import random
import numpy as np
from time import *
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from itertools import count
from collections import deque
import matplotlib.pyplot as plt

from Config import *

model_path = f'{ENV_NAME} {NN_WIDTH}WIDTH {NN_DEPTH}DEPTH.model'

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        # func = nn.Softmax()
        func = nn.ReLU()

        super(QNetwork, self).__init__()
        self.fc=[nn.Linear(state_dim, NN_WIDTH)]
        if DEBUG:
            self.fc.append(nn.Dropout(0.2))
        self.fc.append(func)
        for _ in range(NN_DEPTH - 2):
            self.fc.append(nn.Linear(NN_WIDTH, NN_WIDTH))
            if DEBUG:
                self.fc.append(nn.Dropout())
            self.fc.append(func)
        self.fc.append(nn.Linear(NN_WIDTH, action_dim))
        self.fc = nn.Sequential(*self.fc)
    
    def forward(self, x):
        return self.fc(x)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        if len(self.memory) == self.memory.maxlen:
            self.memory.popleft()
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        '''
            state_dim: 状态维度
            action_dim: 动作维度
        '''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        if os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path))
            self.policy_net.eval()
            print(f'Load model from: {model_path}.')
        else:
            print('New a model.')

        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
        self.epsilon = EPSILON_START
    
    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.steps_done / EPSILON_DECAY)
        if random.random() > self.epsilon:
            with torch.no_grad():
                # print(self.policy_net(state))
                # print(self.policy_net(state).max(0))
                # print(self.policy_net(state).max(0)[1].view(1, 1))
                return self.policy_net(state).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
    
    def optimize_model(self, t):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = [a.unsqueeze(0) for a in batch_state]
        batch_next_state = [a.unsqueeze(0) for a in batch_next_state]
        
        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_next_state = torch.cat(batch_next_state)
        batch_done = torch.cat(batch_done)
        
        current_q_values = self.policy_net(batch_state).gather(1, batch_action)
        next_q_values = self.target_net(batch_next_state).max(1)[0].detach()
        expected_q_values = batch_reward + (GAMMA * next_q_values * (1 - batch_done))

        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        
        # if t % TAU == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())


# Environment setup
env = gym.make(ENV_NAME, render_mode='rgb_array')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(state_dim, action_dim)

start = time()
durations = []
done_rates = []

def plot_durations(show_result=False):
    tensor_durations = torch.tensor(durations, dtype=torch.float)
    tensor_done_rates = torch.tensor(done_rates, dtype=torch.float)
    
    fig, ax1 = plt.subplots()
    
    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')
        
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Pre 100 rounds Duration')
    color = 'tab:blue'
    ax1.tick_params(axis='y', labelcolor=color)
    
    if len(tensor_durations) >= 100:
        means = tensor_durations.unfold(0, 100, 1).mean(1).view(-1)
        prefix_means = torch.cumsum(tensor_durations[:99], 0) / torch.arange(1, 100)
        means = torch.cat((prefix_means, means))
        ax1.plot(means.numpy(), label='Mean Duration', color=color)
        
    else:
        prefix_means = torch.cumsum(tensor_durations, 0) / torch.arange(1, len(tensor_durations) + 1)
        ax1.plot(prefix_means.numpy(), label='Mean Duration', color=color)

    ax2 = ax1.twinx()  # 设置双 y 轴
    color = 'tab:red'
    ax2.set_ylabel('Pre 100 rounds Truncated Rate', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    if len(tensor_done_rates) >= 100:
        means = tensor_done_rates.unfold(0, 100, 1).mean(1).view(-1)
        prefix_means = torch.cumsum(tensor_done_rates[:99], 0) / torch.arange(1, 100)
        means = torch.cat((prefix_means, means))
        ax2.plot(means.numpy(), label='Mean Done Rate', color=color)
    else:
        prefix_means = torch.cumsum(tensor_done_rates, 0) / torch.arange(1, len(tensor_done_rates) + 1)
        ax2.plot(prefix_means.numpy(), label='Mean Done Rate', color=color)

    fig.tight_layout()  # 确保两个 y 轴标签不重叠
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))  # 显示图例并调整其位置
    plt.savefig(f'{ENV_NAME} {NN_WIDTH}WIDTH {NN_DEPTH}DEPTH.png')
    plt.close(fig)
           
ContinuousLoseCounter = 0

try:
    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        state = torch.tensor(state[0], device=device, dtype=torch.float32)
        
        for t in count(start=1):
            action = agent.select_action(state)
            next_state, reward, isLose, isTruncated, _ = env.step(action.item())
  
            if DEBUG:
                # 获取渲染的图像
                img = env.render()
                # 获取原始图像的尺寸
                (h, w) = img.shape[:2]

                # 计算新的尺寸，保持宽高比
                scale = 800.0 / max(h, w)
                new_size = (int(w * scale), int(h * scale))

                # 使用cv2.resize()进行等比例缩放
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                # 将图像转换为OpenCV可以处理的格式
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # 创建文本
                text = f'Episode: {i_episode}, Reward: {reward : .2f}, Duration: {t}'
                font_scale = 1
                font_color = (0, 0, 0)
                line_type = 1

                # 获取文本尺寸
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_type)[0]

                # 计算文本位置
                x, y = 0, int(text_size[1])

                # 在图像上添加文本
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, font_scale, font_color, line_type)

                # 显示图像
                cv2.imshow(f'{ENV_NAME}', img)
                cv2.waitKey(20)
    
            next_state = torch.tensor(np.array(next_state), device=device, dtype=torch.float32)
            reward = torch.tensor(np.array([reward]), device=device, dtype=torch.float32)
            isLose = torch.tensor(np.array([isLose]), device=device, dtype=torch.float32)
            
            agent.memory.push(state, action, reward, next_state, isLose)
            state = next_state
            
            agent.optimize_model(t)

            if isLose or t == STOP_THRESHOLD:
                durations.append(t)
                done_rates.append(1.0 if isTruncated else 0.0)
                print(f"Episode {i_episode}, Duration: {t}")
                if not DEBUG:
                    plot_durations()
                if isLose:
                    ContinuousLoseCounter += 1
                else:
                    ContinuousLoseCounter = 0
                break

            if isTruncated and DEBUG:
                break

        # # 视为完成训练任务
        # if ContinuousDoneCounter == CONTINUOUS_DONE_COUNTER_THRESHOLD:
        #     print('Train Task Done.')
        #     break

        # 失败次数超阈值则重启epsilon
        if ContinuousLoseCounter == CONTINUOUS_LOSE_COUNTER_THRESHOLD:
            agent.steps_done = 0
            print('A new area.')
except KeyboardInterrupt:
    pass

finally:
    if not DEBUG:
        torch.save(agent.policy_net.state_dict(), model_path)
        print('Save model.')

        plot_durations(show_result=True)

    env.close()
    print(f'time use: {time() - start:.2f} s')
