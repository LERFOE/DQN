import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count, product
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
from datetime import datetime

# 禁用matplotlib的交互模式，避免在自动化搜索中弹出窗口
plt.ioff()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_size=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def train_dqn(params, max_episodes=500, early_stop_episodes=50):
    """
    训练DQN模型
    
    Args:
        params: 超参数字典
        max_episodes: 最大训练回合数
        early_stop_episodes: 早停检查的窗口大小
    
    Returns:
        结果字典，包含收敛回合数、最终平均奖励等信息
    """
    # 提取超参数
    BATCH_SIZE = params['batch_size']
    GAMMA = params['gamma']
    EPS_START = params['eps_start']
    EPS_END = params['eps_end']
    EPS_DECAY = params['eps_decay']
    TAU = params['tau']
    LR = params['lr']
    MEMORY_SIZE = params['memory_size']
    HIDDEN_SIZE = params.get('hidden_size', 128)
      # 初始化环境和网络
    env = gym.make("CartPole-v0")
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    
    policy_net = DQN(n_observations, n_actions, HIDDEN_SIZE).to(device)
    target_net = DQN(n_observations, n_actions, HIDDEN_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)
    
    steps_done = 0
    episode_durations = []
    
    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                    device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
    
    # 训练循环
    best_avg_reward = 0
    convergence_episode = None
    target_reward = 180  # 目标平均奖励
    
    for i_episode in range(max_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
            
            # 软更新目标网络
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            if done:
                episode_durations.append(t + 1)
                break
        
        # 检查收敛
        if i_episode >= 99:  # 至少有100个回合的数据
            avg_reward_last_100 = sum(episode_durations[-100:]) / 100
            if avg_reward_last_100 > best_avg_reward:
                best_avg_reward = avg_reward_last_100
            
            # 检查是否达到目标奖励
            if avg_reward_last_100 >= target_reward and convergence_episode is None:
                convergence_episode = i_episode + 1
                
        # 早停条件：如果最近的平均奖励已经很好且稳定
        if (i_episode >= 199 and convergence_episode is not None and 
            i_episode - convergence_episode >= early_stop_episodes):
            break
    
    env.close()
    
    # 如果没有收敛，使用最后的平均奖励
    final_avg_reward = sum(episode_durations[-100:]) / 100 if len(episode_durations) >= 100 else sum(episode_durations) / len(episode_durations)
    
    return {
        'convergence_episode': convergence_episode if convergence_episode else max_episodes,
        'final_avg_reward': final_avg_reward,
        'best_avg_reward': best_avg_reward,
        'total_episodes': i_episode + 1,
        'total_steps': steps_done,
        'episode_durations': episode_durations
    }

def grid_search():
    """执行网格搜索"""
    
    # 定义超参数网格
    param_grid = {
        'batch_size': [64, 128, 256],
        'gamma': [0.95, 0.99, 0.995],
        'eps_start': [0.9, 1.0],
        'eps_end': [0.01, 0.05, 0.1],
        'eps_decay': [500, 1000, 2000],
        'tau': [0.001, 0.005, 0.01],
        'lr': [1e-4, 5e-4, 1e-3],
        'memory_size': [5000, 10000, 20000],
        'hidden_size': [64, 128, 256]
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    print(f"总共有 {len(all_combinations)} 种参数组合需要测试")
    
    results = []
    best_result = None
    best_score = float('-inf')
    
    for i, combination in enumerate(all_combinations):
        # 创建参数字典
        params = dict(zip(param_names, combination))
        
        print(f"\n进度: {i+1}/{len(all_combinations)}")
        print(f"当前参数: {params}")
        
        try:
            # 训练模型
            result = train_dqn(params, max_episodes=400, early_stop_episodes=30)
            result['params'] = params
            
            # 计算评分 (奖励越高越好，收敛回合数越少越好)
            reward_score = result['final_avg_reward'] / 200.0  # 归一化到0-1
            episode_score = max(0, 1 - result['convergence_episode'] / 400.0)  # 归一化到0-1
            overall_score = 0.7 * reward_score + 0.3 * episode_score  # 权重组合
            
            result['score'] = overall_score
            results.append(result)
            
            print(f"结果: 平均奖励={result['final_avg_reward']:.2f}, "
                  f"收敛回合={result['convergence_episode']}, "
                  f"评分={overall_score:.4f}")
            
            # 更新最佳结果
            if overall_score > best_score:
                best_score = overall_score
                best_result = result
                print("*** 新的最佳结果! ***")
                
        except Exception as e:
            print(f"参数组合失败: {e}")
            continue
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"grid_search_results_{timestamp}.json"
    
    # 准备保存的数据（移除不能序列化的对象）
    save_results = []
    for result in results:
        save_result = {k: v for k, v in result.items() if k != 'episode_durations'}
        save_results.append(save_result)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n网格搜索完成! 结果已保存到 {results_file}")
    print(f"\n最佳参数组合:")
    print(json.dumps(best_result['params'], indent=2, ensure_ascii=False))
    print(f"最佳评分: {best_score:.4f}")
    print(f"最佳平均奖励: {best_result['final_avg_reward']:.2f}")
    print(f"最佳收敛回合数: {best_result['convergence_episode']}")
    
    return best_result, results

if __name__ == "__main__":
    best_result, all_results = grid_search()
