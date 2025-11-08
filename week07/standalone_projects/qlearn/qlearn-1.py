"""
Q-Learning 强化学习算法演示
=======================

O - 1 - 2 - 3 - 4 - 5
    ^ 

这是一个简单的Q-Learning算法实现，用于解决一维路径寻找问题。
智能体需要从起点(状态0)走到终点(状态5)，每次可以选择向左或向右移动。

Q-Learning是一种无模型的强化学习算法，通过学习状态-动作价值函数Q(s,a)
来找到最优策略。算法的核心是Q值更新公式：
Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

其中：
- s: 当前状态
- a: 当前动作  
- r: 即时奖励
- s': 下一个状态
- α: 学习率
- γ: 折扣因子
"""

import numpy as np
import pandas as pd
import time

# 设置随机种子，确保结果可重现
np.random.seed(2)

# ==================== 超参数设置 ====================
N_STATES = 6                    # 环境中的状态总数（0到5，共6个状态）
ACTIONS = ['left', 'right']     # 智能体可以选择的动作：向左或向右
EPSILON = 0.9                   # ε-贪婪策略中的探索概率（90%选择最优动作，10%随机探索）
ALPHA = 0.1                     # 学习率，控制Q值更新的步长
GAMMA = 0.9                     # 折扣因子，控制对未来奖励的重视程度
MAX_EPISODES = 13               # 训练的总回合数
FRESH_TIME = 0.01              # 环境显示的刷新时间间隔（秒）

def build_q_table(n_states, actions):
    """
    创建并初始化Q表
    
    Q表是Q-Learning算法的核心数据结构，用于存储每个状态-动作对的价值。
    初始时所有Q值都设为0，随着学习过程逐渐更新。
    
    参数:
        n_states (int): 环境中状态的总数
        actions (list): 可选择的动作列表
    
    返回:
        pandas.DataFrame: 初始化的Q表，行表示状态，列表示动作
    """
    q_table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # 创建全零矩阵
        columns=actions                       # 列名为动作名称
    )
    print("初始Q表:")
    print(q_table)
    print()
    return q_table


def choose_action(current_state, q_table):
    """
    使用ε-贪婪策略选择动作
    
    ε-贪婪策略是强化学习中平衡探索与利用的经典方法：
    - 以ε的概率选择当前最优动作（利用）
    - 以(1-ε)的概率随机选择动作（探索）
    
    参数:
        current_state (int): 当前所处的状态
        q_table (pandas.DataFrame): 当前的Q表
    
    返回:
        str: 选择的动作名称
    """
    state_actions = q_table.iloc[current_state, :]  # 获取当前状态下所有动作的Q值
    
    # 如果随机数大于EPSILON或者当前状态的所有Q值都为0（初始状态）
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        # 随机选择动作（探索）
        action_name = np.random.choice(ACTIONS)
        print(f"  → 探索：随机选择动作 '{action_name}'")
    else:
        # 选择Q值最大的动作（利用）
        action_name = state_actions.idxmax()
        print(f"  → 利用：选择最优动作 '{action_name}' (Q值: {state_actions[action_name]:.3f})")
    
    return action_name


def get_env_feedback(current_state, action):
    """
    环境反馈函数：根据当前状态和动作，返回下一个状态和奖励
    
    环境规则：
    - 向右移动：如果到达终点(状态5)获得奖励1，否则奖励0
    - 向左移动：总是获得奖励0
    - 边界处理：在状态0时向左移动会停留在原地
    
    参数:
        current_state (int): 当前状态
        action (str): 选择的动作
    
    返回:
        tuple: (下一个状态, 获得的奖励)
    """
    if action == 'right':
        if current_state == N_STATES - 1:  # 到达终点
            next_state = 'terminal'
            reward = 1
            print(f"  → 到达终点！获得奖励: {reward}")
        else:
            next_state = current_state + 1
            reward = 0
            print(f"  → 向右移动：{current_state} → {next_state}，奖励: {reward}")
    else:  # action == 'left'
        reward = 0
        if current_state == 0:
            next_state = current_state  # 在边界处停留
            print(f"  → 向左移动：已在边界，停留在状态 {current_state}，奖励: {reward}")
        else:
            next_state = current_state - 1
            print(f"  → 向左移动：{current_state} → {next_state}，奖励: {reward}")
    
    return next_state, reward


def update_env(current_state, episode, step_counter):
    """
    更新并显示环境状态
    
    在控制台中可视化显示智能体在环境中的位置：
    - 'o' 表示智能体当前位置
    - '-' 表示空位置  
    - 'T' 表示终点
    
    参数:
        current_state (int or str): 当前状态
        episode (int): 当前回合数
        step_counter (int): 当前步数
    """
    env_list = ['-'] * (N_STATES - 1) + ['T']  # 创建环境显示列表
    
    if current_state == 'terminal':
        interaction = f'第 {episode + 1} 回合: 总步数 = {step_counter}'
        print(f'\r{interaction}', end="")
        time.sleep(2)
        print('\r                          ', end="")
    else:
        env_list[current_state] = 'o'  # 标记智能体位置
        interaction = ''.join(env_list)
        print(f'\r{interaction}', end="")
        time.sleep(FRESH_TIME)


def q_learning():
    """
    Q-Learning主训练循环
    
    这是Q-Learning算法的核心函数，包含完整的训练过程：
    1. 初始化Q表
    2. 对每个回合进行训练
    3. 在每个回合中，智能体与环境交互直到到达终点
    4. 使用Q-Learning更新公式更新Q值
    
    返回:
        pandas.DataFrame: 训练完成后的Q表
    """
    print("=" * 50)
    print("开始Q-Learning训练")
    print("=" * 50)
    
    # 初始化Q表
    q_table = build_q_table(N_STATES, ACTIONS)
    
    # 开始训练循环
    for episode in range(MAX_EPISODES):
        print(f"\n--- 第 {episode + 1} 回合开始 ---")
        
        step_counter = 0
        current_state = 0           # 从起点开始
        is_terminal = False         # 是否到达终点的标志
        
        # 显示初始环境状态
        update_env(current_state, episode, step_counter)
        
        # 在当前回合中持续学习，直到到达终点
        while not is_terminal:
            print(f"\n步骤 {step_counter + 1}: 当前状态 = {current_state}")
            
            # 1. 选择动作（使用ε-贪婪策略）
            action = choose_action(current_state, q_table)
            
            # 2. 执行动作，获得环境反馈
            next_state, reward = get_env_feedback(current_state, action)
            
            # 3. 计算Q值更新
            q_predict = q_table.loc[current_state, action]  # 当前Q值（预测值）
            
            if next_state != 'terminal':
                # 如果没有到达终点，使用贝尔曼方程计算目标Q值
                q_target = reward + GAMMA * q_table.iloc[next_state, :].max()
                print(f"  → Q值更新：Q({current_state},{action}) = {q_predict:.3f} + {ALPHA} * ({reward} + {GAMMA} * {q_table.iloc[next_state, :].max():.3f} - {q_predict:.3f})")
            else:
                # 如果到达终点，目标Q值就是即时奖励
                q_target = reward
                is_terminal = True
                print(f"  → 到达终点，Q值更新：Q({current_state},{action}) = {q_predict:.3f} + {ALPHA} * ({reward} - {q_predict:.3f})")
            
            # 4. 更新Q表
            q_table.loc[current_state, action] += ALPHA * (q_target - q_predict)
            new_q_value = q_table.loc[current_state, action]
            print(f"  → 新Q值：Q({current_state},{action}) = {new_q_value:.3f}")
            
            # 5. 转移到下一个状态
            current_state = next_state
            
            # 6. 显示更新后的Q表和环境
            print(f"\n当前Q表:")
            print(q_table)
            update_env(current_state, episode, step_counter + 1)
            step_counter += 1
        
        print(f"\n第 {episode + 1} 回合结束，共用 {step_counter} 步")
    
    return q_table


if __name__ == '__main__':
    # 运行Q-Learning算法
    final_q_table = q_learning()
    
    # 显示最终结果
    print('\n' + '=' * 50)
    print('训练完成！最终Q表:')
    print('=' * 50)
    print(final_q_table)
    
    print('\n' + '=' * 50)
    print('Q表解读:')
    print('=' * 50)
    print('每行代表一个状态，每列代表一个动作')
    print('数值越大表示在该状态下选择该动作的价值越高')
    print('智能体会倾向于选择Q值更大的动作')
    
    # 展示学到的最优策略
    print('\n学到的最优策略:')
    for state in range(N_STATES):
        best_action = final_q_table.iloc[state, :].idxmax()
        best_q_value = final_q_table.iloc[state, :].max()
        print(f'状态 {state}: 最优动作 = {best_action}, Q值 = {best_q_value:.3f}')