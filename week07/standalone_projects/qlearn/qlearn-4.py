import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import os
from typing import Tuple, Dict, List

"""
Q-learning算法解决CartPole平衡问题

本程序实现了一个完整的Q-learning强化学习系统，用于训练智能体在CartPole环境中保持杆子平衡。

CartPole环境详解：
- 观察空间：4维连续值 [小车位置, 小车速度, 杆子角度, 杆子角速度]
- 动作空间：2个离散动作 [0: 向左推小车, 1: 向右推小车]
- 成功标准：连续保持杆子平衡195步以上
- 失败条件：杆子倾斜超过±12°或小车移出±2.4单位范围

Q-learning算法原理：
Q-learning是一种无模型的强化学习算法，通过学习状态-动作价值函数Q(s,a)来找到最优策略。
核心更新公式：Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
其中：α为学习率，γ为折扣因子，r为即时奖励
"""


class QLearningAgent:
    """
    Q-learning智能体类
    
    封装了Q-learning算法的核心功能，包括状态离散化、Q值管理、策略选择等
    """
    
    def __init__(self, learning_rate: float = 0.25, discount_factor: float = 0.99, 
                 initial_exploration_rate: float = 0.4, min_exploration_rate: float = 0.005,
                 exploration_decay: float = 0.9985):
        """
        初始化Q-learning智能体
        
        参数:
            learning_rate: 学习率，控制新信息的接受程度 (0-1)
            discount_factor: 折扣因子，控制未来奖励的重要性 (0-1)
            initial_exploration_rate: 初始探索率
            min_exploration_rate: 最小探索率
            exploration_decay: 探索率衰减系数
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_exploration_rate = initial_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        
        # Q表：存储状态-动作价值对
        self.q_table: Dict[Tuple, float] = {}
        
        # 状态空间离散化参数（进一步优化）
        self.cart_position_bins = np.linspace(-2.4, 2.4, 15)    # 小车位置：15个区间
        self.cart_velocity_bins = np.linspace(-4, 4, 15)         # 小车速度：15个区间  
        self.pole_angle_bins = np.linspace(-0.2, 0.2, 25)       # 杆子角度：25个区间（最关键）
        self.pole_velocity_bins = np.linspace(-4, 4, 20)        # 杆子角速度：20个区间
    
    def discretize_state(self, observation: np.ndarray) -> Tuple[int, int, int, int]:
        """
        将连续的观察空间离散化为有限的状态空间
        
        这是Q-learning处理连续状态空间的关键步骤。通过将连续值映射到离散区间，
        我们可以使用表格方法存储和更新Q值。
        
        参数:
            observation: 4维观察向量 [cart_position, cart_velocity, pole_angle, pole_velocity]
        
        返回:
            tuple: 离散化后的状态元组，用作Q表的键
        """
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        
        # 使用np.digitize将连续值映射到离散区间索引
        # 限制索引范围，避免越界问题
        discretized = [
            min(max(np.digitize(cart_position, self.cart_position_bins), 1), len(self.cart_position_bins)),
            min(max(np.digitize(cart_velocity, self.cart_velocity_bins), 1), len(self.cart_velocity_bins)),
            min(max(np.digitize(pole_angle, self.pole_angle_bins), 1), len(self.pole_angle_bins)),
            min(max(np.digitize(pole_velocity, self.pole_velocity_bins), 1), len(self.pole_velocity_bins))
        ]
        
        return tuple(discretized)
    
    def get_q_value(self, state: Tuple, action: int) -> float:
        """
        获取指定状态-动作对的Q值
        
        参数:
            state: 离散化后的状态
            action: 动作（0或1）
        
        返回:
            float: Q值，如果状态-动作对不存在则返回0.0（乐观初始化）
        """
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state: Tuple, action: int, reward: float, 
                      next_state: Tuple, terminated: bool) -> None:
        """
        使用Q-learning算法更新Q值
        
        这是Q-learning的核心更新步骤，实现了时序差分学习。
        更新公式：Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            terminated: 是否为终止状态
        """
        # 确保当前状态-动作对存在于Q表中
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        
        current_q = self.q_table[(state, action)]
        
        if terminated:
            # 终止状态没有未来奖励
            target_q = reward
        else:
            # 计算下一状态的最大Q值（贪婪策略）
            next_q_values = [self.get_q_value(next_state, a) for a in range(2)]
            max_next_q = max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q
        
        # 时序差分更新
        td_error = target_q - current_q
        self.q_table[(state, action)] = current_q + self.learning_rate * td_error

    def choose_action(self, state: Tuple, exploration_rate: float) -> int:
        """
        使用ε-贪婪策略选择动作
        
        参数:
            state: 当前状态
            exploration_rate: 当前探索率
        
        返回:
            int: 选择的动作（0或1）
        """
        if np.random.random() < exploration_rate:
            # 探索：随机选择动作
            return np.random.randint(2)
        else:
            # 利用：选择Q值最大的动作
            q_values = [self.get_q_value(state, a) for a in range(2)]
            return np.argmax(q_values)
    
    def get_shaped_reward(self, observation: np.ndarray, terminated: bool, step_count: int) -> float:
        """
        奖励塑形函数：设计更精细的奖励信号来引导学习（优化版）
        
        奖励塑形是强化学习中的重要技术，通过提供额外的奖励信号来加速学习过程。
        
        参数:
            observation: 当前观察值
            terminated: 是否终止
            step_count: 当前步数
        
        返回:
            float: 塑形后的奖励值
        """
        if terminated:
            # 根据存活时间给予不同程度的惩罚（优化版）
            if step_count < 30:
                return -15.0    # 极早期失败：重罚
            elif step_count < 80:
                return -8.0     # 早期失败：较重惩罚
            elif step_count < 150:
                return -3.0     # 中期失败：中等惩罚
            else:
                return -0.5     # 后期失败：轻微惩罚
        
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        
        # 基础存活奖励
        reward = 1.0
        
        # 位置奖励：鼓励小车保持在中心附近（增强权重）
        position_reward = max(0, 1.0 - abs(cart_position) / 2.4)
        reward += position_reward * 0.15
        
        # 角度奖励：鼓励杆子保持垂直（增强权重）
        angle_reward = max(0, 1.0 - abs(pole_angle) / 0.2)
        reward += angle_reward * 0.3
        
        # 稳定性奖励：惩罚过大的速度（优化权重）
        velocity_penalty = (abs(cart_velocity) / 4.0 + abs(pole_velocity) / 4.0)
        reward -= velocity_penalty * 0.08
        
        # 长期存活奖励：鼓励持续平衡（优化阈值）
        if step_count > 80:
            reward += 0.3
        if step_count > 150:
            reward += 0.7
        if step_count > 250:
            reward += 1.2
        
        # 超级稳定奖励：角度和位置都很好时的额外奖励
        if abs(pole_angle) < 0.05 and abs(cart_position) < 1.0:
            reward += 0.5
        
        return reward
    
    def get_exploration_rate(self, episode: int) -> float:
        """
        计算当前回合的探索率
        
        参数:
            episode: 当前回合数
        
        返回:
            float: 当前探索率
        """
        return max(self.min_exploration_rate, 
                  self.initial_exploration_rate * (self.exploration_decay ** episode))
    
    def get_q_table_size(self) -> int:
        """获取Q表的大小（状态数量）"""
        return len(self.q_table)
    
    def save_q_table(self, filepath: str) -> None:
        """
        保存Q表到文件
        
        参数:
            filepath: 保存文件的路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存Q表和相关参数
            save_data = {
                'q_table': dict(self.q_table),
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'initial_exploration_rate': self.initial_exploration_rate,
                'min_exploration_rate': self.min_exploration_rate,
                'exploration_decay': self.exploration_decay,
                'cart_position_bins': self.cart_position_bins,
                'cart_velocity_bins': self.cart_velocity_bins,
                'pole_angle_bins': self.pole_angle_bins,
                'pole_velocity_bins': self.pole_velocity_bins
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"✅ Q表已保存到: {filepath}")
            print(f"   - Q表大小: {len(self.q_table)} 个状态")
            
        except Exception as e:
            print(f"❌ 保存Q表失败: {e}")
    
    def load_q_table(self, filepath: str) -> bool:
        """
        从文件加载Q表
        
        参数:
            filepath: Q表文件路径
            
        返回:
            bool: 是否成功加载
        """
        try:
            if not os.path.exists(filepath):
                print(f"❌ 文件不存在: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # 恢复Q表和参数
            self.q_table = save_data['q_table']
            self.learning_rate = save_data['learning_rate']
            self.discount_factor = save_data['discount_factor']
            self.initial_exploration_rate = save_data['initial_exploration_rate']
            self.min_exploration_rate = save_data['min_exploration_rate']
            self.exploration_decay = save_data['exploration_decay']
            self.cart_position_bins = save_data['cart_position_bins']
            self.cart_velocity_bins = save_data['cart_velocity_bins']
            self.pole_angle_bins = save_data['pole_angle_bins']
            self.pole_velocity_bins = save_data['pole_velocity_bins']
            
            print(f"✅ Q表已从文件加载: {filepath}")
            print(f"   - Q表大小: {len(self.q_table)} 个状态")
            print(f"   - 学习率: {self.learning_rate}")
            print(f"   - 折扣因子: {self.discount_factor}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载Q表失败: {e}")
            return False
    
    def get_performance_stats(self, episode_rewards: List[float], window_size: int = 100) -> Dict:
        """
        计算训练性能统计信息
        
        参数:
            episode_rewards: 每回合奖励列表
            window_size: 滑动窗口大小
        
        返回:
            dict: 包含各种性能指标的字典
        """
        if not episode_rewards:
            return {}
        
        stats = {
            'total_episodes': len(episode_rewards),
            'max_reward': max(episode_rewards),
            'min_reward': min(episode_rewards),
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': sum(1 for r in episode_rewards if r >= 195) / len(episode_rewards) * 100
        }
        
        # 计算滑动窗口平均值
        if len(episode_rewards) >= window_size:
            recent_rewards = episode_rewards[-window_size:]
            stats['recent_mean'] = np.mean(recent_rewards)
            stats['recent_std'] = np.std(recent_rewards)
            stats['recent_success_rate'] = sum(1 for r in recent_rewards if r >= 195) / len(recent_rewards) * 100
        
        # 计算学习进展
        if len(episode_rewards) >= 200:
            early_mean = np.mean(episode_rewards[:100])
            late_mean = np.mean(episode_rewards[-100:])
            stats['improvement'] = late_mean - early_mean
        
        return stats

def plot_training_progress(episode_rewards: List[float], agent: QLearningAgent) -> None:
    """
    绘制训练过程的可视化图表
    
    参数:
        episode_rewards: 每回合奖励列表
        agent: 训练好的智能体
    """
    if not episode_rewards:
        print("⚠️ 没有训练数据可供可视化")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-learning CartPole 训练过程分析', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(episode_rewards) + 1)
    
    # 1. 每回合奖励曲线
    ax1.plot(episodes, episode_rewards, alpha=0.6, color='lightblue', linewidth=0.8)
    
    # 计算滑动平均
    window_size = min(50, len(episode_rewards) // 10)
    if len(episode_rewards) >= window_size:
        moving_avg = []
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(episode_rewards[start_idx:i+1]))
        ax1.plot(episodes, moving_avg, color='red', linewidth=2, label=f'滑动平均({window_size}回合)')
    
    ax1.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='成功线(195步)')
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('奖励(步数)')
    ax1.set_title('训练过程 - 每回合表现')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 奖励分布直方图
    ax2.hist(episode_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=np.mean(episode_rewards), color='red', linestyle='--', 
                label=f'平均值: {np.mean(episode_rewards):.1f}')
    ax2.axvline(x=195, color='green', linestyle='--', label='成功线: 195')
    ax2.set_xlabel('奖励(步数)')
    ax2.set_ylabel('频次')
    ax2.set_title('奖励分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 学习进展（分段平均）
    if len(episode_rewards) >= 100:
        segment_size = max(50, len(episode_rewards) // 10)
        segments = []
        segment_means = []
        
        for i in range(0, len(episode_rewards), segment_size):
            end_idx = min(i + segment_size, len(episode_rewards))
            segment_rewards = episode_rewards[i:end_idx]
            segments.append(f'{i+1}-{end_idx}')
            segment_means.append(np.mean(segment_rewards))
        
        ax3.bar(range(len(segments)), segment_means, alpha=0.7, color='lightgreen')
        ax3.axhline(y=195, color='red', linestyle='--', alpha=0.7, label='成功线')
        ax3.set_xlabel('训练阶段')
        ax3.set_ylabel('平均奖励')
        ax3.set_title('学习进展 - 分段平均表现')
        ax3.set_xticks(range(len(segments)))
        ax3.set_xticklabels([f'第{i+1}段' for i in range(len(segments))], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 探索率衰减曲线
    exploration_rates = [agent.get_exploration_rate(ep) for ep in range(len(episode_rewards))]
    ax4.plot(episodes, exploration_rates, color='orange', linewidth=2)
    ax4.set_xlabel('回合数')
    ax4.set_ylabel('探索率')
    ax4.set_title('探索率衰减')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细统计信息
    stats = agent.get_performance_stats(episode_rewards)
    print(f"\n📊 详细训练统计：")
    print(f"   - 总回合数: {stats.get('total_episodes', 0)}")
    print(f"   - 平均表现: {stats.get('mean_reward', 0):.2f} ± {stats.get('std_reward', 0):.2f} 步")
    print(f"   - 最佳表现: {stats.get('max_reward', 0)} 步")
    print(f"   - 最差表现: {stats.get('min_reward', 0)} 步")
    print(f"   - 整体成功率: {stats.get('success_rate', 0):.1f}%")
    
    if 'recent_mean' in stats:
        print(f"   - 最近100回合平均: {stats['recent_mean']:.2f} ± {stats['recent_std']:.2f} 步")
        print(f"   - 最近100回合成功率: {stats['recent_success_rate']:.1f}%")
    
    if 'improvement' in stats:
        print(f"   - 学习改进: {stats['improvement']:.2f} 步")


def train_agent(num_episodes: int = 2000) -> Tuple[QLearningAgent, List[float]]:
    """
    训练Q-learning智能体
    
    参数:
        num_episodes: 训练回合数
    
    返回:
        tuple: (训练好的智能体, 每回合奖励列表)
    """
    # 创建环境和智能体
    env = gym.make('CartPole-v1')
    agent = QLearningAgent()
    
    print("🚀 开始Q-learning训练...")
    print("=" * 80)
    
    episode_rewards = []  # 记录每个回合的奖励
    best_reward = 0      # 记录最佳表现
    consecutive_good_episodes = 0  # 连续好表现的回合数
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # 重置环境，获取初始观察
        observation, info = env.reset()
        state = agent.discretize_state(observation)
        
        total_reward = 0
        terminated = False
        truncated = False
        step_count = 0
        
        # 获取当前探索率
        exploration_rate = agent.get_exploration_rate(episode)
        
        # 开始回合循环
        while not (terminated or truncated):
            # 选择动作
            action = agent.choose_action(state, exploration_rate)
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize_state(observation)
            
            # 获取塑形奖励
            shaped_reward = agent.get_shaped_reward(observation, terminated or truncated, step_count)
            
            # 更新Q值
            agent.update_q_value(state, action, shaped_reward, next_state, terminated or truncated)
            
            # 更新状态和统计信息
            state = next_state
            total_reward += reward  # 使用原始奖励记录表现
            step_count += 1
        
        episode_rewards.append(total_reward)
        
        # 跟踪表现
        if total_reward > best_reward:
            best_reward = total_reward
            consecutive_good_episodes = 0
        elif total_reward >= 195:  # CartPole成功标准
            consecutive_good_episodes += 1
        else:
            consecutive_good_episodes = 0
        
        # 定期打印进度
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            elapsed_time = time.time() - start_time
            print(f"回合 {episode:4d} | 奖励: {total_reward:3.0f} | 平均奖励: {avg_reward:6.2f} | "
                  f"探索率: {exploration_rate:.3f} | 最佳: {best_reward} | Q表大小: {agent.get_q_table_size()} | "
                  f"时间: {elapsed_time:.1f}s")
        
        # 早停机制
        if consecutive_good_episodes >= 100:
            print(f"\n🎉 智能体已学会平衡！第 {episode} 回合达到稳定表现")
            break
    
    env.close()
    
    total_time = time.time() - start_time
    print(f"\n✅ 训练完成！")
    print(f"📊 训练统计：")
    print(f"   - 总回合数: {len(episode_rewards)}")
    print(f"   - Q表大小: {agent.get_q_table_size()} 个状态-动作对")
    print(f"   - 最佳表现: {best_reward} 步")
    print(f"   - 平均表现: {np.mean(episode_rewards[-100:]):.2f} 步（最后100回合）")
    print(f"   - 训练时间: {total_time:.2f} 秒")
    print("=" * 80)
    
    return agent, episode_rewards

def test_agent(agent: QLearningAgent, render: bool = True, num_tests: int = 5) -> List[float]:
    """
    测试训练好的智能体
    
    参数:
        agent: 训练好的Q-learning智能体
        render: 是否显示图形界面
        num_tests: 测试回合数
    
    返回:
        list: 每次测试的奖励列表
    """
    print(f"\n🧪 开始测试智能体（{num_tests}回合）...")
    print("=" * 60)
    
    test_rewards = []
    
    for test_episode in range(num_tests):
        # 创建测试环境
        if render and test_episode == 0:  # 只在第一次测试时显示图形
            test_env = gym.make('CartPole-v1', render_mode='human')
            print("🎮 图形界面已开启，观察智能体表现...")
        else:
            test_env = gym.make('CartPole-v1')
        
        observation, info = test_env.reset()
        state = agent.discretize_state(observation)
        
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not (terminated or truncated):
            # 纯利用策略：选择Q值最大的动作
            q_values = [agent.get_q_value(state, a) for a in range(2)]
            action = np.argmax(q_values)
            
            # 执行动作
            observation, reward, terminated, truncated, info = test_env.step(action)
            state = agent.discretize_state(observation)
            
            total_reward += reward
            step_count += 1
            
            # 在图形模式下显示进度
            if render and test_episode == 0 and step_count % 50 == 0:
                print(f"   步数: {step_count}, 当前奖励: {total_reward}")
        
        test_rewards.append(total_reward)
        
        # 判断结束原因
        if terminated:
            end_reason = "杆子倒下或小车超出边界"
        elif truncated:
            end_reason = "达到最大步数限制(500步)"
        else:
            end_reason = "未知原因"
        
        print(f"测试 {test_episode + 1}: {total_reward:3.0f} 步 - {end_reason}")
        
        test_env.close()
        
        # 在图形测试后稍作停顿
        if render and test_episode == 0:
            time.sleep(1)
    
    # 测试结果统计
    avg_reward = np.mean(test_rewards)
    max_reward = max(test_rewards)
    min_reward = min(test_rewards)
    success_rate = sum(1 for r in test_rewards if r >= 195) / len(test_rewards) * 100
    
    print(f"\n📈 测试结果统计：")
    print(f"   - 平均表现: {avg_reward:.2f} 步")
    print(f"   - 最佳表现: {max_reward} 步")
    print(f"   - 最差表现: {min_reward} 步")
    print(f"   - 成功率: {success_rate:.1f}% (≥195步)")
    print("=" * 60)
    
    return test_rewards


def main():
    """主函数：演示Q-learning算法在CartPole环境中的应用"""
    print("🎯 CartPole Q-learning 智能体训练与测试")
    print("=" * 80)
    
    # 检查是否存在已保存的Q表
    q_table_path = "models/cartpole_q_table.pkl"
    load_existing = False
    
    if os.path.exists(q_table_path):
        user_input = input(f"发现已保存的Q表文件: {q_table_path}\n是否加载已有模型？(y/n): ").lower().strip()
        load_existing = user_input in ['y', 'yes', '是']
    
    if load_existing:
        # 加载已有模型
        agent = QLearningAgent()
        if agent.load_q_table(q_table_path):
            print("🔄 使用已加载的模型进行测试...")
            test_rewards = test_agent(agent, render=True, num_tests=5)
        else:
            print("❌ 加载失败，将重新训练...")
            load_existing = False
    
    if not load_existing:
        # 训练新模型
        agent, training_rewards = train_agent(num_episodes=2000)
        
        # 显示训练过程可视化
        print(f"\n📈 生成训练过程可视化图表...")
        plot_training_progress(training_rewards, agent)
        
        # 保存训练好的Q表
        print(f"\n💾 保存训练好的Q表...")
        agent.save_q_table(q_table_path)
        
        # 测试智能体
        test_rewards = test_agent(agent, render=True, num_tests=5)
        
        # 显示学习曲线信息
        stats = agent.get_performance_stats(training_rewards)
        print(f"\n📊 训练统计信息:")
        print(f"   - 总回合数: {stats['total_episodes']}")
        print(f"   - 平均奖励: {stats['mean_reward']:.2f}")
        print(f"   - 最大奖励: {stats['max_reward']:.2f}")
        print(f"   - 成功率: {stats['success_rate']:.1f}%")
        print(f"   - Q表大小: {agent.get_q_table_size()} 个状态")
        
        if 'improvement' in stats:
            print(f"   - 学习改进: {stats['improvement']:.2f} 步")
    
    print(f"\n🎉 程序执行完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()