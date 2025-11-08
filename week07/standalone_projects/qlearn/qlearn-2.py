# 不做任何操作
"""
CartPole 小车平衡杆环境演示
========================

CartPole是一个经典的强化学习环境，目标是通过左右移动小车来保持杆子平衡。

环境说明：
- 观察空间：4维连续空间 [车位置, 车速度, 杆角度, 杆角速度]
- 动作空间：2个离散动作 [向左推车, 向右推车]
- 目标：保持杆子直立，不让它倒下
- 奖励：每个时间步获得+1奖励，杆子倒下或车离开边界时结束
"""

import gymnasium as gym
import time
import numpy as np

def run_cartpole_demo():
    """
    运行CartPole环境演示，显示图形界面
    """
    print("=" * 50)
    print("CartPole 小车平衡杆环境演示")
    print("=" * 50)
    
    # 创建环境并启用图形渲染
    print("正在创建CartPole环境...")
    env = gym.make('CartPole-v1', render_mode='human')
    
    # 重置环境，获取初始观察
    observation, info = env.reset()
    print(f"初始观察: {observation}")
    print(f"  - 车位置: {observation[0]:.3f}")
    print(f"  - 车速度: {observation[1]:.3f}")
    print(f"  - 杆角度: {observation[2]:.3f} 弧度")
    print(f"  - 杆角速度: {observation[3]:.3f}")
    
    # 打印环境信息
    print(f"\n动作空间: {env.action_space}")
    print("  - 0: 向左推车")
    print("  - 1: 向右推车")
    print(f"观察空间: {env.observation_space}")
    print("  - [车位置, 车速度, 杆角度, 杆角速度]")
    
    print("\n开始演示...")
    print("小车将使用随机策略尝试保持杆子平衡")
    print("按 Ctrl+C 可以提前结束演示")
    
    try:
        episode = 1
        while True:
            print(f"\n--- 第 {episode} 回合 ---")
            observation, info = env.reset()
            total_reward = 0
            step = 0
            
            while True:
                # 渲染环境（显示图形）
                env.render()
                
                # 简单的控制策略：根据杆子的角度选择动作
                # 如果杆子向右倾斜，向右推车；如果向左倾斜，向左推车
                pole_angle = observation[2]
                if pole_angle > 0:
                    action = 1  # 向右推车
                else:
                    action = 0  # 向左推车
                
                # 也可以使用完全随机策略：
                # action = env.action_space.sample()
                
                # 执行动作
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                # 显示当前状态信息
                if step % 10 == 0:  # 每10步显示一次
                    print(f"  步骤 {step}: 动作={'右' if action == 1 else '左'}, "
                          f"杆角度={observation[2]:.3f}, 总奖励={total_reward}")
                
                # 添加小延迟，让动画更容易观察
                time.sleep(0.02)
                
                # 检查是否结束
                if terminated or truncated:
                    print(f"  回合结束！总步数: {step}, 总奖励: {total_reward}")
                    if terminated:
                        print("  结束原因: 杆子倒下或小车超出边界")
                    else:
                        print("  结束原因: 达到最大步数限制")
                    break
            
            episode += 1
            
            # 每回合之间稍作停顿
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
    
    finally:
        # 关闭环境
        env.close()
        print("环境已关闭")

if __name__ == '__main__':
    run_cartpole_demo()