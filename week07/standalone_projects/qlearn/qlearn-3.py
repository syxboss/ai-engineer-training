import gymnasium as gym
# 简单策略：根据杆子倾斜方向移动小车

# 创建CartPole环境
env = gym.make('CartPole-v1', render_mode='human')

# 重置环境 - 新版本的gymnasium返回(observation, info)
observation, info = env.reset()

total_reward = 0
terminated = False
truncated = False

print("开始CartPole演示...")
print(f"初始观察: {observation}")

while not (terminated or truncated):
    # 解包观察值：[车位置, 车速度, 杆角度, 杆角速度]
    cart_position, cart_velocity, pole_angle, pole_velocity = observation

    # 简单策略：根据杆子倾斜方向移动小车
    if pole_angle > 0:  # 杆子向右倾斜
        action = 1      # 向右移动小车
    else:               # 杆子向左倾斜
        action = 0      # 向左移动小车

    # 执行动作 - 新版本返回5个值
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # 显示当前状态
    print(f"动作: {'右' if action == 1 else '左'}, 杆角度: {pole_angle:.3f}, 奖励: {reward}, 总奖励: {total_reward}")

print(f"回合结束，总奖励: {total_reward}")
if terminated:
    print("结束原因: 杆子倒下或小车超出边界")
else:
    print("结束原因: 达到最大步数限制")

env.close()