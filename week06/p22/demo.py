#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
咖啡机 DSL Agent 系统演示
展示不同用户输入的处理结果
"""

from coffee_agent import CoffeeDSLAgent


def demo_different_inputs():
    """演示不同用户输入的处理"""
    agent = CoffeeDSLAgent()
    
    # 测试用例
    test_cases = [
        "帮我做一杯90度加热20秒的轻度萃取",
        "我要一杯95度的重度萃取咖啡，加热30秒",
        "制作一杯85度中度萃取15秒的咖啡",
        "来一杯普通咖啡"  # 使用默认参数
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试用例 {i}: {user_input}")
        print('='*60)
        
        result = agent.process_user_request(user_input)
        
        if result["success"]:
            print(f"\n处理成功: {result['message']}")
        else:
            print(f"\n处理失败: {result['message']}")


def interactive_mode():
    """交互模式"""
    agent = CoffeeDSLAgent()
    
    print("\n欢迎使用咖啡机 DSL Agent 系统!")
    print("您可以输入类似以下的指令:")
    print("   - 帮我做一杯90度加热20秒的轻度萃取")
    print("   - 我要一杯95度的重度萃取咖啡，加热30秒")
    print("   - 制作一杯85度中度萃取15秒的咖啡")
    print("   - 输入 'quit' 退出")
    print("-" * 50)
    
    while True:
        user_input = input("\n请输入您的咖啡制作需求: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("👋 感谢使用，再见!")
            break
        
        if not user_input:
            print("请输入有效的指令")
            continue
        
        print(f"\n{'='*50}")
        result = agent.process_user_request(user_input)
        
        if result["success"]:
            print(f"\n咖啡制作完成!")
            if "execution_result" in result and result["execution_result"]["success"]:
                execution_result = result["execution_result"]
                params = execution_result["parameters"]
                print(f"制作参数: 温度{params['temperature']}°C, 时间{params['heating_time']}秒, 强度{params['extraction_strength']}")
        else:
            print(f"\n处理失败: {result['message']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        demo_different_inputs()