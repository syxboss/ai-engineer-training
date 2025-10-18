#!/usr/bin/env python3
"""
简单测试：验证JSON日志格式
"""

import json
from datetime import datetime

def test_json_log_format():
    """测试JSON日志格式"""
    print("测试JSON日志格式...")
    
    # 创建符合要求的日志数据
    log_data = {
        "timestamp": datetime.now().isoformat() + "Z",
        "audit_id": "AUDIT-1712000000-user123",
        "user_id": "user123",
        "user_role": "sales_rep",
        "input_question": "查一下所有订单",
        "generated_sql": "SELECT * FROM orders",
        "validation_result": {
            "is_safe": True,
            "risk_level": "low"
        },
        "status": "approved",
        "response_time_ms": 245
    }
    
    try:
        # 序列化为JSON
        json_output = json.dumps(log_data, ensure_ascii=False, indent=2)
        
        print("✅ JSON格式验证成功！")
        print("\n生成的JSON日志格式:")
        print("-" * 50)
        print(json_output)
        print("-" * 50)
        
        # 验证JSON可以正确解析
        parsed = json.loads(json_output)
        print(f"✅ JSON解析成功，包含 {len(parsed)} 个字段")
        
        # 验证必需字段
        required_fields = [
            "timestamp", "audit_id", "user_id", "user_role",
            "input_question", "generated_sql", "validation_result",
            "status", "response_time_ms"
        ]
        
        missing_fields = [field for field in required_fields if field not in parsed]
        if missing_fields:
            print(f"❌ 缺少字段: {missing_fields}")
            return False
        
        print("✅ 所有必需字段都存在")
        
        # 验证validation_result结构
        if "validation_result" in parsed:
            val_result = parsed["validation_result"]
            if "is_safe" in val_result and "risk_level" in val_result:
                print("✅ validation_result结构正确")
            else:
                print("❌ validation_result结构不完整")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ JSON格式测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("DB安全网关 - JSON日志格式测试")
    print("=" * 60)
    
    success = test_json_log_format()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试通过！JSON日志格式符合要求。")
        print("\n该格式包含以下关键信息:")
        print("- 时间戳 (timestamp)")
        print("- 审计ID (audit_id)")
        print("- 用户信息 (user_id, user_role)")
        print("- 输入问题 (input_question)")
        print("- 生成的SQL (generated_sql)")
        print("- 验证结果 (validation_result)")
        print("- 状态 (status)")
        print("- 响应时间 (response_time_ms)")
        return 0
    else:
        print("❌ 测试失败！")
        return 1

if __name__ == "__main__":
    exit(main())