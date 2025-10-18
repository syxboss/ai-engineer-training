#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
咖啡机 DSL Agent 系统
实现模板加载、解析、渲染和验证功能
"""

import os
import re
import json
from typing import Dict, Any, Optional
from jinja2 import Template
from lark_parser import parse
from llm_parser import QwenParser, CoffeeParameters


class CoffeeDSLAgent:
    """咖啡机DSL智能代理"""
    
    def __init__(self, template_path: str = "coffee.dsl.tpl", api_key: str = None):
        """初始化代理"""
        self.template_path = template_path
        self.template = None
        
        # 初始化通义千问解析器
        self.llm_parser = QwenParser(api_key=api_key)
        
        # 加载模板
        self.load_template()
    
    def load_template(self):
        """加载DSL模板"""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            self.template = Template(template_content)
            print(f"✓ 模板加载成功: {self.template_path}")
        except FileNotFoundError:
            print(f"✗ 模板文件未找到: {self.template_path}")
            raise
        except Exception as e:
            print(f"✗ 模板加载失败: {e}")
            raise
    
    def parse_user_input(self, user_input: str) -> dict:
        """使用通义千问解析用户输入"""
        print(f"解析用户输入: {user_input}")
        
        # 使用通义千问解析器
        coffee_params = self.llm_parser.parse_user_input(user_input)
        
        # 转换为字典格式
        params = coffee_params.to_dict()
        print(f"解析完成，提取参数: {params}")
        
        return params
    
    def render_template(self, params: dict) -> str:
        """渲染DSL模板"""
        print(f"渲染模板，参数: {params}")
        
        try:
            rendered_dsl = self.template.render(**params)
            print("✓ 模板渲染成功")
            return rendered_dsl
        except Exception as e:
            print(f"✗ 模板渲染失败: {e}")
            raise
    
    def validate_dsl(self, dsl_content: str) -> bool:
        """验证DSL语法"""
        print("验证DSL语法...")
        
        try:
            # 使用现有的解析器验证语法
            result = parse(dsl_content)
            print("✓ DSL语法验证通过")
            return True
        except Exception as e:
            print(f"✗ DSL语法验证失败: {e}")
            return False
    
    def save_dsl(self, dsl_content: str, output_path: str = "coffee_rules_generated.dsl") -> bool:
        """保存生成的DSL文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dsl_content)
            print(f"✓ DSL已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"✗ 保存DSL文件失败: {e}")
            return False
    
    def simulate_execution(self, params: dict):
        """模拟执行咖啡制作过程"""
        print("\n开始模拟咖啡制作过程:")
        print(f"   温度: {params['TEMPERATURE']}°C")
        print(f"   加热时间: {params['HEATING_TIME']}秒")
        print(f"   萃取强度: {params['EXTRACTION_STRENGTH']}")
        print("   咖啡制作完成！")
    
    def process_user_request(self, user_input: str) -> dict:
        """处理用户请求的完整流程"""
        print(f"处理用户请求: {user_input}")
        print("=" * 50)
        
        try:
            # 1. 解析用户输入
            params = self.parse_user_input(user_input)
            
            # 2. 渲染模板
            rendered_dsl = self.render_template(params)
            
            # 3. 验证DSL语法
            if not self.validate_dsl(rendered_dsl):
                return {"success": False, "message": "DSL验证失败"}
            
            # 4. 保存生成的DSL
            self.save_dsl(rendered_dsl)
            
            # 5. 模拟执行
            self.simulate_execution(params)
            
            return {"success": True, "message": "咖啡制作完成"}
            
        except Exception as e:
            return {"success": False, "message": f"处理失败: {e}"}


def main():
    """主函数"""
    agent = CoffeeDSLAgent()
    
    # 测试用例
    user_input = "帮我做一杯90度加热20秒的轻度萃取"
    
    result = agent.process_user_request(user_input)
    
    if result["success"]:
        print(f"\n{result['message']}")
    else:
        print(f"\n处理失败: {result['message']}")


if __name__ == "__main__":
    main()