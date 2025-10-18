#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
咖啡机 DSL 工作流
基于 LangGraph 实现，使用 tongyi 大模型进行规则更新
"""

import os
import re
from typing import Dict, Any
from typing_extensions import TypedDict

# LangGraph 相关导入
from langgraph.graph import StateGraph, END

# 通义千问 API
import dashscope
from dashscope import Generation

# Lark 解析器
from lark_parser import parse

# Step 1: 定义状态结构
class State(TypedDict):
    """LangGraph 状态定义"""
    user_input: str  # 用户原始输入
    intent: str      
    current_dsl: str # 当前 DSL 规则
    updated_dsl: str # 更新后的 DSL 规则
    validation_result: bool # 语法验证结果
    final_message: str # 最终回复消息



class CoffeeDSLWorkflow:
    """咖啡机 DSL 工作流管理器"""
    
    def __init__(self):
        """初始化工作流"""
        # 设置通义千问 API
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
        
        # 加载当前 DSL 规则
        self.dsl_file_path = "./coffee_rules.dsl"
        
        # 构建 LangGraph 工作流
        self.workflow = self._build_workflow()
    
    # Step 7: 构建 LangGraph 工作流
    def _build_workflow(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(State)
        
        # 添加节点
        workflow.add_node("intent_recognition", self.intent_recognition_node)
        workflow.add_node("dsl_update", self.dsl_update_node)
        workflow.add_node("syntax_validation", self.syntax_validation_node)
        workflow.add_node("apply_changes", self.apply_changes_node)
        
        # 设置入口点
        workflow.set_entry_point("intent_recognition")
        
        # 添加边
        workflow.add_edge("intent_recognition", "dsl_update")
        workflow.add_edge("dsl_update", "syntax_validation")
        workflow.add_edge("syntax_validation", "apply_changes")
        workflow.add_edge("apply_changes", END)
        
        return workflow.compile()
    
    # Step 2: 意图识别节点
    def intent_recognition_node(self, state: State) -> Dict[str, Any]:
        """意图识别节点"""
        user_input = state["user_input"]
        
        # 意图识别规则(基于关键词匹配)
        modify_keywords = ["改", "修改", "更新", "调整", "设置"]
        time_keywords = ["时间", "秒", "分钟", "s", "min"]
        
        if any(keyword in user_input for keyword in modify_keywords) and \
           any(keyword in user_input for keyword in time_keywords):
            intent = "modify_rule"
        else:
            intent = "unknown"
        
        # 加载当前 DSL
        with open(self.dsl_file_path, 'r', encoding='utf-8') as f:
            current_dsl = f.read()
        
        return {
            **state,
            "intent": intent,
            "current_dsl": current_dsl
        }
    
    # Step 3: DSL 更新节点 - 使用通义千问
    def dsl_update_node(self, state: State) -> Dict[str, Any]:
        """DSL 更新节点 - 使用通义千问"""
        # 构建提示词
        prompt = f"""
            你是一个咖啡机 DSL 规则更新助手。请根据用户指令更新以下 DSL 规则。

            当前 DSL 规则：
            ```
            {state["current_dsl"]}
            ```

            用户指令：{state["user_input"]}

            请按照以下要求更新 DSL：
            1. 保持原有的结构和语法
            2. 只修改相关的参数值
            3. 将版本号从 1.0 升级到 1.1
            4. 确保语法正确

            请直接返回更新后的完整 DSL 代码，不要添加任何解释或格式化标记。
            """
        
        # 调用通义千问 API
        response = Generation.call(
            model='qwen-turbo',
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1
        )
        
        print(response.output.text)
        
        updated_dsl = response.output.text.strip()
        # 清理可能的代码块标记
        updated_dsl = re.sub(r'^```[\w]*\n?', '', updated_dsl)
        updated_dsl = re.sub(r'\n?```$', '', updated_dsl)
        
        return {
            **state,
            "updated_dsl": updated_dsl
        }
    
    # Step 4: 语法验证节点 - 使用 Lark
    def syntax_validation_node(self, state: State) -> Dict[str, Any]:
        """语法验证节点 - 使用 Lark"""
        # 使用现有的解析器验证语法
        result = parse(state["updated_dsl"])
        
        return {
            **state,
            "validation_result": True
        }
    
    # Step 5: 应用更改节点
    def apply_changes_node(self, state: State) -> Dict[str, Any]:
        """应用更改节点"""
        # 备份原文件
        backup_path = self.dsl_file_path + ".backup"
        with open(self.dsl_file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # 写入新的 DSL 规则
        with open(self.dsl_file_path, 'w', encoding='utf-8') as f:
            f.write(state["updated_dsl"])
        
        # 提取更新的时间信息
        time_match = re.search(r'改[成为](\d+)秒', state["user_input"])
        if time_match:
            new_time = time_match.group(1)
            final_message = f"已更新加热时间为{new_time}秒"
        else:
            time_match = re.search(r'(\d+)秒', state["user_input"])
            if time_match:
                new_time = time_match.group(1)
                final_message = f"已更新加热时间为{new_time}秒"
            else:
                final_message = "DSL 规则已成功更新"
        
        return {
            **state,
            "final_message": final_message
        }
    
    # Step 6: 处理用户输入的主入口
    def process_user_input(self, user_input: str) -> str:
        """处理用户输入的主入口"""
        initial_state = {
            "user_input": user_input,
            "intent": "",
            "current_dsl": "",
            "updated_dsl": "",
            "validation_result": False,
            "final_message": ""
        }
        
        # 运行工作流
        result = self.workflow.invoke(initial_state)
        return result["final_message"]


def main():
    # 初始化工作流
    workflow = CoffeeDSLWorkflow()
    
    # 测试用例
    user_instruction = "把加热时间从30秒改成20秒"
    result = workflow.process_user_input(user_instruction)
    print(f"处理结果: {result}")


if __name__ == "__main__":
    main()