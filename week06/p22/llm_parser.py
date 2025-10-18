#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通义千问大模型解析器模块
使用通义千问大模型解析用户需求，提取关键槽位信息
"""

import json
import os
import re
from typing import Dict, Any
from dataclasses import dataclass
import dashscope
from dashscope import Generation


@dataclass
class CoffeeParameters:
    """咖啡制作参数数据类"""
    temperature: int = 90
    heating_time: int = 20
    extraction_strength: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'TEMPERATURE': self.temperature,
            'TEMPERATURE_CHECK': self.temperature - 1,
            'HEATING_TIME': self.heating_time,
            'EXTRACTION_STRENGTH': self.extraction_strength
        }


class QwenParser:
    """通义千问解析器"""
    
    def __init__(self, api_key: str = None):
        """初始化通义千问解析器"""
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        dashscope.api_key = self.api_key
        
        # 定义通义千问提示词
        self.system_prompt = """你是一个专业的咖啡制作参数提取专家。请从用户的自然语言输入中精确提取以下关键信息：

**参数说明：**
1. **温度 (temperature)**: 水温度数，单位摄氏度
   - 范围：80-100°C
   - 默认值：90°C
   - 识别关键词：度、°C、温度、热水、水温

2. **加热时间 (heating_time)**: 萃取时间，单位秒
   - 范围：10-60秒
   - 默认值：20秒
   - 识别关键词：秒、时间、萃取时间、加热时间

3. **萃取强度 (extraction_strength)**: 咖啡浓度
   - 可选值：light(轻度)、medium(中度)、strong(重度)
   - 默认值：medium
   - 识别关键词：轻度/淡、中度/普通/正常、重度/浓/强

**提取规则：**
- 如果用户没有明确指定某个参数，使用默认值
- 温度必须在80-100范围内，超出范围调整到边界值
- 时间必须在10-60范围内，超出范围调整到边界值
- 萃取强度必须是light/medium/strong之一

**输出格式：**
请严格按照以下JSON格式输出，不要包含任何其他文字：
{
  "temperature": 数字,
  "heating_time": 数字,
  "extraction_strength": "字符串"
}

**示例：**
用户输入："帮我做一杯90度加热20秒的轻度萃取"
输出：{"temperature": 90, "heating_time": 20, "extraction_strength": "light"}"""
    
    def parse_user_input(self, user_input: str) -> CoffeeParameters:
        """使用通义千问解析用户输入"""
        response = Generation.call(
            model='qwen-turbo',
            prompt=f"{self.system_prompt}\n\n用户输入：{user_input}",
            max_tokens=200,
            temperature=0.1
        )
        
        content = response.output.text.strip()
        print(f"通义千问响应: {content}")
        
        # 提取JSON部分
        json_match = re.search(r'\{[^}]+\}', content)
        params_dict = json.loads(json_match.group())
        
        params = CoffeeParameters(
            temperature=params_dict.get('temperature', 90),
            heating_time=params_dict.get('heating_time', 20),
            extraction_strength=params_dict.get('extraction_strength', 'medium')
        )
        
        # 参数验证和调整
        params.temperature = max(80, min(100, params.temperature))
        params.heating_time = max(10, min(60, params.heating_time))
        if params.extraction_strength not in ['light', 'medium', 'strong']:
            params.extraction_strength = 'medium'
        
        print(f"通义千问解析成功: {params}")
        return params


def create_parser() -> QwenParser:
    """创建通义千问解析器"""
    return QwenParser()


if __name__ == "__main__":
    # 测试解析器
    parser = QwenParser()
    
    test_cases = [
        "帮我做一杯90度加热20秒的轻度萃取",
        "我要一杯95度的重度萃取咖啡，加热30秒",
        "制作一杯85度中度萃取15秒的咖啡",
        "来一杯普通咖啡"
    ]
    
    for case in test_cases:
        print(f"\n测试输入: {case}")
        result = parser.parse_user_input(case)
        print(f"解析结果: {result.to_dict()}")