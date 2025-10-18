# 咖啡机 DSL 规则修改工作流


## 工作流程

1. **意图识别模块** → 识别为 modify_rule 操作
2. **DSL 更新** → 将当前 DSL 规则和用户指令传递给 tongyi 大模型
3. **语法验证** → 使用 Lark 进行语法验证，确认语法正确
4. **应用更改** → 系统应用新的规则并升级版本号
5. **返回确认** → 向用户返回确认信息

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

设置通义千问 API 密钥：

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

## 使用方法

### 演示模式

```python
from coffee import CoffeeDSLWorkflow

# 创建工作流实例（演示模式）
workflow = CoffeeDSLWorkflow(demo_mode=True)

# 处理用户指令
result = workflow.process_user_input("把加热时间从30秒改成20秒")
print(result)  # 输出：已更新加热时间为20秒
```

### 生产模式

```python
from coffee import CoffeeDSLWorkflow

# 创建工作流实例
workflow = CoffeeDSLWorkflow(demo_mode=False)

# 处理用户指令
result = workflow.process_user_input("把加热时间从30秒改成20秒")
print(result)
```

### 直接运行

```bash
python coffee.py
```

## 文件结构

- `coffee.py` - 主要的工作流实现
- `coffee_dsl.lark` - DSL 语法定义文件
- `coffee_rules.dsl` - 咖啡机规则配置文件
- `lark_parser.py` - Lark 解析器实现
- `requirements.txt` - 项目依赖

## 支持的操作类型

目前支持以下类型的修改指令：

- 时间参数修改：如"把加热时间从30秒改成20秒"
- 温度参数修改：如"将目标温度调整为95°C"
- 其他数值参数修改
