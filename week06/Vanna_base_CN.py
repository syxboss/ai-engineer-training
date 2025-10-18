r"""
# ============================================================================
# 代码阅读步骤指南 - Vanna AI Base 类
# ============================================================================
# 
# 【第一步：理解整体架构】
# 本文件是 Vanna AI 的核心基类 VannaBase，它是一个抽象基类(ABC)
# 主要功能：将自然语言问题转换为SQL查询，并生成可视化图表
# 
# 【第二步：掌握核心概念】
# - RAG (Retrieval-Augmented Generation): 检索增强生成
# - 训练数据包括：DDL语句、问题-SQL对、文档说明
# - 支持多种数据库和LLM模型的扩展
#
# 【第三步：了解方法命名规范】
# - vn.get_*: 获取数据 (如 get_related_ddl)
# - vn.add_*: 添加训练数据 (如 add_question_sql) 
# - vn.generate_*: 使用AI生成内容 (如 generate_sql)
# - vn.run_*: 执行代码 (如 run_sql)
# - vn.connect_*: 连接数据库 (如 connect_to_snowflake)
#
# 【第四步：核心工作流程】
# 1. 训练阶段：添加DDL、问题-SQL对、文档到向量数据库
# 2. 查询阶段：检索相关信息 → 构建提示词 → LLM生成SQL → 执行并可视化

# Nomenclature

| Prefix | Definition | Examples |
| --- | --- | --- |
| `vn.get_` | Fetch some data | [`vn.get_related_ddl(...)`][vanna.base.base.VannaBase.get_related_ddl] |
| `vn.add_` | Adds something to the retrieval layer | [`vn.add_question_sql(...)`][vanna.base.base.VannaBase.add_question_sql] <br> [`vn.add_ddl(...)`][vanna.base.base.VannaBase.add_ddl] |
| `vn.generate_` | Generates something using AI based on the information in the model | [`vn.generate_sql(...)`][vanna.base.base.VannaBase.generate_sql] <br> [`vn.generate_explanation()`][vanna.base.base.VannaBase.generate_explanation] |
| `vn.run_` | Runs code (SQL) | [`vn.run_sql`][vanna.base.base.VannaBase.run_sql] |
| `vn.remove_` | Removes something from the retrieval layer | [`vn.remove_training_data`][vanna.base.base.VannaBase.remove_training_data] |
| `vn.connect_` | Connects to a database | [`vn.connect_to_snowflake(...)`][vanna.base.base.VannaBase.connect_to_snowflake] |
| `vn.update_` | Updates something | N/A -- unused |
| `vn.set_` | Sets something | N/A -- unused  |

# Open-Source and Extending

Vanna.AI is open-source and extensible. If you'd like to use Vanna without the servers, see an example [here](https://vanna.ai/docs/postgres-ollama-chromadb/).

The following is an example of where various functions are implemented in the codebase when using the default "local" version of Vanna. `vanna.base.VannaBase` is the base class which provides a `vanna.base.VannaBase.ask` and `vanna.base.VannaBase.train` function. Those rely on abstract methods which are implemented in the subclasses `vanna.openai_chat.OpenAI_Chat` and `vanna.chromadb_vector.ChromaDB_VectorStore`. `vanna.openai_chat.OpenAI_Chat` uses the OpenAI API to generate SQL and Plotly code. `vanna.chromadb_vector.ChromaDB_VectorStore` uses ChromaDB to store training data and generate embeddings.

If you want to use Vanna with other LLMs or databases, you can create your own subclass of `vanna.base.VannaBase` and implement the abstract methods.

```mermaid
flowchart
    subgraph VannaBase
        ask
        train
    end

    subgraph OpenAI_Chat
        get_sql_prompt
        submit_prompt
        generate_question
        generate_plotly_code
    end

    subgraph ChromaDB_VectorStore
        generate_embedding
        add_question_sql
        add_ddl
        add_documentation
        get_similar_question_sql
        get_related_ddl
        get_related_documentation
    end
```

"""

# ============================================================================
# 【第五步：导入模块分析】
# ============================================================================
# 标准库导入：处理JSON、文件操作、正则表达式、SQLite、异常追踪等
import json
import os
import re
import sqlite3
import traceback
# ABC相关：定义抽象基类和抽象方法
from abc import ABC, abstractmethod
# 类型提示：用于代码可读性和IDE支持
from typing import List, Tuple, Union
# URL解析：处理数据库连接字符串
from urllib.parse import urlparse

# 数据处理和可视化库
import pandas as pd          # 数据框架操作
import plotly               # 交互式图表库
import plotly.express as px # 快速图表创建
import plotly.graph_objects as go  # 底层图表对象
import requests             # HTTP请求处理
import sqlparse            # SQL语句解析

# Vanna内部模块
from ..exceptions import DependencyError, ImproperlyConfigured, ValidationError
from ..types import TrainingPlan, TrainingPlanItem
from ..utils import validate_config_path


# ============================================================================
# 【第六步：核心类定义 - VannaBase】
# ============================================================================
class VannaBase(ABC):
    """
    【阅读重点】Vanna AI的抽象基类
    
    核心职责：
    1. 定义统一的接口规范
    2. 提供通用的业务逻辑实现
    3. 管理配置和状态
    4. 协调各个子模块的工作
    
    【继承关系】
    - 继承自ABC（抽象基类）
    - 子类需要实现抽象方法来支持不同的LLM和向量数据库
    """
    
    def __init__(self, config=None):
        """
        【初始化方法】设置基本配置和状态
        
        参数说明：
        - config: 配置字典，包含方言、语言、最大token数等设置
        
        【阅读要点】
        1. 配置管理：使用字典存储各种配置项
        2. 状态跟踪：run_sql_is_set标记SQL执行器是否已设置
        3. 默认值设置：为关键参数提供合理的默认值
        """
        if config is None:
            config = {}

        self.config = config
        self.run_sql_is_set = False  # 标记SQL执行函数是否已设置
        self.static_documentation = ""  # 静态文档内容
        self.dialect = self.config.get("dialect", "SQL")  # SQL方言
        self.language = self.config.get("language", None)  # 响应语言
        self.max_tokens = self.config.get("max_tokens", 14000)  # 最大token数

    def log(self, message: str, title: str = "Info"):
        """
        【日志记录方法】统一的日志输出接口
        
        【设计模式】简单的日志门面模式
        - 提供统一的日志接口
        - 便于后续扩展到更复杂的日志系统
        """
        print(f"{title}: {message}")

    def _response_language(self) -> str:
        """
        【私有辅助方法】生成语言指令
        
        【用途】在提示词中添加语言要求
        - 如果设置了language，返回语言指令
        - 否则返回空字符串
        """
        if self.language is None:
            return ""

        return f"Respond in the {self.language} language."

    # ============================================================================
    # 【第七步：核心业务方法 - SQL生成】
    # ============================================================================
    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """
        【核心方法】将自然语言问题转换为SQL查询
        
        【工作流程详解】
        1. 获取初始提示词配置
        2. 检索相似的问题-SQL对 (RAG的R部分)
        3. 获取相关的DDL语句
        4. 获取相关的文档说明
        5. 构建完整的提示词
        6. 调用LLM生成SQL (RAG的G部分)
        7. 处理中间SQL查询（如需要）
        8. 提取并返回最终SQL
        
        【参数说明】
        - question: 用户的自然语言问题
        - allow_llm_to_see_data: 是否允许LLM查看数据内容
        - **kwargs: 其他可选参数
        
        【返回值】生成的SQL查询字符串
        
        Example:
        ```python
        vn.generate_sql("What are the top 10 customers by sales?")
        ```

        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        - [`get_similar_question_sql`][vanna.base.base.VannaBase.get_similar_question_sql]

        - [`get_related_ddl`][vanna.base.base.VannaBase.get_related_ddl]

        - [`get_related_documentation`][vanna.base.base.VannaBase.get_related_documentation]

        - [`get_sql_prompt`][vanna.base.base.VannaBase.get_sql_prompt]

        - [`submit_prompt`][vanna.base.base.VannaBase.submit_prompt]


        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        # 【步骤1】获取初始提示词配置
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
            
        # 【步骤2-4】RAG检索阶段：获取相关信息
        question_sql_list = self.get_similar_question_sql(question, **kwargs)  # 相似问题
        ddl_list = self.get_related_ddl(question, **kwargs)  # 相关表结构
        doc_list = self.get_related_documentation(question, **kwargs)  # 相关文档
        
        # 【步骤5】构建提示词
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=prompt)
        
        # 【步骤6】调用LLM生成响应
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        # 【步骤7】处理中间SQL查询（用于复杂查询的分步处理）
        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    # 使用中间结果重新构建提示词
                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list+[f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"


        # 【步骤8】提取最终SQL
        return self.extract_sql(llm_response)

    def extract_sql(self, llm_response: str) -> str:
        """
        【SQL提取方法】从LLM响应中提取SQL查询
        
        【处理的SQL格式】
        1. CREATE TABLE ... AS SELECT 语句
        2. WITH子句（CTE公用表表达式）
        3. 普通SELECT语句
        4. Markdown代码块中的SQL
        5. 普通代码块
        
        【正则表达式解析】
        - 使用re.DOTALL标志处理多行SQL
        - 使用re.IGNORECASE忽略大小写
        - 按优先级顺序匹配不同格式
        
        Example:
        ```python
        vn.extract_sql("Here's the SQL query in a code block: ```sql\nSELECT * FROM customers\n```")
        ```

        Extracts the SQL query from the LLM response. This is useful in case the LLM response contains other information besides the SQL query.
        Override this function if your LLM responses need custom extraction logic.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted SQL query.
        """

        import re
        """
        Extracts the SQL query from the LLM response, handling various formats including:
        - WITH clause
        - SELECT statement
        - CREATE TABLE AS SELECT
        - Markdown code blocks
        """

        # 【优先级1】匹配 CREATE TABLE ... AS SELECT 语句
        sqls = re.findall(r"\bCREATE\s+TABLE\b.*?\bAS\b.*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]  # 取最后一个匹配项
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # 【优先级2】匹配 WITH 子句 (CTEs - 公用表表达式)
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # 【优先级3】匹配 SELECT ... ; 语句
        sqls = re.findall(r"\bSELECT\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # 【优先级4】匹配 ```sql ... ``` 代码块
        sqls = re.findall(r"```sql\s*\n(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1].strip()
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # 【优先级5】匹配任意 ``` ... ``` 代码块
        sqls = re.findall(r"```(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1].strip()
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # 【兜底策略】如果都没匹配到，返回原始响应
        return llm_response

    def is_sql_valid(self, sql: str) -> bool:
        """
        【SQL验证方法】检查SQL查询是否有效
        
        【验证逻辑】
        - 使用sqlparse库解析SQL语句
        - 默认只允许SELECT语句执行
        - 可以重写此方法来支持其他类型的SQL
        
        【安全考虑】
        - 防止执行危险的DDL/DML语句
        - 提供可扩展的验证机制
        
        Example:
        ```python
        vn.is_sql_valid("SELECT * FROM customers")
        ```
        Checks if the SQL query is valid. This is usually used to check if we should run the SQL query or not.
        By default it checks if the SQL query is a SELECT statement. You can override this method to enable running other types of SQL queries.

        Args:
            sql (str): The SQL query to check.

        Returns:
            bool: True if the SQL query is valid, False otherwise.
        """

        parsed = sqlparse.parse(sql)

        for statement in parsed:
            if statement.get_type() == 'SELECT':
                return True

        return False

    def should_generate_chart(self, df: pd.DataFrame) -> bool:
        """
        【图表生成判断】确定是否应该为数据生成图表
        
        【判断条件】
        1. 数据框有多于1行的数据
        2. 包含数值型列
        
        【设计思路】
        - 避免为空数据或单行数据生成图表
        - 确保有可视化的数值数据
        
        Example:
        ```python
        vn.should_generate_chart(df)
        ```

        Checks if a chart should be generated for the given DataFrame. By default, it checks if the DataFrame has more than one row and has numerical columns.
        You can override this method to customize the logic for generating charts.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Returns:
            bool: True if a chart should be generated, False otherwise.
        """

        if len(df) > 1 and df.select_dtypes(include=['number']).shape[1] > 0:
            return True

        return False

    def generate_rewritten_question(self, last_question: str, new_question: str, **kwargs) -> str:
        """
        【问题重写方法】结合上下文重写用户问题
        
        【核心功能】
        - 将新问题与上一个问题结合
        - 处理对话上下文的连续性
        - 生成更完整的问题描述
        
        【应用场景】
        - 用户追问："显示他们的邮箱地址"
        - 需要结合前一个问题的上下文
        - 生成完整的问题："显示销售额前5的客户的邮箱地址"
        
        【实现策略】
        - 使用LLM判断问题是否相关
        - 如果相关则合并，否则返回新问题
        
        **Example:**
        ```python
        rewritten_question = vn.generate_rewritten_question("Who are the top 5 customers by sales?", "Show me their email addresses")
        ```

        Generate a rewritten question by combining the last question and the new question if they are related. If the new question is self-contained and not related to the last question, return the new question.

        Args:
            last_question (str): The previous question that was asked.
            new_question (str): The new question to be combined with the last question.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The combined question if related, otherwise the new question.
        """
        # 【边界条件处理】如果没有上一个问题，直接返回新问题
        if last_question is None:
            return new_question

        # 【构建提示词】让LLM判断两个问题是否相关并合并
        prompt = [
            self.system_message("Your goal is to combine a sequence of questions into a singular question if they are related. If the second question does not relate to the first question and is fully self-contained, return the second question. Return just the new combined question with no additional explanations. The question should theoretically be answerable with a single SQL statement."),
            self.user_message("First question: " + last_question + "\nSecond question: " + new_question),
        ]

        # 【调用LLM】生成重写后的问题
        return self.submit_prompt(prompt=prompt, **kwargs)

    def generate_followup_questions(
        self, question: str, sql: str, df: pd.DataFrame, n_questions: int = 5, **kwargs
    ) -> list:
        """
        【后续问题生成】基于查询结果生成相关的后续问题
        
        【核心功能】
        - 分析当前问题、SQL和结果数据
        - 生成用户可能感兴趣的后续问题
        - 提供数据探索的引导
        
        【生成策略】
        1. 基于当前SQL的变体（如添加过滤条件）
        2. 深入挖掘数据的不同维度
        3. 确保每个问题都能生成明确的SQL
        
        【应用场景】
        - 用户查询"销售额前10的客户"后
        - 可能想了解"这些客户的地理分布"
        - 或者"他们的购买频率如何"
        
        **Example:**
        ```python
        vn.generate_followup_questions("What are the top 10 customers by sales?", sql, df)
        ```

        Generate a list of followup questions that you can ask Vanna.AI.

        Args:
            question (str): The question that was asked.
            sql (str): The LLM-generated SQL query.
            df (pd.DataFrame): The results of the SQL query.
            n_questions (int): Number of follow-up questions to generate.

        Returns:
            list: A list of followup questions that you can ask Vanna.AI.
        """

        # 【构建上下文消息】包含原问题、SQL和结果数据
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe SQL query for this question was: {sql}\n\nThe following is a pandas DataFrame with the results of the query: \n{df.head(25).to_markdown()}\n\n"
            ),
            self.user_message(
                f"Generate a list of {n_questions} followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions. Remember that there should be an unambiguous SQL query that can be generated from the question. Prefer questions that are answerable outside of the context of this conversation. Prefer questions that are slight modifications of the SQL query that was generated that allow digging deeper into the data. Each question will be turned into a button that the user can click to generate a new SQL query so don't use 'example' type questions. Each question must have a one-to-one correspondence with an instantiated SQL query." +
                self._response_language()
            ),
        ]

        # 【调用LLM生成问题】
        llm_response = self.submit_prompt(message_log, **kwargs)

        # 【清理格式】移除编号，按行分割
        numbers_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return numbers_removed.split("\n")

    def generate_questions(self, **kwargs) -> List[str]:
        """
        【通用问题生成】生成用户可能询问的常见问题列表
        
        【核心功能】
        - 基于训练数据生成示例问题
        - 帮助用户了解系统能力
        - 提供问题模板和灵感
        
        【应用场景】
        - 用户首次使用系统
        - 不知道可以问什么问题
        - 需要问题示例作为参考
        
        **Example:**
        ```python
        vn.generate_questions()
        ```

        Generate a list of questions that you can ask Vanna.AI.
        """
        question_sql = self.get_similar_question_sql(question="", **kwargs)

        return [q["question"] for q in question_sql]

    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        """
        **Example:**
        ```python
        vn.generate_summary("What are the top 10 customers by sales?", df)
        ```

        Generate a summary of the results of a SQL query.

        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.

        Returns:
            str: The summary of the results of the SQL query.
        """

        # 【构建摘要提示】包含问题和查询结果
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary." +
                self._response_language()
            ),
        ]

        # 【生成摘要】调用LLM生成结果摘要
        summary = self.submit_prompt(message_log, **kwargs)

        return summary

    # ================= 嵌入向量接口 ================= #
    # 【抽象方法】子类必须实现具体的嵌入向量生成逻辑
    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        """
        【嵌入向量生成】将文本转换为向量表示
        
        【核心作用】
        - 支持语义相似度搜索
        - 为RAG检索提供向量基础
        - 实现问题-SQL的语义匹配
        
        【实现要求】
        - 子类必须实现具体的嵌入模型调用
        - 返回固定维度的浮点数向量
        - 支持批量处理以提高效率
        """
        pass

    # ================= 数据库存储和检索接口 ================= #
    # 【RAG核心组件】以下方法实现检索增强生成的数据存储和检索
    
    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        【相似问题检索】根据问题检索相似的历史问题和SQL
        
        【检索策略】
        1. 使用嵌入向量计算语义相似度
        2. 返回最相关的问题-SQL对
        3. 为新问题提供参考模板
        
        【应用场景】
        - 用户问"销售额最高的客户"
        - 检索到历史问题"营收最多的用户"
        - 复用对应的SQL模式
        
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        【相关DDL检索】根据问题检索相关的数据库表结构
        
        【检索逻辑】
        1. 分析问题中的实体和概念
        2. 匹配相关的表、列、索引等DDL
        3. 为SQL生成提供schema上下文
        
        【重要性】
        - 确保生成的SQL使用正确的表名和列名
        - 理解表之间的关系和约束
        - 避免引用不存在的数据库对象
        
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        【相关文档检索】根据问题检索相关的业务文档
        
        【文档类型】
        - 业务规则说明
        - 数据字典定义
        - 计算逻辑文档
        - 领域知识说明
        
        【应用价值】
        - 理解业务术语的具体含义
        - 获取复杂计算的标准公式
        - 确保SQL符合业务逻辑
        
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        pass

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        【训练数据添加】将问题-SQL对添加到训练数据库
        
        【数据管理】
        1. 存储用户验证过的问题-SQL对
        2. 为向量检索建立索引
        3. 支持系统持续学习和改进
        
        【质量控制】
        - 只存储经过验证的高质量数据
        - 支持数据去重和更新
        - 维护数据的一致性和准确性
        
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        【DDL数据添加】将数据库表结构添加到知识库
        
        【Schema管理】
        1. 存储CREATE TABLE等DDL语句
        2. 建立表结构的向量索引
        3. 支持schema变更的版本管理
        
        【自动化集成】
        - 支持从数据库自动提取DDL
        - 检测schema变更并更新
        - 维护表关系的完整性
        
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        【文档数据添加】将业务文档添加到知识库
        
        【文档管理】
        1. 存储业务规则、数据字典等文档
        2. 建立文档内容的语义索引
        3. 支持文档的版本控制和更新
        
        【知识整合】
        - 将非结构化的业务知识结构化
        - 为SQL生成提供业务上下文
        - 支持复杂业务逻辑的理解
        
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        """
        【训练数据获取】获取所有存储的训练数据
        
        【数据管理】
        - 返回问题-SQL对、DDL、文档等所有数据
        - 支持数据的查看、审核和管理
        - 提供数据质量评估的基础
        
        【应用场景】
        - 系统管理员查看训练数据质量
        - 数据科学家分析系统性能
        - 支持数据的导出和备份
        
        Example:
        ```python
        vn.get_training_data()
        ```

        This method is used to get all the training data from the retrieval layer.

        Returns:
            pd.DataFrame: The training data.
        """
        pass

    @abstractmethod
    def remove_training_data(self, id: str, **kwargs) -> bool:
        """
        【训练数据删除】从知识库中删除指定的训练数据
        
        【数据维护】
        - 删除错误或过时的训练数据
        - 支持数据的清理和优化
        - 维护知识库的质量和准确性
        
        【安全考虑】
        - 提供数据删除的审计日志
        - 支持批量删除操作
        - 防止误删重要数据
        
        Example:
        ```python
        vn.remove_training_data(id="123-ddl")
        ```

        This method is used to remove training data from the retrieval layer.

        Args:
            id (str): The ID of the training data to remove.

        Returns:
            bool: True if the training data was removed, False otherwise.
        """
        pass

    # ================= 语言模型接口 ================= #
    # 【LLM抽象层】定义与不同语言模型的统一接口

    @abstractmethod
    def system_message(self, message: str) -> any:
        """
        【系统消息构造】创建系统级提示消息
        
        【作用】
        - 设置AI助手的角色和行为规范
        - 定义输出格式和约束条件
        - 提供上下文和背景信息
        """
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        """
        【用户消息构造】创建用户输入消息
        
        【作用】
        - 封装用户的问题和请求
        - 标准化用户输入格式
        - 支持多轮对话的上下文
        """
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> any:
        """
        【助手消息构造】创建AI助手的回复消息
        
        【作用】
        - 封装AI的历史回复
        - 支持多轮对话的上下文维护
        - 提供few-shot学习的示例
        """
        pass

    def str_to_approx_token_count(self, string: str) -> int:
        """
        【Token计数估算】粗略估算字符串的token数量
        
        【估算方法】
        - 使用简单的字符数除以4的方法
        - 适用于英文文本的快速估算
        - 用于控制提示词长度，避免超出模型限制
        
        【注意事项】
        - 这是一个粗略估算，实际token数可能有差异
        - 不同语言和编码方式会影响准确性
        - 建议在关键场景使用精确的tokenizer
        """
        return len(string) / 4

    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        """
        【DDL添加到提示词】将数据库表结构信息添加到提示词中
        
        【核心功能】
        1. 将相关的DDL语句添加到提示词
        2. 控制总token数不超过限制
        3. 为SQL生成提供必要的schema信息
        
        【Token管理策略】
        - 优先添加最相关的DDL
        - 动态控制添加的DDL数量
        - 确保不超出模型的上下文限制
        
        【格式化输出】
        - 使用标准的"===Tables"分隔符
        - 保持DDL的原始格式和结构
        - 便于LLM理解和解析
        """
        if len(ddl_list) > 0:
            initial_prompt += "\n===Tables \n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt

    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: list[str],
        max_tokens: int = 14000,
    ) -> str:
        """
        将文档信息添加到提示词中的方法
        
        功能说明:
        1. 将相关文档内容添加到初始提示词中，为LLM提供额外的上下文信息
        2. 通过token计数控制添加的文档数量，避免超出模型的token限制
        3. 按顺序添加文档，直到达到token限制为止
        
        应用场景:
        - 为SQL生成提供数据库文档说明
        - 添加业务规则和约束条件
        - 提供表结构和字段含义的详细说明
        
        设计特点:
        - 智能token管理: 动态计算token数量，防止超出限制
        - 渐进式添加: 按优先级顺序添加文档内容
        - 格式化输出: 使用标准的分隔符组织文档内容
        """
        if len(documentation_list) > 0:
            initial_prompt += "\n===Additional Context \n\n"

            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def add_sql_to_prompt(
        self, initial_prompt: str, sql_list: list[str], max_tokens: int = 14000
    ) -> str:
        """
        将问题-SQL对添加到提示词中的方法
        
        功能说明:
        1. 将历史的问题-SQL示例添加到提示词中，为LLM提供few-shot学习样本
        2. 通过token计数控制添加的示例数量，确保不超出模型限制
        3. 按顺序添加问题-SQL对，直到达到token限制
        
        应用场景:
        - 提供SQL生成的示例模板
        - 展示特定数据库的查询模式
        - 帮助LLM理解业务逻辑和查询风格
        
        设计特点:
        - Few-shot学习: 通过示例提高SQL生成质量
        - 动态token管理: 智能控制示例数量
        - 结构化格式: 清晰的问题-SQL配对格式
        """
        if len(sql_list) > 0:
            initial_prompt += "\n===Question-SQL Pairs\n\n"

            for question in sql_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(question["sql"])
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"

        return initial_prompt

    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        构建SQL生成提示词的核心方法
        
        功能说明:
        1. 整合多种上下文信息构建完整的SQL生成提示词
        2. 按优先级添加DDL、文档、示例SQL等信息
        3. 设置详细的响应指导原则和格式要求
        4. 构建消息日志用于与LLM交互
        
        核心流程:
        - 设置基础提示词(如果未提供)
        - 添加DDL信息(数据库表结构)
        - 添加静态文档和动态文档
        - 设置响应指导原则
        - 构建few-shot示例
        - 添加用户问题
        
        应用场景:
        - Text-to-SQL任务的提示词构建
        - 多轮对话中的上下文管理
        - 复杂查询的引导生成
        
        设计特点:
        - 模块化组装: 分步骤构建完整提示词
        - 智能token管理: 控制各部分内容长度
        - 标准化格式: 统一的提示词结构
        - 灵活配置: 支持自定义初始提示词
        
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. " + \
            "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, please explain why it can't be generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            f"6. Ensure that the output SQL is {self.dialect}-compliant and executable, and free of syntax errors. \n"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    def get_followup_questions_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ) -> list:
        """
        构建后续问题生成提示词的方法
        
        功能说明:
        1. 基于用户的初始问题和相关上下文，构建用于生成后续问题的提示词
        2. 整合DDL、文档和历史问题-SQL对，为LLM提供充分的上下文
        3. 引导LLM生成相关的后续问题，增强用户体验
        
        核心流程:
        - 设置初始问题上下文
        - 添加数据库结构信息(DDL)
        - 添加相关文档说明
        - 添加历史问题-SQL示例
        - 构建生成指令
        
        应用场景:
        - 智能问答系统的问题推荐
        - 数据探索的引导式查询
        - 用户交互体验的增强
        
        设计特点:
        - 上下文感知: 基于当前问题生成相关后续问题
        - 结构化输出: 要求每行一个问题的格式
        - 简洁指令: 明确要求只输出问题，不要解释
        """
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "Generate a list of followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions."
            )
        )

        return message_log

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        向语言模型提交提示词的抽象方法
        
        功能说明:
        1. 这是一个抽象方法，需要在具体的子类中实现
        2. 负责将构建好的提示词发送给LLM并获取响应
        3. 是整个Vanna系统与LLM交互的核心接口
        
        实现要求:
        - 子类必须实现此方法
        - 处理不同LLM的API调用细节
        - 管理认证、限流、错误处理等
        - 返回LLM的文本响应
        
        应用场景:
        - SQL生成任务的LLM调用
        - 问题生成和重写的LLM交互
        - 图表代码生成的LLM请求
        
        设计模式:
        - 抽象方法模式: 定义接口，延迟实现
        - 策略模式: 支持不同的LLM提供商
        - 适配器模式: 统一不同LLM的调用方式
        
        Example:
        ```python
        vn.submit_prompt(
            [
                vn.system_message("The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."),
                vn.user_message("What are the top 10 customers by sales?"),
            ]
        )
        ```

        This method is used to submit a prompt to the LLM.

        Args:
            prompt (any): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        """
        根据SQL语句生成对应业务问题的方法
        
        功能说明:
        1. 接收SQL查询语句，通过LLM推测其对应的业务问题
        2. 用于SQL到自然语言的反向转换
        3. 帮助理解和验证SQL查询的业务含义
        
        核心流程:
        - 构建系统提示词，指导LLM理解任务
        - 将SQL作为用户消息发送
        - 获取LLM生成的业务问题
        
        应用场景:
        - SQL查询的业务含义解释
        - 数据分析报告的问题生成
        - SQL查询的可读性增强
        
        设计特点:
        - 反向工程: 从SQL推导业务问题
        - 简洁输出: 只返回问题，不包含解释
        - 表名抽象: 避免在问题中直接引用表名
        """
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    def _extract_python_code(self, markdown_string: str) -> str:
        """
        从Markdown格式文本中提取Python代码的私有方法
        
        功能说明:
        1. 使用正则表达式从Markdown文本中提取Python代码块
        2. 支持带有python标识符的代码块和普通代码块
        3. 处理LLM生成的代码响应，去除格式标记
        
        核心流程:
        - 去除字符串首尾空白字符
        - 使用正则表达式匹配代码块模式
        - 提取第一个匹配的代码块
        - 如果没有找到代码块，返回原始字符串
        
        应用场景:
        - 处理LLM生成的Plotly代码
        - 清理代码响应的格式
        - 提取可执行的Python代码
        
        设计特点:
        - 正则表达式匹配: 灵活处理不同格式
        - 容错处理: 未找到代码块时返回原文
        - 优先级处理: 优先提取标记为python的代码块
        """
        # Strip whitespace to avoid indentation errors in LLM-generated code
        markdown_string = markdown_string.strip()

        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        """
        清理Plotly代码的私有方法
        
        功能说明:
        1. 移除Plotly代码中的fig.show()语句
        2. 确保生成的代码适合在不同环境中使用
        3. 避免在非交互式环境中执行显示命令
        
        处理逻辑:
        - 简单的字符串替换操作
        - 移除可能导致问题的显示命令
        - 保持代码的其他部分不变
        
        应用场景:
        - 准备用于嵌入的Plotly代码
        - 清理LLM生成的可视化代码
        - 适配不同的执行环境
        
        设计特点:
        - 简单有效: 直接字符串替换
        - 专用性强: 专门处理Plotly代码
        - 环境适配: 确保代码在各种环境中可用
        """
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        """
        生成Plotly可视化代码的方法
        
        功能说明:
        1. 基于问题、SQL查询和DataFrame元数据生成Plotly图表代码
        2. 通过LLM智能选择合适的图表类型和样式
        3. 自动处理单值数据的指示器显示
        
        核心流程:
        - 构建系统消息，描述数据上下文
        - 添加问题和SQL查询信息(如果提供)
        - 添加DataFrame的元数据信息
        - 设置代码生成指令
        - 提取和清理生成的Python代码
        
        参数说明:
        - question: 用户的原始问题，用于指导图表类型选择
        - sql: 生成数据的SQL查询，提供数据来源上下文
        - df_metadata: DataFrame的元数据信息，包含列名、数据类型等
        
        应用场景:
        - 自动化数据可视化
        - 智能图表推荐
        - 数据分析结果的可视化展示
        
        设计特点:
        - 智能图表选择: 根据数据特征选择合适的图表类型
        - 单值处理: 自动识别单值数据并使用指示器
        - 代码清理: 自动提取和清理生成的代码
        - 灵活配置: 支持多种参数组合
        """
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate the Python plotly code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    # ----------------- 连接各种数据库以执行生成的SQL ----------------- #

    def connect_to_snowflake(
        self,
        account: str,
        username: str,
        password: str,
        database: str,
        role: Union[str, None] = None,
        warehouse: Union[str, None] = None,
        **kwargs
    ):
        """
        连接到Snowflake数据库的方法
        
        功能说明:
        1. 建立与Snowflake数据仓库的连接
        2. 配置数据库方言和SQL执行函数
        3. 支持环境变量配置和参数验证
        
        核心流程:
        - 检查并导入snowflake.connector依赖
        - 从环境变量或参数获取连接信息
        - 建立数据库连接
        - 定义SQL执行函数
        - 设置数据库方言
        
        参数说明:
        - account: Snowflake账户标识符
        - username: 用户名
        - password: 密码
        - database: 数据库名称
        - role: 可选的角色设置
        - warehouse: 可选的仓库设置
        
        环境变量支持:
        - SNOWFLAKE_USERNAME: 用户名
        - SNOWFLAKE_PASSWORD: 密码
        - SNOWFLAKE_ACCOUNT: 账户
        - SNOWFLAKE_DATABASE: 数据库
        
        应用场景:
        - 企业级数据仓库连接
        - 大规模数据分析
        - 云原生数据处理
        
        设计特点:
        - 环境变量优先: 支持从环境变量读取敏感信息
        - 依赖检查: 自动检查必要的依赖包
        - 错误处理: 详细的配置错误提示
        - 会话保持: 启用客户端会话保持活跃
        """
        try:
            snowflake = __import__("snowflake.connector")
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[snowflake]"
            )

        if username == "my-username":
            username_env = os.getenv("SNOWFLAKE_USERNAME")

            if username_env is not None:
                username = username_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake username.")

        if password == "mypassword":
            password_env = os.getenv("SNOWFLAKE_PASSWORD")

            if password_env is not None:
                password = password_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake password.")

        if account == "my-account":
            account_env = os.getenv("SNOWFLAKE_ACCOUNT")

            if account_env is not None:
                account = account_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake account.")

        if database == "my-database":
            database_env = os.getenv("SNOWFLAKE_DATABASE")

            if database_env is not None:
                database = database_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake database.")

        conn = snowflake.connector.connect(
            user=username,
            password=password,
            account=account,
            database=database,
            client_session_keep_alive=True,
            **kwargs
        )

        def run_sql_snowflake(sql: str) -> pd.DataFrame:
            cs = conn.cursor()

            if role is not None:
                cs.execute(f"USE ROLE {role}")

            if warehouse is not None:
                cs.execute(f"USE WAREHOUSE {warehouse}")
            cs.execute(f"USE DATABASE {database}")

            cur = cs.execute(sql)

            results = cur.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])

            return df

        self.dialect = "Snowflake SQL"
        self.run_sql = run_sql_snowflake
        self.run_sql_is_set = True

    def connect_to_sqlite(self, url: str, check_same_thread: bool = False,  **kwargs):
        """
        连接到SQLite数据库的方法
        
        功能说明:
        1. 连接到SQLite数据库，支持本地文件和远程URL
        2. 自动下载远程数据库文件到本地
        3. 配置数据库方言和SQL执行函数
        
        核心流程:
        - 解析数据库URL路径
        - 检查本地文件是否存在
        - 如果是远程URL，下载数据库文件
        - 建立SQLite连接
        - 定义SQL执行函数
        - 设置数据库方言
        
        参数说明:
        - url: 数据库文件路径或下载URL
        - check_same_thread: 是否允许多线程访问连接
        
        应用场景:
        - 轻量级数据库连接
        - 本地数据分析
        - 原型开发和测试
        - 小型应用数据存储
        
        设计特点:
        - 自动下载: 支持从URL自动下载数据库
        - 线程安全: 可配置多线程访问
        - 简单易用: 最少配置即可使用
        - 错误处理: 包含下载和连接的错误检查
        
        Connect to a SQLite database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            url (str): The URL of the database to connect to.
            check_same_thread (str): Allow the connection may be accessed in multiple threads.
        Returns:
            None
        """

        # URL of the database to download

        # Path to save the downloaded database
        path = os.path.basename(urlparse(url).path)

        # Download the database if it doesn't exist
        if not os.path.exists(url):
            response = requests.get(url)
            response.raise_for_status()  # Check that the request was successful
            with open(path, "wb") as f:
                f.write(response.content)
            url = path

        # Connect to the database
        conn = sqlite3.connect(
            url,
            check_same_thread=check_same_thread,
            **kwargs
        )

        def run_sql_sqlite(sql: str):
            return pd.read_sql_query(sql, conn)

        self.dialect = "SQLite"
        self.run_sql = run_sql_sqlite
        self.run_sql_is_set = True

    def connect_to_postgres(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        """
        连接到PostgreSQL数据库的方法
        
        功能说明:
        1. 使用psycopg2连接器连接到PostgreSQL数据库
        2. 支持环境变量配置和参数验证
        3. 包含连接重试和错误处理机制
        
        核心流程:
        - 检查并导入psycopg2依赖
        - 从环境变量或参数获取连接信息
        - 建立数据库连接
        - 定义带重试机制的SQL执行函数
        - 设置数据库方言
        
        参数说明:
        - host: PostgreSQL服务器主机地址
        - dbname: 数据库名称
        - user: 用户名
        - password: 密码
        - port: 端口号
        
        环境变量支持:
        - HOST: 主机地址
        - DATABASE: 数据库名称
        - PG_USER: 用户名
        - PASSWORD: 密码
        - PORT: 端口号
        
        应用场景:
        - 企业级关系数据库连接
        - 高并发数据处理
        - 复杂查询和事务处理
        - 生产环境数据分析
        
        设计特点:
        - 环境变量优先: 支持从环境变量读取配置
        - 连接重试: 自动处理连接中断和重连
        - 错误处理: 详细的错误分类和处理
        - 事务管理: 支持事务回滚和错误恢复
        
        Connect to postgres using the psycopg2 connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_postgres(
            host="myhost",
            dbname="mydatabase",
            user="myuser",
            password="mypassword",
            port=5432
        )
        ```
        Args:
            host (str): The postgres host.
            dbname (str): The postgres database name.
            user (str): The postgres user.
            password (str): The postgres password.
            port (int): The postgres Port.
        """

        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[postgres]"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your postgres host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your postgres database")

        if not user:
            user = os.getenv("PG_USER")

        if not user:
            raise ImproperlyConfigured("Please set your postgres user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your postgres password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your postgres port")

        conn = None

        try:
            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
                **kwargs
            )
        except psycopg2.Error as e:
            raise ValidationError(e)

        def connect_to_db():
            return psycopg2.connect(host=host, dbname=dbname,
                        user=user, password=password, port=port, **kwargs)


        def run_sql_postgres(sql: str) -> Union[pd.DataFrame, None]:
            conn = None
            try:
                conn = connect_to_db()  # Initial connection attempt
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.InterfaceError as e:
                # Attempt to reconnect and retry the operation
                if conn:
                    conn.close()  # Ensure any existing connection is closed
                conn = connect_to_db()
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.Error as e:
                if conn:
                    conn.rollback()
                    raise ValidationError(e)

            except Exception as e:
                        conn.rollback()
                        raise e

        self.dialect = "PostgreSQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres


    def connect_to_mysql(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):
        """
        连接到MySQL数据库的方法
        
        功能说明:
        1. 使用PyMySQL连接器连接到MySQL数据库
        2. 支持环境变量配置和参数验证
        3. 包含连接重试和错误处理机制
        
        核心流程:
        - 检查并导入PyMySQL依赖
        - 从环境变量或参数获取连接信息
        - 建立数据库连接
        - 定义带重试机制的SQL执行函数
        - 设置数据库方言
        
        参数说明:
        - host: MySQL服务器主机地址
        - dbname: 数据库名称
        - user: 用户名
        - password: 密码
        - port: 端口号
        
        环境变量支持:
        - HOST: 主机地址
        - DATABASE: 数据库名称
        - USER: 用户名
        - PASSWORD: 密码
        - PORT: 端口号
        
        应用场景:
        - Web应用数据库连接
        - 中小型企业数据处理
        - 快速原型开发
        - 数据分析和报表生成
        
        设计特点:
        - 轻量级连接: PyMySQL纯Python实现
        - 环境变量优先: 支持从环境变量读取配置
        - 连接重试: 自动处理连接中断和重连
        - 错误处理: 详细的错误分类和处理
        """

        try:
            import pymysql.cursors
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install PyMySQL"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your MySQL host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your MySQL database")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your MySQL user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your MySQL password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your MySQL port")

        conn = None

        try:
            conn = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=dbname,
                port=port,
                cursorclass=pymysql.cursors.DictCursor,
                **kwargs
            )
        except pymysql.Error as e:
            raise ValidationError(e)

        def run_sql_mysql(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    conn.ping(reconnect=True)
                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except pymysql.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_mysql

    def connect_to_clickhouse(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):
        """
        连接到ClickHouse数据库的方法
        
        功能说明:
        1. 使用clickhouse_connect连接器连接到ClickHouse数据库
        2. 支持环境变量配置和参数验证
        3. 专为OLAP分析场景优化
        
        核心流程:
        - 检查并导入clickhouse_connect依赖
        - 从环境变量或参数获取连接信息
        - 建立数据库连接
        - 定义SQL执行函数
        - 设置数据库方言
        
        参数说明:
        - host: ClickHouse服务器主机地址
        - dbname: 数据库名称
        - user: 用户名
        - password: 密码
        - port: 端口号
        
        环境变量支持:
        - HOST: 主机地址
        - DATABASE: 数据库名称
        - USER: 用户名
        - PASSWORD: 密码
        - PORT: 端口号
        
        应用场景:
        - 大数据分析和OLAP查询
        - 实时数据仓库
        - 高性能数据聚合
        - 时序数据分析
        
        设计特点:
        - 列式存储: 专为分析查询优化
        - 高性能: 支持大规模数据处理
        - 实时查询: 低延迟数据分析
        - 压缩存储: 高效的数据压缩
        """

        try:
            import clickhouse_connect
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install clickhouse_connect"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your ClickHouse host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your ClickHouse database")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your ClickHouse user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your ClickHouse password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your ClickHouse port")

        conn = None

        try:
            conn = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=user,
                password=password,
                database=dbname,
                **kwargs
            )
            print(conn)
        except Exception as e:
            raise ValidationError(e)

        def run_sql_clickhouse(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    result = conn.query(sql)
                    results = result.result_rows

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(results, columns=result.column_names)
                    return df

                except Exception as e:
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_clickhouse

    def connect_to_oracle(
        self,
        user: str = None,
        password: str = None,
        dsn: str = None,
        **kwargs
    ):

        """
        连接到Oracle数据库的方法
        
        功能说明:
        1. 使用oracledb包连接到Oracle数据库
        2. 支持环境变量配置和参数验证
        3. 专为企业级数据库连接优化
        
        核心流程:
        - 检查并导入oracledb依赖
        - 从环境变量或参数获取连接信息
        - 建立数据库连接
        - 定义SQL执行函数
        - 设置数据库方言
        
        参数说明:
        - user: Oracle数据库用户名
        - password: 用户密码
        - dsn: 数据源名称(格式: host:port/sid)
        
        环境变量支持:
        - USER: 用户名
        - PASSWORD: 密码
        - DSN: 数据源名称
        
        应用场景:
        - 企业级数据库连接
        - 大型事务处理系统
        - 复杂业务逻辑处理
        - 高可用性数据服务
        
        设计特点:
        - 企业级: 支持大规模企业应用
        - 高可靠性: 强一致性和事务支持
        - 安全性: 完善的权限和安全机制
        - 性能优化: 针对复杂查询优化
        
        Connect to an Oracle db using oracledb package. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_oracle(
        user="username",
        password="password",
        dsn="host:port/sid",
        )
        ```
        Args:
            USER (str): Oracle db user name.
            PASSWORD (str): Oracle db user password.
            DSN (str): Oracle db host ip - host:port/sid.
        """

        try:
            import oracledb
        except ImportError:

            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install oracledb"
            )

        if not dsn:
            dsn = os.getenv("DSN")

        if not dsn:
            raise ImproperlyConfigured("Please set your Oracle dsn which should include host:port/sid")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your Oracle db user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your Oracle db password")

        conn = None

        try:
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                **kwargs
            )
        except oracledb.Error as e:
            raise ValidationError(e)

        def run_sql_oracle(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    sql = sql.rstrip()
                    if sql.endswith(';'): #fix for a known problem with Oracle db where an extra ; will cause an error.
                        sql = sql[:-1]

                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except oracledb.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_oracle

    def connect_to_bigquery(
        self,
        cred_file_path: str = None,
        project_id: str = None,
        **kwargs
    ):
        """
        连接到Google BigQuery数据库的方法
        
        功能说明:
        1. 使用Google Cloud BigQuery连接器连接到BigQuery
        2. 支持多种认证方式(服务账号、隐式认证、Colab认证)
        3. 专为大数据分析和云端数据仓库优化
        
        核心流程:
        - 检查并导入BigQuery相关依赖
        - 从环境变量或参数获取项目ID
        - 处理不同环境的认证方式
        - 建立BigQuery客户端连接
        - 定义SQL执行函数
        
        参数说明:
        - cred_file_path: Google Cloud服务账号凭证文件路径
        - project_id: Google Cloud项目ID
        
        环境变量支持:
        - PROJECT_ID: Google Cloud项目ID
        
        认证方式:
        - 隐式认证: 使用环境默认凭证
        - 服务账号: 使用JSON凭证文件
        - Colab认证: Google Colab环境自动认证
        
        应用场景:
        - 大数据分析和数据仓库
        - 云端数据处理
        - 机器学习数据准备
        - 实时数据分析
        
        设计特点:
        - 云原生: 完全托管的云端数据仓库
        - 无服务器: 按需扩展，无需管理基础设施
        - 高性能: 支持PB级数据分析
        - 标准SQL: 支持ANSI SQL标准
        
        Connect to gcs using the bigquery connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_bigquery(
            project_id="myprojectid",
            cred_file_path="path/to/credentials.json",
        )
        ```
        Args:
            project_id (str): The gcs project id.
            cred_file_path (str): The gcs credential file path
        """

        try:
            from google.api_core.exceptions import GoogleAPIError
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[bigquery]"
            )

        if not project_id:
            project_id = os.getenv("PROJECT_ID")

        if not project_id:
            raise ImproperlyConfigured("Please set your Google Cloud Project ID.")

        import sys

        if "google.colab" in sys.modules:
            try:
                from google.colab import auth

                auth.authenticate_user()
            except Exception as e:
                raise ImproperlyConfigured(e)
        else:
            print("Not using Google Colab.")

        conn = None

        if not cred_file_path:
            try:
                conn = bigquery.Client(project=project_id)
            except:
                print("Could not found any google cloud implicit credentials")
        else:
            # Validate file path and pemissions
            validate_config_path(cred_file_path)

        if not conn:
            with open(cred_file_path, "r") as f:
                credentials = service_account.Credentials.from_service_account_info(
                    json.loads(f.read()),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

            try:
                conn = bigquery.Client(
                    project=project_id,
                    credentials=credentials,
                    **kwargs
                )
            except:
                raise ImproperlyConfigured(
                    "Could not connect to bigquery please correct credentials"
                )

        def run_sql_bigquery(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                job = conn.query(sql)
                df = job.result().to_dataframe()
                return df
            return None

        self.dialect = "BigQuery SQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_bigquery

    def connect_to_duckdb(self, url: str, init_sql: str = None, **kwargs):
        """
        连接到DuckDB数据库的方法
        
        功能说明:
        1. 连接到DuckDB数据库(本地文件、内存或MotherDuck)
        2. 支持自动下载远程数据库文件
        3. 配置数据库方言和SQL执行函数
        4. 支持初始化SQL语句执行
        
        核心流程:
        - 依赖检查: 验证DuckDB库是否已安装
        - 路径处理: 处理不同类型的数据库URL
        - 文件下载: 自动下载远程数据库文件(如需要)
        - 连接建立: 创建DuckDB数据库连接
        - 初始化: 执行初始化SQL语句(可选)
        - 配置设置: 设置SQL执行函数和方言
        
        参数说明:
        - url: 数据库URL(:memory:表示内存数据库, md:或motherduck:表示MotherDuck)
        - init_sql: 连接后执行的初始化SQL语句
        - **kwargs: 其他连接参数
        
        支持的URL类型:
        - :memory: 或空字符串: 内存数据库
        - 本地文件路径: 本地DuckDB文件
        - md:或motherduck:: MotherDuck云数据库
        - HTTP URL: 远程数据库文件(自动下载)
        
        应用场景:
        - 本地数据分析
        - 快速原型开发
        - 云端数据处理(MotherDuck)
        - 数据科学实验
        
        设计特点:
        - 多源支持: 支持多种数据库来源
        - 自动化: 自动处理文件下载和连接
        - 灵活性: 支持内存和持久化存储
        - 云集成: 原生支持MotherDuck云服务
        
        Connect to a DuckDB database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            url (str): The URL of the database to connect to. Use :memory: to create an in-memory database. Use md: or motherduck: to use the MotherDuck database.
            init_sql (str, optional): SQL to run when connecting to the database. Defaults to None.

        Returns:
            None
        """
        try:
            import duckdb
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[duckdb]"
            )
        # URL of the database to download
        if url == ":memory:" or url == "":
            path = ":memory:"
        else:
            # Path to save the downloaded database
            print(os.path.exists(url))
            if os.path.exists(url):
                path = url
            elif url.startswith("md") or url.startswith("motherduck"):
                path = url
            else:
                path = os.path.basename(urlparse(url).path)
                # Download the database if it doesn't exist
                if not os.path.exists(path):
                    response = requests.get(url)
                    response.raise_for_status()  # Check that the request was successful
                    with open(path, "wb") as f:
                        f.write(response.content)

        # Connect to the database
        conn = duckdb.connect(path, **kwargs)
        if init_sql:
            conn.query(init_sql)

        def run_sql_duckdb(sql: str):
            return conn.query(sql).to_df()

        self.dialect = "DuckDB SQL"
        self.run_sql = run_sql_duckdb
        self.run_sql_is_set = True

    def connect_to_mssql(self, odbc_conn_str: str, **kwargs):
        """
        连接到Microsoft SQL Server数据库的方法
        
        功能说明:
        1. 使用pyodbc和SQLAlchemy连接到SQL Server数据库
        2. 支持ODBC连接字符串配置
        3. 专为Windows环境和企业级应用优化
        
        核心流程:
        - 检查并导入pyodbc和SQLAlchemy依赖
        - 构建SQLAlchemy连接URL
        - 创建数据库引擎
        - 定义SQL执行函数
        - 设置数据库方言
        
        参数说明:
        - odbc_conn_str: ODBC连接字符串
        
        应用场景:
        - Windows企业环境数据库连接
        - .NET应用数据集成
        - 企业级数据仓库
        - 商业智能和报表系统
        
        设计特点:
        - Windows集成: 与Windows环境深度集成
        - 企业级: 支持大型企业应用
        - T-SQL支持: 完整的T-SQL语法支持
        - 高性能: 优化的查询执行引擎
        
        Connect to a Microsoft SQL Server database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            odbc_conn_str (str): The ODBC connection string.

        Returns:
            None
        """
        try:
            import pyodbc
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install pyodbc"
            )

        try:
            import sqlalchemy as sa
            from sqlalchemy.engine import URL
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install sqlalchemy"
            )

        connection_url = URL.create(
            "mssql+pyodbc", query={"odbc_connect": odbc_conn_str}
        )

        from sqlalchemy import create_engine

        engine = create_engine(connection_url, **kwargs)

        def run_sql_mssql(sql: str):
            # Execute the SQL statement and return the result as a pandas DataFrame
            with engine.begin() as conn:
                df = pd.read_sql_query(sa.text(sql), conn)
                conn.close()
                return df

            raise Exception("Couldn't run sql")

        self.dialect = "T-SQL / Microsoft SQL Server"
        self.run_sql = run_sql_mssql
        self.run_sql_is_set = True
    def connect_to_presto(
        self,
        host: str,
        catalog: str = 'hive',
        schema: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        combined_pem_path: str = None,
        protocol: str = 'https',
        requests_kwargs: dict = None,
        **kwargs
    ):
      """
        连接到Presto数据库的方法
        
        功能说明:
        1. 使用PyHive连接器连接到Presto分布式SQL查询引擎
        2. 支持SSL认证和多种配置参数
        3. 专为大数据分析和分布式查询优化
        
        核心流程:
        - 检查并导入PyHive依赖
        - 从环境变量或参数获取连接信息
        - 配置SSL连接参数
        - 建立Presto连接
        - 定义SQL执行函数
        
        参数说明:
        - host: Presto服务器主机地址
        - catalog: Presto目录名称
        - schema: 模式名称
        - user: 用户名
        - password: 密码
        - port: 端口号
        - combined_pem_path: SSL证书文件路径
        - protocol: 连接协议(默认https)
        - requests_kwargs: 请求参数字典
        
        环境变量支持:
        - PRESTO_HOST: 主机地址
        - PRESTO_CATALOG: 目录名称
        - PRESTO_USER: 用户名
        - PRESTO_PASSWORD: 密码
        - PRESTO_PORT: 端口号
        
        应用场景:
        - 大数据分析和OLAP查询
        - 分布式数据处理
        - 多数据源联合查询
        - 实时数据分析
        
        设计特点:
        - 分布式: 支持大规模分布式查询
        - 多数据源: 可连接多种数据源
        - 高性能: 内存计算优化
        - 标准SQL: 支持ANSI SQL标准
        
        Connect to a Presto database using the specified parameters.

        Args:
            host (str): The host address of the Presto database.
            catalog (str): The catalog to use in the Presto environment.
            schema (str): The schema to use in the Presto environment.
            user (str): The username for authentication.
            password (str): The password for authentication.
            port (int): The port number for the Presto connection.
            combined_pem_path (str): The path to the combined pem file for SSL connection.
            protocol (str): The protocol to use for the connection (default is 'https').
            requests_kwargs (dict): Additional keyword arguments for requests.

        Raises:
            DependencyError: If required dependencies are not installed.
            ImproperlyConfigured: If essential configuration settings are missing.

        Returns:
            None
      """
      try:
        from pyhive import presto
      except ImportError:
        raise DependencyError(
          "You need to install required dependencies to execute this method,"
          " run command: \npip install pyhive"
        )

      if not host:
        host = os.getenv("PRESTO_HOST")

      if not host:
        raise ImproperlyConfigured("Please set your presto host")

      if not catalog:
        catalog = os.getenv("PRESTO_CATALOG")

      if not catalog:
        raise ImproperlyConfigured("Please set your presto catalog")

      if not user:
        user = os.getenv("PRESTO_USER")

      if not user:
        raise ImproperlyConfigured("Please set your presto user")

      if not password:
        password = os.getenv("PRESTO_PASSWORD")

      if not port:
        port = os.getenv("PRESTO_PORT")

      if not port:
        raise ImproperlyConfigured("Please set your presto port")

      conn = None

      try:
        if requests_kwargs is None and combined_pem_path is not None:
          # use the combined pem file to verify the SSL connection
          requests_kwargs = {
            'verify': combined_pem_path,  # 使用转换后得到的 PEM 文件进行 SSL 验证
          }
        conn = presto.Connection(host=host,
                                 username=user,
                                 password=password,
                                 catalog=catalog,
                                 schema=schema,
                                 port=port,
                                 protocol=protocol,
                                 requests_kwargs=requests_kwargs,
                                 **kwargs)
      except presto.Error as e:
        raise ValidationError(e)

      def run_sql_presto(sql: str) -> Union[pd.DataFrame, None]:
        if conn:
          try:
            sql = sql.rstrip()
            # fix for a known problem with presto db where an extra ; will cause an error.
            if sql.endswith(';'):
                sql = sql[:-1]
            cs = conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(
              results, columns=[desc[0] for desc in cs.description]
            )
            return df

          except presto.Error as e:
            print(e)
            raise ValidationError(e)

          except Exception as e:
            print(e)
            raise e

      self.run_sql_is_set = True
      self.run_sql = run_sql_presto

    def connect_to_hive(
        self,
        host: str = None,
        dbname: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        auth: str = 'CUSTOM',
        **kwargs
    ):
      """
        连接到Hive数据库的方法
        
        功能说明:
        1. 使用PyHive连接器连接到Apache Hive数据仓库
        2. 支持多种认证方式和配置参数
        3. 专为Hadoop生态系统和大数据处理优化
        
        核心流程:
        - 检查并导入PyHive依赖
        - 从环境变量或参数获取连接信息
        - 建立Hive连接
        - 定义SQL执行函数
        - 设置数据库方言
        
        参数说明:
        - host: Hive服务器主机地址
        - dbname: 数据库名称
        - user: 用户名
        - password: 密码
        - port: 端口号
        - auth: 认证方式
        
        环境变量支持:
        - HIVE_HOST: 主机地址
        - HIVE_DATABASE: 数据库名称
        - HIVE_USER: 用户名
        - HIVE_PASSWORD: 密码
        - HIVE_PORT: 端口号
        
        应用场景:
        - Hadoop生态系统数据查询
        - 大数据仓库分析
        - ETL数据处理
        - 批量数据分析
        
        设计特点:
        - Hadoop集成: 与Hadoop生态系统深度集成
        - 大数据处理: 支持PB级数据分析
        - 批处理优化: 针对批量处理优化
        - 分布式存储: 支持HDFS分布式存储
        
        Connect to a Hive database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        Connect to a Hive database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            host (str): The host of the Hive database.
            dbname (str): The name of the database to connect to.
            user (str): The username to use for authentication.
            password (str): The password to use for authentication.
            port (int): The port to use for the connection.
            auth (str): The authentication method to use.

        Returns:
            None
      """

      try:
        from pyhive import hive
      except ImportError:
        raise DependencyError(
          "You need to install required dependencies to execute this method,"
          " run command: \npip install pyhive"
        )

      if not host:
        host = os.getenv("HIVE_HOST")

      if not host:
        raise ImproperlyConfigured("Please set your hive host")

      if not dbname:
        dbname = os.getenv("HIVE_DATABASE")

      if not dbname:
        raise ImproperlyConfigured("Please set your hive database")

      if not user:
        user = os.getenv("HIVE_USER")

      if not user:
        raise ImproperlyConfigured("Please set your hive user")

      if not password:
        password = os.getenv("HIVE_PASSWORD")

      if not port:
        port = os.getenv("HIVE_PORT")

      if not port:
        raise ImproperlyConfigured("Please set your hive port")

      conn = None

      try:
        conn = hive.Connection(host=host,
                               username=user,
                               password=password,
                               database=dbname,
                               port=port,
                               auth=auth)
      except hive.Error as e:
        raise ValidationError(e)

      def run_sql_hive(sql: str) -> Union[pd.DataFrame, None]:
        if conn:
          try:
            cs = conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(
              results, columns=[desc[0] for desc in cs.description]
            )
            return df

          except hive.Error as e:
            print(e)
            raise ValidationError(e)

          except Exception as e:
            print(e)
            raise e

      self.run_sql_is_set = True
      self.run_sql = run_sql_hive

    def run_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        """
        执行SQL查询的抽象方法
        
        功能说明:
        1. 在连接的数据库上执行SQL查询
        2. 返回查询结果的DataFrame格式
        3. 需要先连接到数据库才能使用
        
        核心功能:
        - SQL查询执行: 在目标数据库上执行SQL语句
        - 结果返回: 将查询结果转换为pandas DataFrame
        - 错误处理: 处理SQL执行过程中的异常
        
        参数说明:
        - sql: 要执行的SQL查询语句
        - **kwargs: 其他可选参数
        
        返回值:
        - pd.DataFrame: SQL查询结果的数据框
        
        使用前提:
        - 必须先通过connect_to_*方法连接到数据库
        - 确保SQL语句语法正确
        - 具有相应的数据库访问权限
        
        应用场景:
        - 数据查询和分析
        - 报表生成
        - 数据验证
        - 业务逻辑实现
        
        Example:
        ```python
        vn.run_sql("SELECT * FROM my_table")
        ```

        Run a SQL query on the connected database.

        Args:
            sql (str): The SQL query to run.

        Returns:
            pd.DataFrame: The results of the SQL query.
        """
        raise Exception(
            "You need to connect to a database first by running vn.connect_to_snowflake(), vn.connect_to_postgres(), similar function, or manually set vn.run_sql"
        )

    def ask(
        self,
        question: Union[str, None] = None,
        print_results: bool = True,
        auto_train: bool = True,
        visualize: bool = True,  # if False, will not generate plotly code
        allow_llm_to_see_data: bool = False,
    ) -> Union[
        Tuple[
            Union[str, None],
            Union[pd.DataFrame, None],
            Union[plotly.graph_objs.Figure, None],
        ],
        None,
    ]:
        """
        向Vanna AI提问并获取SQL查询结果的核心方法
        
        功能说明:
        1. 接收自然语言问题并生成对应的SQL查询
        2. 执行SQL查询并返回结果数据
        3. 可选择性地生成数据可视化图表
        4. 支持自动训练和结果展示
        
        核心流程:
        - 问题处理: 接收并处理用户的自然语言问题
        - SQL生成: 调用generate_sql方法生成SQL查询
        - 查询执行: 在数据库上执行生成的SQL
        - 结果处理: 格式化并展示查询结果
        - 可视化: 可选生成Plotly图表
        - 自动训练: 可选将问题-SQL对加入训练集
        
        参数说明:
        - question: 用户的自然语言问题
        - print_results: 是否打印SQL查询结果
        - auto_train: 是否自动训练模型
        - visualize: 是否生成可视化图表
        - allow_llm_to_see_data: 是否允许LLM查看数据内容
        
        返回值:
        - Tuple: (SQL查询语句, 查询结果DataFrame, Plotly图表对象)
        
        应用场景:
        - 自然语言数据查询
        - 商业智能分析
        - 数据探索和发现
        - 自助式数据分析
        
        设计特点:
        - 端到端: 从问题到结果的完整流程
        - 智能化: 自动SQL生成和优化
        - 可视化: 自动图表生成
        - 交互式: 支持交互式数据探索
        
        **Example:**
        ```python
        vn.ask("What are the top 10 customers by sales?")
        ```

        Ask Vanna.AI a question and get the SQL query that answers it.

        Args:
            question (str): The question to ask.
            print_results (bool): Whether to print the results of the SQL query.
            auto_train (bool): Whether to automatically train Vanna.AI on the question and SQL query.
            visualize (bool): Whether to generate plotly code and display the plotly figure.

        Returns:
            Tuple[str, pd.DataFrame, plotly.graph_objs.Figure]: The SQL query, the results of the SQL query, and the plotly figure.
        """

        if question is None:
            question = input("Enter a question: ")

        try:
            sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
        except Exception as e:
            print(e)
            return None, None, None

        if print_results:
            try:
                Code = __import__("IPython.display", fromList=["Code"]).Code
                display(Code(sql))
            except Exception as e:
                print(sql)

        if self.run_sql_is_set is False:
            print(
                "If you want to run the SQL query, connect to a database first."
            )

            if print_results:
                return None
            else:
                return sql, None, None

        try:
            df = self.run_sql(sql)

            if print_results:
                try:
                    display = __import__(
                        "IPython.display", fromList=["display"]
                    ).display
                    display(df)
                except Exception as e:
                    print(df)

            if len(df) > 0 and auto_train:
                self.add_question_sql(question=question, sql=sql)
            # Only generate plotly code if visualize is True
            if visualize:
                try:
                    plotly_code = self.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                    )
                    fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                    if print_results:
                        try:
                            display = __import__(
                                "IPython.display", fromlist=["display"]
                            ).display
                            Image = __import__(
                                "IPython.display", fromlist=["Image"]
                            ).Image
                            img_bytes = fig.to_image(format="png", scale=2)
                            display(Image(img_bytes))
                        except Exception as e:
                            fig.show()
                except Exception as e:
                    # Print stack trace
                    traceback.print_exc()
                    print("Couldn't run plotly code: ", e)
                    if print_results:
                        return None
                    else:
                        return sql, df, None
            else:
                return sql, df, None

        except Exception as e:
            print("Couldn't run sql: ", e)
            if print_results:
                return None
            else:
                return sql, None, None
        return sql, df, fig

    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
    ) -> str:
        """
        训练Vanna AI模型的核心方法
        
        功能说明:
        1. 训练Vanna AI模型以提高SQL生成准确性
        2. 支持多种训练数据类型(问题-SQL对、DDL、文档等)
        3. 可以使用训练计划进行批量训练
        4. 自动从数据库元数据中学习
        
        核心功能:
        - 问题-SQL训练: 通过问题和对应SQL提升模型理解
        - DDL训练: 学习数据库结构和表关系
        - 文档训练: 学习业务逻辑和领域知识
        - 元数据训练: 自动从连接的数据库学习结构
        - 批量训练: 通过训练计划执行批量训练
        
        参数说明:
        - question: 要训练的问题
        - sql: 对应的SQL查询语句
        - ddl: 数据定义语言(DDL)语句
        - documentation: 相关文档和说明
        - plan: 训练计划对象
        
        训练模式:
        - 无参数: 从连接的数据库自动学习元数据
        - question+sql: 添加问题-SQL训练对
        - ddl: 添加数据库结构信息
        - documentation: 添加业务文档
        - plan: 执行完整的训练计划
        
        应用场景:
        - 模型初始化训练
        - 持续学习和优化
        - 领域知识注入
        - 数据库结构学习
        
        设计特点:
        - 多模态: 支持多种类型的训练数据
        - 自适应: 根据参数自动选择训练方式
        - 增量式: 支持持续训练和更新
        - 智能化: 自动从数据库学习结构
        
        **Example:**
        ```python
        vn.train()
        ```

        Train Vanna.AI on a question and its corresponding SQL query.
        If you call it with no arguments, it will check if you connected to a database and it will attempt to train on the metadata of that database.
        If you call it with the sql argument, it's equivalent to [`vn.add_question_sql()`][vanna.base.base.VannaBase.add_question_sql].
        If you call it with the ddl argument, it's equivalent to [`vn.add_ddl()`][vanna.base.base.VannaBase.add_ddl].
        If you call it with the documentation argument, it's equivalent to [`vn.add_documentation()`][vanna.base.base.VannaBase.add_documentation].
        Additionally, you can pass a [`TrainingPlan`][vanna.types.TrainingPlan] object. Get a training plan with [`vn.get_training_plan_generic()`][vanna.base.base.VannaBase.get_training_plan_generic].

        Args:
            question (str): The question to train on.
            sql (str): The SQL query to train on.
            ddl (str):  The DDL statement.
            documentation (str): The documentation to train on.
            plan (TrainingPlan): The training plan to train on.
        """

        if question and not sql:
            raise ValidationError("Please also provide a SQL query")

        if documentation:
            print("Adding documentation....")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                print("Question generated with sql:", question, "\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

    def _get_databases(self) -> List[str]:
        try:
            print("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.DATABASES")
        except Exception as e:
            print(e)
            try:
                print("Trying SHOW DATABASES")
                df_databases = self.run_sql("SHOW DATABASES")
            except Exception as e:
                print(e)
                return []

        return df_databases["DATABASE_NAME"].unique().tolist()

    def _get_information_schema_tables(self, database: str) -> pd.DataFrame:
        df_tables = self.run_sql(f"SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES")

        return df_tables

    def get_training_plan_generic(self, df) -> TrainingPlan:
        """
        This method is used to generate a training plan from an information schema dataframe.

        Basically what it does is breaks up INFORMATION_SCHEMA.COLUMNS into groups of table/column descriptions that can be used to pass to the LLM.

        Args:
            df (pd.DataFrame): The dataframe to generate the training plan from.

        Returns:
            TrainingPlan: The training plan.
        """
        # For each of the following, we look at the df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        columns = [database_column,
                    schema_column,
                    table_column]
        candidates = ["column_name",
                      "data_type",
                      "comment"]
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                    doc += df_columns_filtered_to_table[columns].to_markdown()

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        return plan

    def get_training_plan_snowflake(
        self,
        filter_databases: Union[List[str], None] = None,
        filter_schemas: Union[List[str], None] = None,
        include_information_schema: bool = False,
        use_historical_queries: bool = True,
    ) -> TrainingPlan:
        plan = TrainingPlan([])

        if self.run_sql_is_set is False:
            raise ImproperlyConfigured("Please connect to a database first.")

        if use_historical_queries:
            try:
                print("Trying query history")
                df_history = self.run_sql(
                    """ select * from table(information_schema.query_history(result_limit => 5000)) order by start_time"""
                )

                df_history_filtered = df_history.query("ROWS_PRODUCED > 1")
                if filter_databases is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_databases]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if filter_schemas is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_schemas]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if len(df_history_filtered) > 10:
                    df_history_filtered = df_history_filtered.sample(10)

                for query in df_history_filtered["QUERY_TEXT"].unique().tolist():
                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_SQL,
                            item_group="",
                            item_name=self.generate_question(query),
                            item_value=query,
                        )
                    )

            except Exception as e:
                print(e)

        databases = self._get_databases()

        for database in databases:
            if filter_databases is not None and database not in filter_databases:
                continue

            try:
                df_tables = self._get_information_schema_tables(database=database)

                print(f"Trying INFORMATION_SCHEMA.COLUMNS for {database}")
                df_columns = self.run_sql(
                    f"SELECT * FROM {database}.INFORMATION_SCHEMA.COLUMNS"
                )

                for schema in df_tables["TABLE_SCHEMA"].unique().tolist():
                    if filter_schemas is not None and schema not in filter_schemas:
                        continue

                    if (
                        not include_information_schema
                        and schema == "INFORMATION_SCHEMA"
                    ):
                        continue

                    df_columns_filtered_to_schema = df_columns.query(
                        f"TABLE_SCHEMA == '{schema}'"
                    )

                    try:
                        tables = (
                            df_columns_filtered_to_schema["TABLE_NAME"]
                            .unique()
                            .tolist()
                        )

                        for table in tables:
                            df_columns_filtered_to_table = (
                                df_columns_filtered_to_schema.query(
                                    f"TABLE_NAME == '{table}'"
                                )
                            )
                            doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                            doc += df_columns_filtered_to_table[
                                [
                                    "TABLE_CATALOG",
                                    "TABLE_SCHEMA",
                                    "TABLE_NAME",
                                    "COLUMN_NAME",
                                    "DATA_TYPE",
                                    "COMMENT",
                                ]
                            ].to_markdown()

                            plan._plan.append(
                                TrainingPlanItem(
                                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                                    item_group=f"{database}.{schema}",
                                    item_name=table,
                                    item_value=doc,
                                )
                            )

                    except Exception as e:
                        print(e)
                        pass
            except Exception as e:
                print(e)

        return plan

    def get_plotly_figure(
        self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        **Example:**
        ```python
        fig = vn.get_plotly_figure(
            plotly_code="fig = px.bar(df, x='name', y='salary')",
            df=df
        )
        fig.show()
        ```
        Get a Plotly figure from a dataframe and Plotly code.

        Args:
            df (pd.DataFrame): The dataframe to use.
            plotly_code (str): The Plotly code to use.

        Returns:
            plotly.graph_objs.Figure: The Plotly figure.
        """
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)

            fig = ldict.get("fig", None)
        except Exception as e:
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Decision-making for plot type
            if len(numeric_cols) >= 2:
                # Use the first two numeric columns for a scatter plot
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                # Use a bar plot if there's one numeric and one categorical column
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
                # Use a pie chart for categorical data with fewer unique values
                fig = px.pie(df, names=categorical_cols[0])
            else:
                # Default to a simple line plot if above conditions are not met
                fig = px.line(df)

        if fig is None:
            return None

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig
