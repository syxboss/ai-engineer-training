# 使用提醒:
# 0. 本段代码只能在影刀RPA软件内部运行
# 1. xbot包提供软件自动化、数据表格、Excel、日志、AI等功能。注意，此包不是开源的
# 2. package包提供访问当前应用数据的功能，如获取元素、访问全局变量、获取资源文件等功能
# 3. 当此模块作为流程独立运行时执行main函数
# 4. 可视化流程中可以通过"调用模块"的指令使用此模块

import xbot
from xbot import print, sleep
from .import package
from .package import variables as glv
# 需通过Python 包管理 安装 requests
import requests
import json

def main(args):
    pass


def run_dify_workflow(Dify_workflow_Input):
    """
    执行Dify工作流并返回结果。

    :param Dify_workflow_Input: 工作流的输入，字典类型,且所有值均为字符串类型
    :return: 工作流的输出，字典类型
    """
    # url = "https://api.dify.ai/v1/workflows/run"
    url = "http://localhost/v1/chat-messages"
    headers = {
        'Authorization': f'Bearer app-mMl5Qiq3Gv9yGoJGjDRjH8m6',
        'Content-Type': 'application/json',
    }
    #保证所有输入的值均为字符串类型
    Dify_workflow_Input = {k: str(v) for k, v in Dify_workflow_Input.items()}
    # 从入参中拆分 query 与其余 inputs（用于 Chat 应用）
    query = Dify_workflow_Input.get("query", "")
    inputs = {k: v for k, v in Dify_workflow_Input.items() if k != "query"}
    
    # 数据结构
    data = {
        "inputs": inputs,
        "query": query,
        "response_mode": "blocking",
        "user": "default-user-id"
    }

    proxies = {
        'https': 'http://127.0.0.1:7897',
        'http': 'http://127.0.0.1:7897'
    }

    try:
        # response = requests.post(url, headers=headers, json=data, proxies=proxies)
        response = requests.post(url, headers=headers, json=data)
        
        # 打印完整的响应内容，以便调试
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        print(f"执行结果: {response.json()['answer']}")
        
        response.raise_for_status()  # 如果响应状态码不是2xx，将引发异常

        result = response.json()
        if 'data' in result and 'outputs' in result['data']:
            Dify_workflow_Output = result['data']['outputs']
            return Dify_workflow_Output
        elif 'data' in result and 'answer' in result['data']:
            return {"answer": result['data']['answer']}
        elif 'answer' in result:
            # Chat 应用在 blocking 模式下，answer 位于响应顶层
            return {"answer": result['answer']}
        else:
            print("响应中没有找到预期的数据结构: ", json.dumps(result, ensure_ascii=False))
            return None
    except requests.RequestException as e:
        print(f"请求出错: {str(e)}")
        return None
