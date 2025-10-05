import os
from openai import OpenAI
from dotenv import load_dotenv

# 默认环境变量优先级高，设置 override=True 后，会强制覆盖默认环境变量
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE')
print(f"-- debug -- openai api key is {api_key[0:10]}******")

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


response = client.chat.completions.create(
    model="o3-mini",
    messages=[
        {"role": "user", "content": "Hello world!"}
    ]
)

print(response.choices[0].message.content)


# 正常会输出结果：Hello! It's great to see you. How can I assist you today?