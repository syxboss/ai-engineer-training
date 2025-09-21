import http.client
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

conn = http.client.HTTPSConnection("aihubmix.com")
payload = json.dumps({
   "model": "o3-mini",
   "messages": [
      {
         "role": "user",
         "content": "晚上好"
      }
   ],
   #"max_tokens": 1688,
   "temperature": 0.5,
   "stream": False
})

# 从环境变量获取API token
api_token = os.getenv('OPENAI_API_KEY')  # 或者使用其他环境变量名

headers = {
   'Content-Type': 'application/json',
   'Authorization': f'Bearer {api_token}'
}
conn.request("POST", "/v1/chat/completions", payload, headers)
res = conn.getresponse()
data = res.read()

# 打印原始响应
print("原始响应:")
print(data.decode("utf-8"))

# 解析JSON并提取消息内容
response_json = json.loads(data.decode("utf-8"))
message_content = response_json['choices'][0]['message']['content']
print("\n提取的消息内容:")
print(message_content)


#原始响应:
#{"id":"chatcmpl-CI7CKgO2FsWt365jw3ECPpx3Dlnki","model":"o3-mini-2025-01-31","object":"chat.completion","created":1758433608,"choices":[{"index":0,"message":{"role":"assistant","content":"晚上好！有什么我可以帮您的吗？"},"finish_reason":"stop"}],"system_fingerprint":"fp_e1882df059","usage":{"prompt_tokens":8,"completion_tokens":85,"total_tokens":93,"prompt_tokens_details":{"audio_tokens":0,"cached_tokens":0},"completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,"reasoning_tokens":64,"rejected_prediction_tokens":0}}}