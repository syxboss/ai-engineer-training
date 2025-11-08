import redis
import json
import time
from datetime import datetime

class SimpleChatMessageHistory:
    def __init__(self, session_id, redis_url="redis://localhost:6379", ttl=10):
        """
        简化的Redis聊天消息历史管理
        
        Args:
            session_id: 会话ID
            redis_url: Redis连接URL
            ttl: 消息过期时间（秒）
        """
        self.session_id = session_id
        self.ttl = ttl
        self.redis_client = redis.from_url(redis_url)
        self.key = f"chat_history:{session_id}"
        
    def add_message(self, message_type, content):
        """添加消息到历史记录"""
        message = {
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # 将消息添加到Redis列表
        self.redis_client.lpush(self.key, json.dumps(message))
        
        # 设置过期时间
        self.redis_client.expire(self.key, self.ttl)
        
        print(f"添加消息: {message_type} - {content}")
        
    def get_messages(self):
        """获取所有消息"""
        messages = self.redis_client.lrange(self.key, 0, -1)
        result = []
        for msg in reversed(messages):  # 反转以保持时间顺序
            try:
                parsed_msg = json.loads(msg.decode('utf-8'))
                result.append(parsed_msg)
            except json.JSONDecodeError:
                continue
        return result
        
    def clear(self):
        """清除会话历史"""
        self.redis_client.delete(self.key)
        print("历史记录已清除")
        
    def check_ttl(self):
        """检查剩余TTL时间"""
        ttl = self.redis_client.ttl(self.key)
        if ttl == -2:
            return "键不存在"
        elif ttl == -1:
            return "键存在但没有设置过期时间"
        else:
            return f"剩余 {ttl} 秒"

def main():
    print("=== Redis消息历史TTL演示：对话遗忘功能 ===")
    
    # 创建聊天历史实例
    history = SimpleChatMessageHistory(
        session_id="user123",
        redis_url="redis://localhost:6379",
        ttl=8  # 8秒后过期，更快演示
    )
    
    try:
        # 清除之前的数据
        history.clear()
        
        # 模拟一段对话
        print("\n 开始一段对话...")
        history.add_message("human", "我叫张三，今年25岁")
        history.add_message("ai", "你好张三！很高兴认识你，25岁正是年轻有为的年纪。")
        history.add_message("human", "我喜欢编程，特别是Python")
        history.add_message("ai", "太棒了！Python是一门很优秀的编程语言，你主要用它做什么项目呢？")
        history.add_message("human", "我在做一个聊天机器人项目")
        history.add_message("ai", "聊天机器人很有趣！你是用什么框架开发的？需要什么帮助吗？")
        
        # 显示完整对话历史
        print("\n 当前完整对话历史:")
        messages = history.get_messages()
        for i, msg in enumerate(messages, 1):
            print(f"  {i}. [{msg['type']}] {msg['content']}")
        
        print(f"\n 对话将在 {history.ttl} 秒后自动遗忘...")
        print(f"   当前TTL状态: {history.check_ttl()}")
        
        # 倒计时演示
        print("\n⏳ TTL倒计时 (对话遗忘倒计时):")
        for i in range(10):
            ttl_status = history.check_ttl()
            if "键不存在" in ttl_status:
                print(f"  第{i+1}秒: 🔥 对话已被遗忘！")
                break
            else:
                print(f"  第{i+1}秒: {ttl_status}")
            time.sleep(1)
        
        # 演示对话遗忘效果
        print("\n 验证AI遗忘效果...")
        
        # 尝试获取历史消息
        forgotten_messages = history.get_messages()
        if not forgotten_messages:
            print("   确认：对话历史已从Redis中完全删除！")
        else:
            print(f"   意外：仍有 {len(forgotten_messages)} 条对话记录存在")
            
        print(f"\n 遗忘效果总结:")
        print(f"   - 遗忘前对话数量: 6条")
        print(f"   - 遗忘后对话数量: {len(forgotten_messages)}条")
            
    except redis.ConnectionError:
        print("错误：无法连接到Redis服务器")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()



# # 需安装RediSearch模块
# git clone https://github.com/RediSearch/RediSearch.git
# cd RediSearch && make
# # 启动Redis并加载模块
# redis-server --loadmodule ./src/redisearch.so
