"""
训练脚本 - 从FAQ.txt加载数据到faiss向量数据库并持久化
"""
import os
from data_loader import FAQDataLoader
from config import settings


def main():
    """主函数：从FAQ.txt加载数据并构建faiss索引"""
    print("开始构建FAQ索引...")
    
    # 检查API密钥
    if not settings.dashscope_api_key:
        print("错误：请设置DASHSCOPE_API_KEY环境变量")
        return
    
    # 检查FAQ文件
    if not os.path.exists(settings.faq_file_path):
        print(f"错误：FAQ文件不存在: {settings.faq_file_path}")
        return
    
    # 创建数据加载器并构建索引
    data_loader = FAQDataLoader()
    index = data_loader.initialize_faq_system(force_rebuild=True)
    
    if index:
        print("索引构建并持久化成功！")
    else:
        print("索引构建失败")


if __name__ == "__main__":
    main()