"""
配置管理模块
使用Pydantic进行配置验证和类型检查，符合企业级开发规范
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class APISettings(BaseSettings):
    """API相关配置"""
    
    # OpenAI配置
    openai_api_key: str = Field(..., description="OpenAI API密钥")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API基础URL")
    
    # 高德地图API配置
    amap_api_key: str = Field(..., description="高德地图API密钥")
    amap_base_url: str = Field(default="https://restapi.amap.com/v3", description="高德地图API基础URL")
    
    # Tavily搜索API配置
    tavily_api_key: str = Field(..., description="Tavily搜索API密钥")
    
    @validator('openai_api_key', 'amap_api_key', 'tavily_api_key')
    def validate_api_keys(cls, v):
        """验证API密钥不能为空"""
        if not v or v.strip() == "":
            raise ValueError("API密钥不能为空")
        return v.strip()
    
    class Config:
        env_prefix = ""
        case_sensitive = False


class RedisSettings(BaseSettings):
    """Redis缓存配置"""
    
    redis_host: str = Field(default="localhost", description="Redis主机地址")
    redis_port: int = Field(default=6379, description="Redis端口")
    redis_db: int = Field(default=0, description="Redis数据库编号")
    redis_password: Optional[str] = Field(default=None, description="Redis密码")
    
    @validator('redis_port')
    def validate_port(cls, v):
        """验证端口范围"""
        if not 1 <= v <= 65535:
            raise ValueError("端口号必须在1-65535范围内")
        return v
    
    class Config:
        env_prefix = ""
        case_sensitive = False


class AppSettings(BaseSettings):
    """应用程序配置"""
    
    app_name: str = Field(default="MultiTaskQAAssistant", description="应用名称")
    app_version: str = Field(default="1.0.0", description="应用版本")
    log_level: str = Field(default="INFO", description="日志级别")
    max_conversation_history: int = Field(default=50, description="最大对话历史记录数")
    cache_ttl: int = Field(default=3600, description="缓存过期时间(秒)")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是以下之一: {valid_levels}")
        return v.upper()
    
    @validator('max_conversation_history', 'cache_ttl')
    def validate_positive_int(cls, v):
        """验证正整数"""
        if v <= 0:
            raise ValueError("值必须大于0")
        return v
    
    class Config:
        env_prefix = ""
        case_sensitive = False


class Settings:
    """
    全局配置管理器
    
    采用单例模式，确保配置的一致性和性能
    提供统一的配置访问接口
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            try:
                self.api = APISettings()
                self.redis = RedisSettings()
                self.app = AppSettings()
                self._initialized = True
            except Exception as e:
                raise RuntimeError(f"配置初始化失败: {str(e)}")
    
    def get_city_data_path(self) -> str:
        """获取城市数据文件路径"""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "China-City-List-latest.csv")
    
    def validate_all(self) -> bool:
        """验证所有配置"""
        try:
            # 检查必需的API密钥
            if not self.api.openai_api_key:
                print("❌ OpenAI API密钥未配置")
                return False
            
            if not self.api.amap_api_key:
                print("❌ 高德地图API密钥未配置")
                return False
            
            print("✅ 配置验证通过")
            return True
            
        except Exception as e:
            print(f"❌ 配置验证失败: {str(e)}")
            return False


# 全局配置实例
settings = Settings()