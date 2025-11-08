"""
高德地图天气查询工具
提供基于高德地图API的天气信息查询功能
"""

import requests
import json
from typing import Dict, Any, Optional

from config.settings import settings
from core.logger import app_logger


class AmapWeatherTool:
    """简化的高德地图天气查询工具"""
    
    def __init__(self):
        """初始化天气工具"""
        self.api_key = settings.api.amap_api_key
        self.base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        
        # 常见城市的adcode映射
        self.city_adcodes = {
            "北京": "110000",
            "上海": "310000", 
            "广州": "440100",
            "深圳": "440300",
            "杭州": "330100",
            "南京": "320100",
            "武汉": "420100",
            "成都": "510100",
            "西安": "610100",
            "重庆": "500000",
            "天津": "120000",
            "苏州": "320500",
            "郑州": "410100",
            "长沙": "430100",
            "东莞": "441900",
            "青岛": "370200",
            "沈阳": "210100",
            "宁波": "330200",
            "昆明": "530100",
            "佛山": "440600"
        }
        
        app_logger.info("高德天气工具初始化完成")
    
    def get_weather(self, city_name: str) -> Dict[str, Any]:
        """
        获取指定城市的天气信息
        
        Args:
            city_name: 城市名称
            
        Returns:
            Dict[str, Any]: 天气信息结果
        """
        try:
            # 获取城市adcode
            adcode = self._get_city_adcode(city_name)
            if not adcode:
                return {
                    "success": False,
                    "error": f"未找到城市 '{city_name}' 的信息"
                }
            
            # 调用天气API
            params = {
                "key": self.api_key,
                "city": adcode,
                "extensions": "base"  # 获取实况天气
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") == "1" and data.get("lives"):
                weather_info = data["lives"][0]
                formatted_data = self._format_weather_info(weather_info, city_name)
                
                app_logger.info(f"成功获取 {city_name} 的天气信息")
                return {
                    "success": True,
                    "data": formatted_data
                }
            else:
                error_msg = data.get("info", "未知错误")
                app_logger.error(f"天气API返回错误: {error_msg}")
                return {
                    "success": False,
                    "error": f"获取天气信息失败: {error_msg}"
                }
                
        except requests.exceptions.Timeout:
            app_logger.error(f"获取 {city_name} 天气信息超时")
            return {
                "success": False,
                "error": "请求超时，请稍后重试"
            }
        except requests.exceptions.RequestException as e:
            app_logger.error(f"天气API请求失败: {str(e)}")
            return {
                "success": False,
                "error": f"网络请求失败: {str(e)}"
            }
        except Exception as e:
            app_logger.error(f"获取天气信息时出现未知错误: {str(e)}")
            return {
                "success": False,
                "error": f"获取天气信息失败: {str(e)}"
            }
    
    def _get_city_adcode(self, city_name: str) -> Optional[str]:
        """获取城市的adcode"""
        city_name = city_name.strip()
        
        # 直接匹配
        if city_name in self.city_adcodes:
            return self.city_adcodes[city_name]
        
        # 去掉"市"后缀再匹配
        if city_name.endswith("市"):
            city_name_without_suffix = city_name[:-1]
            if city_name_without_suffix in self.city_adcodes:
                return self.city_adcodes[city_name_without_suffix]
        
        # 模糊匹配
        for city, adcode in self.city_adcodes.items():
            if city_name in city or city in city_name:
                return adcode
        
        return None
    
    def _format_weather_info(self, weather_info: Dict[str, Any], city_name: str) -> str:
        """格式化天气信息"""
        try:
            temperature = weather_info.get("temperature", "未知")
            weather = weather_info.get("weather", "未知")
            wind_direction = weather_info.get("winddirection", "未知")
            wind_power = weather_info.get("windpower", "未知")
            humidity = weather_info.get("humidity", "未知")
            report_time = weather_info.get("reporttime", "未知")
            
            formatted_info = f"""
            🌡️ 温度: {temperature}°C
            🌤️ 天气: {weather}
            💨 风向: {wind_direction}风
            🌪️ 风力: {wind_power}级
            💧 湿度: {humidity}%
            🕐 更新时间: {report_time}
                        """.strip()
            
            return formatted_info
            
        except Exception as e:
            app_logger.error(f"格式化天气信息失败: {str(e)}")
            return f"天气信息格式化失败: {str(e)}"


# 创建全局实例
amap_weather_tool = AmapWeatherTool()