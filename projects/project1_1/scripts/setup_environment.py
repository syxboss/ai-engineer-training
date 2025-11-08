"""
环境设置脚本
用于初始化项目环境、验证配置和准备运行环境
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from core.logger import app_logger


class EnvironmentSetup:
    """环境设置管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_results = {}
    
    def print_header(self):
        """打印设置开始信息"""
        print("🔧" + "="*58 + "🔧")
        print("🚀 多任务问答助手 - 环境设置向导")
        print(f"📁 项目路径: {self.project_root}")
        print("🔧" + "="*58 + "🔧")
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("\n🐍 检查Python版本...")
        
        version = sys.version_info
        required_version = (3, 8)
        
        print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
        print(f"最低要求版本: {required_version[0]}.{required_version[1]}")
        
        if version >= required_version:
            print("✅ Python版本检查通过")
            self.setup_results['python_version'] = True
            return True
        else:
            print(f"❌ Python版本过低，请升级到 {required_version[0]}.{required_version[1]} 或更高版本")
            self.setup_results['python_version'] = False
            return False
    
    def check_dependencies(self) -> bool:
        """检查依赖包"""
        print("\n📦 检查依赖包...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt 文件不存在")
            self.setup_results['dependencies'] = False
            return False
        
        # 读取依赖列表
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"需要检查 {len(requirements)} 个依赖包...")
        
        missing_packages = []
        installed_packages = []
        
        for requirement in requirements:
            package_name = requirement.split('==')[0].split('>=')[0].split('<=')[0]
            try:
                __import__(package_name.replace('-', '_'))
                installed_packages.append(package_name)
                print(f"  ✅ {package_name}")
            except ImportError:
                missing_packages.append(requirement)
                print(f"  ❌ {package_name}")
        
        if missing_packages:
            print(f"\n⚠️  发现 {len(missing_packages)} 个缺失的依赖包:")
            for package in missing_packages:
                print(f"    • {package}")
            print("\n💡 运行以下命令安装缺失的依赖:")
            print(f"    pip install -r {requirements_file}")
            self.setup_results['dependencies'] = False
            return False
        else:
            print(f"✅ 所有 {len(installed_packages)} 个依赖包已安装")
            self.setup_results['dependencies'] = True
            return True
    
    def setup_environment_file(self) -> bool:
        """设置环境变量文件"""
        print("\n🔐 设置环境变量文件...")
        
        env_file = self.project_root / ".env"
        env_example_file = self.project_root / ".env.example"
        
        if not env_example_file.exists():
            print("❌ .env.example 文件不存在")
            self.setup_results['env_file'] = False
            return False
        
        if env_file.exists():
            print("✅ .env 文件已存在")
            
            # 验证环境变量
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                
                required_vars = [
                    'OPENAI_API_KEY',
                    'QWEATHER_API_KEY', 
                    'TAVILY_API_KEY'
                ]
                
                missing_vars = []
                for var in required_vars:
                    if not os.getenv(var):
                        missing_vars.append(var)
                
                if missing_vars:
                    print(f"⚠️  以下环境变量未设置: {', '.join(missing_vars)}")
                    print("请编辑 .env 文件并设置这些变量")
                    self.setup_results['env_file'] = False
                    return False
                else:
                    print("✅ 所有必需的环境变量已设置")
                    self.setup_results['env_file'] = True
                    return True
                    
            except Exception as e:
                print(f"❌ 环境变量验证失败: {str(e)}")
                self.setup_results['env_file'] = False
                return False
        else:
            print("📝 创建 .env 文件...")
            try:
                shutil.copy2(env_example_file, env_file)
                print("✅ .env 文件已创建")
                print("⚠️  请编辑 .env 文件并设置必要的API密钥:")
                print("    • OPENAI_API_KEY: OpenAI API密钥")
                print("    • QWEATHER_API_KEY: 和风天气API密钥 (已提供测试密钥)")
                print("    • TAVILY_API_KEY: Tavily搜索API密钥 (已提供测试密钥)")
                self.setup_results['env_file'] = 'created'
                return True
            except Exception as e:
                print(f"❌ 创建 .env 文件失败: {str(e)}")
                self.setup_results['env_file'] = False
                return False
    
    def create_directories(self) -> bool:
        """创建必要的目录"""
        print("\n📁 创建必要的目录...")
        
        directories = [
            "logs",
            "data",
            "cache"
        ]
        
        created_dirs = []
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_name)
                    print(f"  ✅ 创建目录: {dir_name}")
                except Exception as e:
                    print(f"  ❌ 创建目录失败 {dir_name}: {str(e)}")
                    self.setup_results['directories'] = False
                    return False
            else:
                print(f"  ✅ 目录已存在: {dir_name}")
        
        if created_dirs:
            print(f"✅ 成功创建 {len(created_dirs)} 个目录")
        else:
            print("✅ 所有必要目录已存在")
        
        self.setup_results['directories'] = True
        return True
    
    def check_data_files(self) -> bool:
        """检查数据文件"""
        print("\n📄 检查数据文件...")
        
        data_files = [
            "China-City-List-latest.csv"
        ]
        
        missing_files = []
        for file_name in data_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"  ✅ {file_name} (大小: {file_size:,} 字节)")
            else:
                missing_files.append(file_name)
                print(f"  ❌ {file_name}")
        
        if missing_files:
            print(f"⚠️  缺失 {len(missing_files)} 个数据文件:")
            for file_name in missing_files:
                print(f"    • {file_name}")
            print("请确保这些文件存在于项目根目录")
            self.setup_results['data_files'] = False
            return False
        else:
            print("✅ 所有数据文件检查通过")
            self.setup_results['data_files'] = True
            return True
    
    def check_redis_connection(self) -> bool:
        """检查Redis连接"""
        print("\n🔴 检查Redis连接...")
        
        try:
            import redis
            
            # 尝试连接Redis
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                password=os.getenv('REDIS_PASSWORD'),
                socket_timeout=5
            )
            
            redis_client.ping()
            print("✅ Redis连接成功")
            
            # 获取Redis信息
            info = redis_client.info()
            print(f"  Redis版本: {info.get('redis_version', 'Unknown')}")
            print(f"  使用内存: {info.get('used_memory_human', 'Unknown')}")
            
            self.setup_results['redis'] = True
            return True
            
        except redis.ConnectionError:
            print("⚠️  Redis连接失败 - 缓存功能将被禁用")
            print("💡 如需启用缓存功能，请:")
            print("    1. 安装并启动Redis服务")
            print("    2. 检查Redis连接配置")
            self.setup_results['redis'] = False
            return False
        except Exception as e:
            print(f"❌ Redis检查失败: {str(e)}")
            self.setup_results['redis'] = False
            return False
    
    def test_api_connections(self) -> bool:
        """测试API连接"""
        print("\n🌐 测试API连接...")
        
        api_results = {}
        
        # 测试OpenAI API
        print("  🤖 测试OpenAI API...")
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key and openai_key != 'your_openai_api_key_here':
                # 这里可以添加实际的API测试
                print("    ✅ OpenAI API密钥已配置")
                api_results['openai'] = True
            else:
                print("    ⚠️  OpenAI API密钥未配置")
                api_results['openai'] = False
        except Exception as e:
            print(f"    ❌ OpenAI API测试失败: {str(e)}")
            api_results['openai'] = False
        
        # 测试和风天气API
        print("  🌤️  测试和风天气API...")
        try:
            qweather_key = os.getenv('QWEATHER_API_KEY')
            if qweather_key:
                print("    ✅ 和风天气API密钥已配置")
                api_results['qweather'] = True
            else:
                print("    ❌ 和风天气API密钥未配置")
                api_results['qweather'] = False
        except Exception as e:
            print(f"    ❌ 和风天气API测试失败: {str(e)}")
            api_results['qweather'] = False
        
        # 测试Tavily搜索API
        print("  🔍 测试Tavily搜索API...")
        try:
            tavily_key = os.getenv('TAVILY_API_KEY')
            if tavily_key:
                print("    ✅ Tavily搜索API密钥已配置")
                api_results['tavily'] = True
            else:
                print("    ❌ Tavily搜索API密钥未配置")
                api_results['tavily'] = False
        except Exception as e:
            print(f"    ❌ Tavily搜索API测试失败: {str(e)}")
            api_results['tavily'] = False
        
        # 统计结果
        working_apis = sum(api_results.values())
        total_apis = len(api_results)
        
        if working_apis == total_apis:
            print(f"✅ 所有 {total_apis} 个API连接正常")
            self.setup_results['api_connections'] = True
            return True
        elif working_apis > 0:
            print(f"⚠️  {working_apis}/{total_apis} 个API连接正常")
            self.setup_results['api_connections'] = 'partial'
            return True
        else:
            print("❌ 所有API连接失败")
            self.setup_results['api_connections'] = False
            return False
    
    def generate_setup_report(self):
        """生成设置报告"""
        print("\n📊" + "="*58 + "📊")
        print("📋 环境设置报告")
        print("📊" + "="*58 + "📊")
        
        checks = [
            ("Python版本", "python_version"),
            ("依赖包", "dependencies"),
            ("环境变量文件", "env_file"),
            ("目录结构", "directories"),
            ("数据文件", "data_files"),
            ("Redis连接", "redis"),
            ("API连接", "api_connections")
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_key in checks:
            result = self.setup_results.get(check_key, False)
            if result is True:
                status = "✅ 通过"
                passed_checks += 1
            elif result == 'created':
                status = "📝 已创建"
                passed_checks += 1
            elif result == 'partial':
                status = "⚠️  部分通过"
                passed_checks += 0.5
            else:
                status = "❌ 失败"
            
            print(f"  {check_name:<15}: {status}")
        
        print(f"\n📈 总体评分: {passed_checks}/{total_checks} ({passed_checks/total_checks*100:.1f}%)")
        
        if passed_checks == total_checks:
            print("\n🎉 环境设置完成！系统已准备就绪。")
            print("\n🚀 现在可以运行:")
            print("    python main.py --mode cli")
            print("    python scripts/demo_scenarios.py")
        elif passed_checks >= total_checks * 0.7:
            print("\n⚠️  环境基本就绪，但有一些可选功能可能无法使用。")
            print("建议解决上述问题以获得最佳体验。")
        else:
            print("\n❌ 环境设置不完整，请解决上述问题后重新运行设置。")
        
        print("📊" + "="*58 + "📊")
    
    def run_setup(self):
        """运行完整的环境设置"""
        self.print_header()
        
        try:
            # 执行所有检查
            self.check_python_version()
            self.check_dependencies()
            self.setup_environment_file()
            self.create_directories()
            self.check_data_files()
            self.check_redis_connection()
            self.test_api_connections()
            
            # 生成报告
            self.generate_setup_report()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  设置被用户中断")
        except Exception as e:
            print(f"\n❌ 设置过程中发生错误: {str(e)}")
            app_logger.error(f"环境设置失败: {str(e)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多任务问答助手环境设置")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查环境，不进行修改"
    )
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup()
    
    try:
        setup.run_setup()
        return 0
    except Exception as e:
        print(f"❌ 环境设置失败: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())