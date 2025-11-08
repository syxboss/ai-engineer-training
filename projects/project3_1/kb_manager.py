"""
知识库管理器 - 实现知识库的动态管理功能
包括FAQ条目的增删改查、批量导入导出、版本管理等功能
"""
import os
import json
import shutil
import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import re

from data_loader import FAQDataLoader
from config import settings


class KnowledgeBaseManager:
    """知识库管理器"""
    
    def __init__(self, faq_file_path: str = None, index_path: str = None):
        """初始化知识库管理器"""
        self.faq_file_path = faq_file_path or settings.faq_file_path
        self.index_path = index_path or settings.faiss_index_path
        self.data_loader = FAQDataLoader()
        
        # 版本管理相关路径
        self.backup_dir = "./data/backups"
        self.version_file = "./data/version.json"
        
        # 确保必要目录存在
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.version_file), exist_ok=True)
        
        # 初始化版本信息
        self._init_version_info()
    
    def _init_version_info(self):
        """初始化版本信息"""
        if not os.path.exists(self.version_file):
            version_info = {
                "current_version": "1.0.0",
                "versions": [],
                "created_at": datetime.datetime.now().isoformat()
            }
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(version_info, f, ensure_ascii=False, indent=2)
    
    def _get_version_info(self) -> Dict[str, Any]:
        """获取版本信息"""
        with open(self.version_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _update_version_info(self, version_info: Dict[str, Any]):
        """更新版本信息"""
        with open(self.version_file, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, ensure_ascii=False, indent=2)
    
    def _generate_version_number(self) -> str:
        """生成新的版本号"""
        version_info = self._get_version_info()
        current = version_info["current_version"]
        major, minor, patch = map(int, current.split('.'))
        return f"{major}.{minor}.{patch + 1}"
    
    # FAQ条目的增删改查功能
    def get_all_faqs(self) -> List[Dict[str, Any]]:
        """获取所有FAQ条目"""
        if not os.path.exists(self.faq_file_path):
            return []
        return self.data_loader.parse_faq_file(self.faq_file_path)
    
    def get_faq_by_id(self, faq_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取FAQ条目"""
        faqs = self.get_all_faqs()
        for faq in faqs:
            if faq['id'] == faq_id:
                return faq
        return None
    
    def search_faqs(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索FAQ条目"""
        faqs = self.get_all_faqs()
        results = []
        keyword_lower = keyword.lower()
        
        for faq in faqs:
            if (keyword_lower in faq['question'].lower() or 
                keyword_lower in faq['answer'].lower()):
                results.append(faq)
        
        return results
    
    def add_faq(self, question: str, answer: str, auto_rebuild: bool = True) -> Dict[str, Any]:
        """添加新的FAQ条目"""
        faqs = self.get_all_faqs()
        
        # 生成新的ID
        new_id = max([faq['id'] for faq in faqs], default=0) + 1
        
        new_faq = {
            'id': new_id,
            'question': question.strip(),
            'answer': answer.strip()
        }
        
        faqs.append(new_faq)
        self._save_faqs_to_file(faqs)
        
        if auto_rebuild:
            self.rebuild_index()
        
        return new_faq
    
    def update_faq(self, faq_id: int, question: str = None, answer: str = None, 
                   auto_rebuild: bool = True) -> bool:
        """更新FAQ条目"""
        faqs = self.get_all_faqs()
        
        for faq in faqs:
            if faq['id'] == faq_id:
                if question is not None:
                    faq['question'] = question.strip()
                if answer is not None:
                    faq['answer'] = answer.strip()
                
                self._save_faqs_to_file(faqs)
                
                if auto_rebuild:
                    self.rebuild_index()
                
                return True
        
        return False
    
    def delete_faq(self, faq_id: int, auto_rebuild: bool = True) -> bool:
        """删除FAQ条目"""
        faqs = self.get_all_faqs()
        original_count = len(faqs)
        
        faqs = [faq for faq in faqs if faq['id'] != faq_id]
        
        if len(faqs) < original_count:
            # 重新分配ID以保持连续性
            for i, faq in enumerate(faqs, 1):
                faq['id'] = i
            
            self._save_faqs_to_file(faqs)
            
            if auto_rebuild:
                self.rebuild_index()
            
            return True
        
        return False
    
    def _save_faqs_to_file(self, faqs: List[Dict[str, Any]]):
        """将FAQ列表保存到文件"""
        content = ""
        for faq in faqs:
            content += f"Q: {faq['question']}\n"
            content += f"A: {faq['answer']}\n\n"
        
        with open(self.faq_file_path, 'w', encoding='utf-8') as f:
            f.write(content.rstrip() + '\n')
    
    # 动态知识库更新功能
    def rebuild_index(self, force: bool = True) -> bool:
        """重建向量索引"""
        try:
            print("正在重建向量索引...")
            index = self.data_loader.initialize_faq_system(force_rebuild=force)
            print("向量索引重建完成")
            return True
        except Exception as e:
            print(f"重建索引失败: {str(e)}")
            return False
    
    def update_knowledge_base(self, new_faqs: List[Dict[str, str]], 
                            merge_strategy: str = "append") -> bool:
        """更新知识库
        
        Args:
            new_faqs: 新的FAQ列表，格式为[{"question": "...", "answer": "..."}]
            merge_strategy: 合并策略，"append"(追加) 或 "replace"(替换)
        """
        try:
            if merge_strategy == "replace":
                # 替换模式：清空现有FAQ，使用新的FAQ
                faqs = []
                for i, faq_data in enumerate(new_faqs, 1):
                    faqs.append({
                        'id': i,
                        'question': faq_data['question'].strip(),
                        'answer': faq_data['answer'].strip()
                    })
            else:
                # 追加模式：在现有FAQ基础上添加新的FAQ
                faqs = self.get_all_faqs()
                current_max_id = max([faq['id'] for faq in faqs], default=0)
                
                for i, faq_data in enumerate(new_faqs, 1):
                    faqs.append({
                        'id': current_max_id + i,
                        'question': faq_data['question'].strip(),
                        'answer': faq_data['answer'].strip()
                    })
            
            self._save_faqs_to_file(faqs)
            self.rebuild_index()
            
            print(f"知识库更新完成，共{len(faqs)}个FAQ条目")
            return True
            
        except Exception as e:
            print(f"更新知识库失败: {str(e)}")
            return False
    
    # 批量导入导出功能
    def export_to_json(self, output_path: str) -> bool:
        """导出FAQ到JSON文件"""
        try:
            faqs = self.get_all_faqs()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(faqs, f, ensure_ascii=False, indent=2)
            print(f"FAQ已导出到: {output_path}")
            return True
        except Exception as e:
            print(f"导出到JSON失败: {str(e)}")
            return False
    
    def export_to_csv(self, output_path: str) -> bool:
        """导出FAQ到CSV文件"""
        try:
            faqs = self.get_all_faqs()
            df = pd.DataFrame(faqs)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"FAQ已导出到: {output_path}")
            return True
        except Exception as e:
            print(f"导出到CSV失败: {str(e)}")
            return False
    
    def export_to_txt(self, output_path: str) -> bool:
        """导出FAQ到TXT文件"""
        try:
            shutil.copy2(self.faq_file_path, output_path)
            print(f"FAQ已导出到: {output_path}")
            return True
        except Exception as e:
            print(f"导出到TXT失败: {str(e)}")
            return False
    
    def import_from_json(self, input_path: str, merge_strategy: str = "append") -> bool:
        """从JSON文件导入FAQ"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换格式
            new_faqs = []
            for item in data:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    new_faqs.append({
                        'question': item['question'],
                        'answer': item['answer']
                    })
            
            return self.update_knowledge_base(new_faqs, merge_strategy)
            
        except Exception as e:
            print(f"从JSON导入失败: {str(e)}")
            return False
    
    def import_from_csv(self, input_path: str, merge_strategy: str = "append") -> bool:
        """从CSV文件导入FAQ"""
        try:
            df = pd.read_csv(input_path, encoding='utf-8-sig')
            
            # 检查必要的列
            if 'question' not in df.columns or 'answer' not in df.columns:
                print("CSV文件必须包含'question'和'answer'列")
                return False
            
            new_faqs = []
            for _, row in df.iterrows():
                new_faqs.append({
                    'question': str(row['question']),
                    'answer': str(row['answer'])
                })
            
            return self.update_knowledge_base(new_faqs, merge_strategy)
            
        except Exception as e:
            print(f"从CSV导入失败: {str(e)}")
            return False
    
    def import_from_txt(self, input_path: str, merge_strategy: str = "append") -> bool:
        """从TXT文件导入FAQ"""
        try:
            # 解析TXT文件
            new_faqs_data = self.data_loader.parse_faq_file(input_path)
            
            new_faqs = []
            for faq in new_faqs_data:
                new_faqs.append({
                    'question': faq['question'],
                    'answer': faq['answer']
                })
            
            return self.update_knowledge_base(new_faqs, merge_strategy)
            
        except Exception as e:
            print(f"从TXT导入失败: {str(e)}")
            return False
    
    # 知识库版本管理
    def create_backup(self, description: str = "") -> str:
        """创建知识库备份"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version_number = self._generate_version_number()
            backup_name = f"backup_{version_number}_{timestamp}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            os.makedirs(backup_path, exist_ok=True)
            
            # 备份FAQ文件
            if os.path.exists(self.faq_file_path):
                shutil.copy2(self.faq_file_path, 
                           os.path.join(backup_path, "FAQ.txt"))
            
            # 备份索引文件
            if os.path.exists(self.index_path):
                shutil.copytree(self.index_path, 
                              os.path.join(backup_path, "faiss_index"))
            
            # 更新版本信息
            version_info = self._get_version_info()
            version_record = {
                "version": version_number,
                "backup_name": backup_name,
                "timestamp": timestamp,
                "description": description,
                "faq_count": len(self.get_all_faqs())
            }
            
            version_info["versions"].append(version_record)
            version_info["current_version"] = version_number
            self._update_version_info(version_info)
            
            print(f"备份创建成功: {backup_name}")
            return backup_name
            
        except Exception as e:
            print(f"创建备份失败: {str(e)}")
            return ""
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份版本"""
        version_info = self._get_version_info()
        return version_info.get("versions", [])
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """从备份恢复知识库"""
        try:
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            if not os.path.exists(backup_path):
                print(f"备份不存在: {backup_name}")
                return False
            
            # 恢复FAQ文件
            backup_faq_path = os.path.join(backup_path, "FAQ.txt")
            if os.path.exists(backup_faq_path):
                shutil.copy2(backup_faq_path, self.faq_file_path)
            
            # 恢复索引文件
            backup_index_path = os.path.join(backup_path, "faiss_index")
            if os.path.exists(backup_index_path):
                if os.path.exists(self.index_path):
                    shutil.rmtree(self.index_path)
                shutil.copytree(backup_index_path, self.index_path)
            
            print(f"从备份恢复成功: {backup_name}")
            return True
            
        except Exception as e:
            print(f"从备份恢复失败: {str(e)}")
            return False
    
    def delete_backup(self, backup_name: str) -> bool:
        """删除指定备份"""
        try:
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
                
                # 从版本信息中移除
                version_info = self._get_version_info()
                version_info["versions"] = [
                    v for v in version_info["versions"] 
                    if v["backup_name"] != backup_name
                ]
                self._update_version_info(version_info)
                
                print(f"备份已删除: {backup_name}")
                return True
            else:
                print(f"备份不存在: {backup_name}")
                return False
                
        except Exception as e:
            print(f"删除备份失败: {str(e)}")
            return False
    
    # 统计和信息功能
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        faqs = self.get_all_faqs()
        
        stats = {
            "total_faqs": len(faqs),
            "avg_question_length": 0,
            "avg_answer_length": 0,
            "index_exists": os.path.exists(self.index_path),
            "last_updated": "",
            "version_info": self._get_version_info()
        }
        
        if faqs:
            stats["avg_question_length"] = sum(len(faq['question']) for faq in faqs) / len(faqs)
            stats["avg_answer_length"] = sum(len(faq['answer']) for faq in faqs) / len(faqs)
        
        if os.path.exists(self.faq_file_path):
            stats["last_updated"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.faq_file_path)
            ).isoformat()
        
        return stats
    
    def validate_knowledge_base(self) -> Dict[str, Any]:
        """验证知识库完整性"""
        issues = []
        warnings = []
        
        # 检查FAQ文件
        if not os.path.exists(self.faq_file_path):
            issues.append("FAQ文件不存在")
        else:
            faqs = self.get_all_faqs()
            if not faqs:
                issues.append("FAQ文件为空")
            
            # 检查FAQ格式
            for faq in faqs:
                if not faq.get('question', '').strip():
                    issues.append(f"FAQ ID {faq.get('id', 'unknown')} 缺少问题")
                if not faq.get('answer', '').strip():
                    issues.append(f"FAQ ID {faq.get('id', 'unknown')} 缺少答案")
        
        # 检查索引文件
        if not os.path.exists(self.index_path):
            warnings.append("向量索引不存在，建议重建索引")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "checked_at": datetime.datetime.now().isoformat()
        }


def main():
    """测试知识库管理器功能"""
    kb_manager = KnowledgeBaseManager()
    
    print("=== 知识库管理器测试 ===")
    
    # 获取统计信息
    stats = kb_manager.get_statistics()
    print(f"当前FAQ数量: {stats['total_faqs']}")
    
    # 验证知识库
    validation = kb_manager.validate_knowledge_base()
    print(f"知识库验证: {'通过' if validation['is_valid'] else '失败'}")
    
    # 创建备份
    backup_name = kb_manager.create_backup("测试备份")
    if backup_name:
        print(f"备份创建成功: {backup_name}")
    
    # 列出备份
    backups = kb_manager.list_backups()
    print(f"备份数量: {len(backups)}")


if __name__ == "__main__":
    main()