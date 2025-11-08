"""
知识库管理器演示脚本
展示各种功能的使用方法
"""
from kb_manager import KnowledgeBaseManager
import json


def demo_crud_operations():
    """演示增删改查操作"""
    print("\n=== FAQ条目增删改查演示 ===")
    
    kb_manager = KnowledgeBaseManager()
    
    # 查看当前所有FAQ
    print("1. 查看当前所有FAQ:")
    faqs = kb_manager.get_all_faqs()
    print(f"   当前共有 {len(faqs)} 个FAQ条目")
    
    # 添加新FAQ
    print("\n2. 添加新FAQ:")
    new_faq = kb_manager.add_faq(
        question="测试问题：如何联系技术支持？",
        answer="您可以通过邮箱 tech@example.com 或电话 400-999-8888 联系我们的技术支持团队。",
        auto_rebuild=False  # 暂时不重建索引
    )
    print(f"   添加成功，新FAQ ID: {new_faq['id']}")
    
    # 搜索FAQ
    print("\n3. 搜索FAQ:")
    results = kb_manager.search_faqs("技术支持")
    print(f"   搜索'技术支持'找到 {len(results)} 个结果")
    for result in results:
        print(f"   - ID {result['id']}: {result['question'][:30]}...")
    
    # 更新FAQ
    print("\n4. 更新FAQ:")
    success = kb_manager.update_faq(
        faq_id=new_faq['id'],
        answer="您可以通过邮箱 tech@example.com、电话 400-999-8888 或在线客服联系我们的技术支持团队。工作时间：周一至周五 9:00-18:00。",
        auto_rebuild=False
    )
    print(f"   更新{'成功' if success else '失败'}")
    
    # 获取特定FAQ
    print("\n5. 获取特定FAQ:")
    faq = kb_manager.get_faq_by_id(new_faq['id'])
    if faq:
        print(f"   问题: {faq['question']}")
        print(f"   答案: {faq['answer'][:50]}...")
    
    # 删除FAQ
    print("\n6. 删除FAQ:")
    success = kb_manager.delete_faq(new_faq['id'], auto_rebuild=False)
    print(f"   删除{'成功' if success else '失败'}")


def demo_import_export():
    """演示导入导出功能"""
    print("\n=== 批量导入导出演示 ===")
    
    kb_manager = KnowledgeBaseManager()
    
    # 导出到不同格式
    print("1. 导出FAQ到不同格式:")
    
    # 导出到JSON
    json_path = "./data/export_demo.json"
    success = kb_manager.export_to_json(json_path)
    print(f"   导出到JSON: {'成功' if success else '失败'}")
    
    # 导出到CSV
    csv_path = "./data/export_demo.csv"
    success = kb_manager.export_to_csv(csv_path)
    print(f"   导出到CSV: {'成功' if success else '失败'}")
    
    # 导出到TXT
    txt_path = "./data/export_demo.txt"
    success = kb_manager.export_to_txt(txt_path)
    print(f"   导出到TXT: {'成功' if success else '失败'}")
    
    # 创建测试数据用于导入演示
    print("\n2. 创建测试导入数据:")
    test_faqs = [
        {
            "question": "如何重置密码？",
            "answer": "您可以在登录页面点击'忘记密码'链接，输入注册邮箱，系统会发送重置链接到您的邮箱。"
        },
        {
            "question": "支持哪些支付方式？",
            "answer": "我们支持支付宝、微信支付、银行卡支付和PayPal等多种支付方式。"
        }
    ]
    
    test_json_path = "./data/test_import.json"
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_faqs, f, ensure_ascii=False, indent=2)
    print(f"   测试数据已保存到: {test_json_path}")
    
    # 从JSON导入（追加模式）
    print("\n3. 从JSON导入FAQ（追加模式）:")
    original_count = len(kb_manager.get_all_faqs())
    success = kb_manager.import_from_json(test_json_path, merge_strategy="append")
    new_count = len(kb_manager.get_all_faqs())
    print(f"   导入{'成功' if success else '失败'}")
    print(f"   FAQ数量从 {original_count} 增加到 {new_count}")


def demo_version_management():
    """演示版本管理功能"""
    print("\n=== 版本管理演示 ===")
    
    kb_manager = KnowledgeBaseManager()
    
    # 创建备份
    print("1. 创建知识库备份:")
    backup_name = kb_manager.create_backup("演示备份 - 包含导入的测试数据")
    print(f"   备份名称: {backup_name}")
    
    # 列出所有备份
    print("\n2. 列出所有备份:")
    backups = kb_manager.list_backups()
    for backup in backups:
        print(f"   - 版本: {backup['version']}")
        print(f"     时间: {backup['timestamp']}")
        print(f"     描述: {backup['description']}")
        print(f"     FAQ数量: {backup['faq_count']}")
        print()
    
    # 修改知识库（删除一些FAQ）
    print("3. 修改知识库（删除最后两个FAQ）:")
    faqs = kb_manager.get_all_faqs()
    if len(faqs) >= 2:
        kb_manager.delete_faq(faqs[-1]['id'], auto_rebuild=False)
        kb_manager.delete_faq(faqs[-2]['id'], auto_rebuild=False)
        print(f"   删除后FAQ数量: {len(kb_manager.get_all_faqs())}")
    
    # 从备份恢复
    print("\n4. 从备份恢复知识库:")
    if backups:
        latest_backup = backups[-1]['backup_name']
        success = kb_manager.restore_from_backup(latest_backup)
        print(f"   恢复{'成功' if success else '失败'}")
        print(f"   恢复后FAQ数量: {len(kb_manager.get_all_faqs())}")


def demo_statistics_and_validation():
    """演示统计和验证功能"""
    print("\n=== 统计信息和验证演示 ===")
    
    kb_manager = KnowledgeBaseManager()
    
    # 获取统计信息
    print("1. 知识库统计信息:")
    stats = kb_manager.get_statistics()
    print(f"   FAQ总数: {stats['total_faqs']}")
    print(f"   平均问题长度: {stats['avg_question_length']:.1f} 字符")
    print(f"   平均答案长度: {stats['avg_answer_length']:.1f} 字符")
    print(f"   索引文件存在: {'是' if stats['index_exists'] else '否'}")
    print(f"   最后更新时间: {stats['last_updated']}")
    print(f"   当前版本: {stats['version_info']['current_version']}")
    
    # 验证知识库
    print("\n2. 知识库验证:")
    validation = kb_manager.validate_knowledge_base()
    print(f"   验证结果: {'通过' if validation['is_valid'] else '失败'}")
    
    if validation['issues']:
        print("   发现的问题:")
        for issue in validation['issues']:
            print(f"   - {issue}")
    
    if validation['warnings']:
        print("   警告信息:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
    
    print(f"   检查时间: {validation['checked_at']}")


def main():
    """主演示函数"""
    print("知识库管理器功能演示")
    print("=" * 50)
    
    try:
        # 演示各种功能
        demo_crud_operations()
        demo_import_export()
        demo_version_management()
        demo_statistics_and_validation()
        
        print("\n" + "=" * 50)
        print("演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()