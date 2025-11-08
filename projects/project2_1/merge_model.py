import os
import json
import torch
import logging
import argparse
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_paths(lora_path, base_model_name, output_path):
    """
    验证输入路径和参数
    
    Args:
        lora_path: LoRA适配器路径
        base_model_name: 基础模型名称
        output_path: 输出路径
    """
    # 检查LoRA路径是否存在
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA适配器路径不存在: {lora_path}")
    
    # 检查LoRA路径中是否包含必要文件
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    for file in required_files:
        file_path = os.path.join(lora_path, file)
        if not os.path.exists(file_path):
            logger.warning(f"LoRA文件可能缺失: {file_path}")
    
    # 创建输出目录
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录已准备: {output_path}")

def load_label_mapping(lora_path):
    """
    加载标签映射文件
    
    Args:
        lora_path: LoRA适配器路径
        
    Returns:
        dict: 标签映射字典
    """
    label_mapping_path = os.path.join(lora_path, "label_mapping.json")
    
    if os.path.exists(label_mapping_path):
        try:
            with open(label_mapping_path, 'r', encoding='utf-8') as f:
                label_mapping = json.load(f)
            logger.info(f"加载标签映射成功，共 {len(label_mapping.get('id2label', {}))} 个类别")
            return label_mapping
        except Exception as e:
            logger.error(f"加载标签映射失败: {str(e)}")
            raise
    else:
        logger.warning(f"标签映射文件不存在: {label_mapping_path}")
        return None

def merge_lora_model(lora_path, base_model_name, output_path):
    """
    合并LoRA适配器与基础模型
    
    Args:
        lora_path: LoRA适配器路径
        base_model_name: 基础模型名称  
        output_path: 合并后模型保存路径
    """
    try:
        logger.info("=" * 60)
        logger.info("开始LoRA模型合并过程")
        logger.info("=" * 60)
        
        # 验证输入参数
        validate_paths(lora_path, base_model_name, output_path)
        
        # 加载标签映射
        label_mapping = load_label_mapping(lora_path)
        num_labels = len(label_mapping["id2label"]) if label_mapping else 2
        
        logger.info(f"LoRA适配器路径: {lora_path}")
        logger.info(f"基础模型: {base_model_name}")
        logger.info(f"输出路径: {output_path}")
        logger.info(f"标签数量: {num_labels}")
        
        # 加载分词器
        logger.info("正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            trust_remote_code=True
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("设置pad_token为eos_token")
        
        logger.info("分词器加载完成")
        
        # 加载基础模型
        logger.info("正在加载基础模型...")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info("基础模型加载完成")
        
        # 加载LoRA适配器
        logger.info("正在加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        logger.info("LoRA适配器加载完成")
        
        # 合并LoRA权重到基础模型
        logger.info("正在合并LoRA权重...")
        merged_model = model.merge_and_unload()
        logger.info("LoRA权重合并完成")
        
        # 保存合并后的模型
        logger.info("正在保存合并后的模型...")
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        logger.info("模型保存完成")
        
        # 保存分词器
        logger.info("正在保存分词器...")
        tokenizer.save_pretrained(output_path)
        logger.info("分词器保存完成")
        
        # 保存标签映射（如果存在）
        if label_mapping:
            label_mapping_output_path = os.path.join(output_path, "label_mapping.json")
            with open(label_mapping_output_path, 'w', encoding='utf-8') as f:
                json.dump(label_mapping, f, ensure_ascii=False, indent=2)
            logger.info("标签映射保存完成")
        
        # 保存模型配置信息
        config_info = {
            "base_model": base_model_name,
            "lora_adapter": lora_path,
            "num_labels": num_labels,
            "merge_timestamp": str(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
            "model_type": "merged_lora_model"
        }
        
        config_path = os.path.join(output_path, "merge_info.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)
        logger.info("合并信息保存完成")
        
        logger.info("=" * 60)
        logger.info("LoRA模型合并成功完成！")
        logger.info(f"合并后的模型已保存到: {output_path}")
        logger.info("=" * 60)
        
        # 显示输出目录内容
        logger.info("输出目录内容:")
        for item in os.listdir(output_path):
            item_path = os.path.join(output_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path) / (1024 * 1024)  # MB
                logger.info(f"  - {item} ({size:.2f} MB)")
            else:
                logger.info(f"  - {item}/ (目录)")
        
    except Exception as e:
        logger.error(f"模型合并过程中出现错误: {str(e)}")
        raise

def main():
    """
    主函数，处理命令行参数并执行模型合并
    """
    parser = argparse.ArgumentParser(
        description="合并LoRA适配器与Qwen3基础模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python merge_model.py
  python merge_model.py --lora_path ./custom-lora --output_path ./custom-merged
  python merge_model.py --base_model Qwen/Qwen3-7B --lora_path ./lora --output_path ./merged
        """
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./qwen3-intent-lora",
        help="LoRA适配器路径 (默认: ./qwen3-intent-lora)"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="基础模型名称 (默认: Qwen/Qwen3-8B)"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="./qwen3-intent-merged",
        help="合并后模型保存路径 (默认: ./qwen3-intent-merged)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志信息"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 执行模型合并
        merge_lora_model(
            lora_path=args.lora_path,
            base_model_name=args.base_model,
            output_path=args.output_path
        )
        
        print("\n✅ 模型合并成功完成！")
        print(f"📁 合并后的模型保存在: {args.output_path}")
        print("🚀 现在可以使用合并后的模型进行推理了")
        
    except KeyboardInterrupt:
        logger.info("用户中断了合并过程")
        print("\n❌ 合并过程被用户中断")
    except Exception as e:
        logger.error(f"合并失败: {str(e)}")
        print(f"\n❌ 合并失败: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())