import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import argparse

def merge_lora_weights(
    base_model_path: str,
    lora_model_path: str,
    output_path: str,
    device: str = "auto"
):
    """
    将LoRA权重合并到基础模型中
    
    Args:
        base_model_path: 基础模型路径
        lora_model_path: LoRA适配器路径
        output_path: 合并后模型保存路径
        device: 设备类型
    """
    
    print("开始合并LoRA权重...")
    
    try:
        # 1. 加载LoRA配置
        print("加载LoRA配置...")
        peft_config = PeftConfig.from_pretrained(lora_model_path)
        
        # 2. 加载基础模型
        print(f"加载基础模型: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # 3. 加载LoRA适配器
        print(f"加载LoRA适配器: {lora_model_path}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            torch_dtype=torch.float16
        )
        
        # 4. 合并权重
        print("合并LoRA权重到基础模型...")
        merged_model = model.merge_and_unload()
        
        # 5. 保存合并后的模型
        print(f"保存合并后的模型到: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        # 6. 保存分词器
        print("保存分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)
        
        # 7. 保存模型配置信息
        config_info = {
            "base_model": base_model_path,
            "lora_model": lora_model_path,
            "merge_method": "merge_and_unload",
            "torch_dtype": "float16"
        }
        
        import json
        with open(os.path.join(output_path, "merge_info.json"), "w", encoding="utf-8") as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        
        print("LoRA权重合并完成!")
        print(f"合并后的模型已保存到: {output_path}")
        
        # 8. 验证合并后的模型
        print("验证合并后的模型...")
        test_model = AutoModelForCausalLM.from_pretrained(
            output_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        print("模型验证成功!")
        
        return True
        
    except Exception as e:
        print(f"合并过程中出现错误: {str(e)}")
        return False

def compare_model_sizes(base_model_path: str, merged_model_path: str):
    """
    比较基础模型和合并后模型的大小
    
    Args:
        base_model_path: 基础模型路径
        merged_model_path: 合并后模型路径
    """
    try:
        # 计算基础模型参数量
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        base_params = sum(p.numel() for p in base_model.parameters())
        
        # 计算合并后模型参数量
        merged_model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        merged_params = sum(p.numel() for p in merged_model.parameters())
        
        print(f"\n模型参数对比:")
        print(f"基础模型参数量: {base_params:,}")
        print(f"合并后模型参数量: {merged_params:,}")
        print(f"参数量差异: {merged_params - base_params:,}")
        
        # 清理内存
        del base_model, merged_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"模型对比过程中出现错误: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="合并LoRA权重到基础模型")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                       help="基础模型路径")
    parser.add_argument("--lora_model", type=str, default="./qwen-medical-qa-lora", 
                       help="LoRA适配器路径")
    parser.add_argument("--output", type=str, default="./qwen-medical-qa-merged", 
                       help="合并后模型保存路径")
    parser.add_argument("--device", type=str, default="auto", 
                       help="设备类型 (auto, cpu, cuda)")
    parser.add_argument("--compare", action="store_true", 
                       help="是否比较模型大小")
    
    args = parser.parse_args()
    
    # 检查LoRA模型是否存在
    if not os.path.exists(args.lora_model):
        print(f"错误: LoRA模型路径不存在: {args.lora_model}")
        print("请先运行训练脚本生成LoRA适配器")
        return
    
    # 执行合并
    success = merge_lora_weights(
        base_model_path=args.base_model,
        lora_model_path=args.lora_model,
        output_path=args.output,
        device=args.device
    )
    
    if success and args.compare:
        compare_model_sizes(args.base_model, args.output)

if __name__ == "__main__":
    main()