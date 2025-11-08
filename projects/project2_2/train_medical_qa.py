import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 配置参数
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./qwen-medical-qa-lora"
MAX_LENGTH = 512
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 2e-5

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局变量声明
tokenizer = None

def load_medical_qa_data():
    """
    加载并预处理医疗QA数据集
    
    Returns:
        dataset: 预处理后的数据集
    """
    global tokenizer
    
    # 加载数据集
    dataset = load_dataset('json', data_files='data/medical_qa_data.jsonl', split='train')
    
    # 预处理函数
    def preprocess_function(examples):
        """
        预处理函数：将问答对转换为模型训练格式
        """
        questions = examples['question']
        answers = examples['answer']
        
        # 构建训练文本，使用指令格式
        texts = []
        for question, answer in zip(questions, answers):
            # 使用医疗助手的角色设定
            text = f"<|im_start|>system\n你是一个专业的医疗助手，请根据医学知识准确回答用户的健康相关问题。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            texts.append(text)
        
        # 分词处理
        tokenized = tokenizer(
            texts, 
            truncation=True, 
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors=None
        )
        
        # 为语言模型任务设置labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # 应用预处理
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )
    
    # 划分训练集和验证集
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"数据集加载完成:")
    print(f"  - 训练样本数: {len(dataset['train'])}")
    print(f"  - 验证样本数: {len(dataset['test'])}")
    
    return dataset

def main():
    """
    主训练函数
    """
    global tokenizer
    
    try:
        # 加载分词器
        print("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载数据集
        print("加载数据集...")
        dataset = load_medical_qa_data()
        
        # 加载基础模型 (4-bit量化以节省显存)
        print("加载基础模型...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        
        # 准备模型进行k-bit训练
        print("准备模型进行LoRA训练...")
        model = prepare_model_for_kbit_training(model)
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数信息
        model.print_trainable_parameters()
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=5,
            fp16=True,
            gradient_checkpointing=True,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            warmup_steps=10,
            gradient_accumulation_steps=4
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # 开始训练
        print("开始训练...")
        trainer.train()
        
        # 保存LoRA适配器
        print("保存模型...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"模型已保存至 {OUTPUT_DIR}")
        print("训练完成!")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()