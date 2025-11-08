import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 配置参数
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "./qwen3-intent-lora"
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局变量声明
tokenizer = None

def load_intent_data():
    """
    加载并预处理意图识别数据集
    
    Returns:
        tuple: (dataset, num_labels, label2id, id2label)
    """
    global tokenizer
    
    # 加载数据集
    dataset = load_dataset('json', data_files='data/intent_data.jsonl', split='train')
    
    # 创建标签映射
    intents = sorted(list(set(dataset['intent'])))  # 排序确保一致性
    label2id = {intent: idx for idx, intent in enumerate(intents)}
    id2label = {idx: intent for intent, idx in label2id.items()}
    
    # 保存标签映射
    with open(os.path.join(OUTPUT_DIR, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    
    # 预处理函数
    def preprocess_function(examples):
        """
        预处理函数：将文本转换为模型输入格式
        """
        texts = examples['text']
        labels = [label2id[intent] for intent in examples['intent']]
        
        # 分词处理
        tokenized = tokenizer(
            texts, 
            truncation=True, 
            max_length=MAX_LENGTH,
            padding=False  # 在DataCollator中进行padding
        )
        
        # 添加标签
        tokenized["labels"] = labels
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
    print(f"  - 意图类别数: {len(intents)}")
    print(f"  - 训练样本数: {len(dataset['train'])}")
    print(f"  - 验证样本数: {len(dataset['test'])}")
    print(f"  - 意图类别: {intents}")
    
    return dataset, len(intents), label2id, id2label

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
        dataset, num_labels, label2id, id2label = load_intent_data()
        
        # 加载基础模型 (4-bit量化以节省显存)
        print("加载基础模型...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
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
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
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
            logging_steps=10,
            fp16=True,
            gradient_checkpointing=True,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2
        )
        
        # 数据整理器
        data_collator = DataCollatorWithPadding(
            tokenizer,
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