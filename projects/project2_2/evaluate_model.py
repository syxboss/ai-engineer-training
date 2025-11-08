import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import argparse
from tqdm import tqdm

class MedicalQAEvaluator:
    def __init__(self, model_path: str, tokenizer_path: str = None, device: str = "auto"):
        """
        医疗QA模型评估器
        
        Args:
            model_path: 模型路径
            tokenizer_path: 分词器路径，默认与模型路径相同
            device: 设备类型
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = device
        
        # 加载模型和分词器
        self.load_model()
        
        # 初始化评估指标
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def load_model(self):
        """加载模型和分词器"""
        print(f"加载模型: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("模型加载完成")
    
    def generate_answer(self, question: str, max_length: int = 512) -> str:
        """
        生成问题的答案
        
        Args:
            question: 输入问题
            max_length: 最大生成长度
            
        Returns:
            生成的答案
        """
        # 构建输入提示
        prompt = f"<|im_start|>system\n你是一个专业的医疗助手，请根据医学知识准确回答用户的健康相关问题。<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成答案
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手回答部分
        if "<|im_start|>assistant\n" in generated_text:
            answer = generated_text.split("<|im_start|>assistant\n")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        return answer
    
    def evaluate_rouge_scores(self, predictions: list, references: list) -> dict:
        """
        计算ROUGE分数
        
        Args:
            predictions: 预测答案列表
            references: 参考答案列表
            
        Returns:
            ROUGE分数字典
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # 计算平均分数
        avg_scores = {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
        
        return avg_scores
    
    def evaluate_bert_score(self, predictions: list, references: list) -> dict:
        """
        计算BERTScore
        
        Args:
            predictions: 预测答案列表
            references: 参考答案列表
            
        Returns:
            BERTScore结果字典
        """
        try:
            P, R, F1 = bert_score(predictions, references, lang="zh", verbose=False)
            
            return {
                'bert_precision': P.mean().item(),
                'bert_recall': R.mean().item(),
                'bert_f1': F1.mean().item()
            }
        except Exception as e:
            print(f"BERTScore计算失败: {e}")
            return {
                'bert_precision': 0.0,
                'bert_recall': 0.0,
                'bert_f1': 0.0
            }
    
    def evaluate_dataset(self, data_path: str, sample_size: int = None) -> dict:
        """
        评估整个数据集
        
        Args:
            data_path: 数据集路径
            sample_size: 采样大小，None表示使用全部数据
            
        Returns:
            评估结果字典
        """
        print(f"加载评估数据集: {data_path}")
        
        # 加载数据集
        dataset = load_dataset('json', data_files=data_path, split='train')
        
        if sample_size and sample_size < len(dataset):
            dataset = dataset.select(range(sample_size))
        
        print(f"评估样本数: {len(dataset)}")
        
        # 生成预测答案
        predictions = []
        references = []
        
        print("生成预测答案...")
        for example in tqdm(dataset):
            question = example['question']
            reference = example['answer']
            
            prediction = self.generate_answer(question)
            
            predictions.append(prediction)
            references.append(reference)
        
        # 计算评估指标
        print("计算评估指标...")
        
        # ROUGE分数
        rouge_scores = self.evaluate_rouge_scores(predictions, references)
        
        # BERTScore
        bert_scores = self.evaluate_bert_score(predictions, references)
        
        # 答案长度统计
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        results = {
            'dataset_info': {
                'data_path': data_path,
                'sample_size': len(dataset),
                'avg_pred_length': np.mean(pred_lengths),
                'avg_ref_length': np.mean(ref_lengths)
            },
            'rouge_scores': rouge_scores,
            'bert_scores': bert_scores,
            'examples': []
        }
        
        # 保存一些示例
        for i in range(min(5, len(predictions))):
            results['examples'].append({
                'question': dataset[i]['question'],
                'reference': references[i],
                'prediction': predictions[i]
            })
        
        return results
    
    def save_results(self, results: dict, output_path: str):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            output_path: 输出路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {output_path}")
    
    def print_results(self, results: dict):
        """
        打印评估结果
        
        Args:
            results: 评估结果
        """
        print("\n" + "="*50)
        print("评估结果")
        print("="*50)
        
        # 数据集信息
        dataset_info = results['dataset_info']
        print(f"数据集: {dataset_info['data_path']}")
        print(f"样本数: {dataset_info['sample_size']}")
        print(f"平均预测长度: {dataset_info['avg_pred_length']:.2f} 词")
        print(f"平均参考长度: {dataset_info['avg_ref_length']:.2f} 词")
        
        # ROUGE分数
        rouge_scores = results['rouge_scores']
        print(f"\nROUGE分数:")
        print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        # BERTScore
        bert_scores = results['bert_scores']
        print(f"\nBERTScore:")
        print(f"  Precision: {bert_scores['bert_precision']:.4f}")
        print(f"  Recall: {bert_scores['bert_recall']:.4f}")
        print(f"  F1: {bert_scores['bert_f1']:.4f}")
        
        # 示例
        print(f"\n示例预测:")
        for i, example in enumerate(results['examples'][:3]):
            print(f"\n示例 {i+1}:")
            print(f"问题: {example['question']}")
            print(f"参考答案: {example['reference']}")
            print(f"预测答案: {example['prediction']}")

def main():
    parser = argparse.ArgumentParser(description="评估医疗QA模型")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="模型路径")
    parser.add_argument("--data_path", type=str, default="data/medical_qa_data.jsonl", 
                       help="评估数据路径")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json", 
                       help="结果保存路径")
    parser.add_argument("--sample_size", type=int, default=None, 
                       help="评估样本数量")
    parser.add_argument("--device", type=str, default="auto", 
                       help="设备类型")
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return
    
    # 检查数据路径
    if not os.path.exists(args.data_path):
        print(f"错误: 数据路径不存在: {args.data_path}")
        return
    
    # 创建评估器
    evaluator = MedicalQAEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # 执行评估
    results = evaluator.evaluate_dataset(
        data_path=args.data_path,
        sample_size=args.sample_size
    )
    
    # 打印结果
    evaluator.print_results(results)
    
    # 保存结果
    evaluator.save_results(results, args.output_path)

if __name__ == "__main__":
    main()