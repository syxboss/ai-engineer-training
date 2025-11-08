"""
CLIP编码器模块
封装CLIP模型的加载、图像编码和文本编码功能
"""

import logging
from typing import List, Union
import torch
import clip
from PIL import Image
from config import config


class CLIPEncoder:
    """CLIP特征编码器
    
    负责加载CLIP模型并提供图像和文本的特征编码功能
    """
    
    def __init__(self, model_name: str = None):
        """初始化CLIP编码器
        
        Args:
            model_name: CLIP模型名称，默认使用配置中的模型
        """
        self.model_name = model_name or config.clip.model_name
        self.model = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self) -> None:
        """加载CLIP模型"""
        try:
            logging.info(f"正在加载CLIP模型: {self.model_name}")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()  # 设置为评估模式
            logging.info(f"CLIP模型加载成功，使用设备: {self.device}")
        except Exception as e:
            logging.error(f"CLIP模型加载失败: {e}")
            raise
    
    def encode_image(self, image_path: str) -> List[float]:
        """编码单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            归一化的图像特征向量列表
            
        Raises:
            FileNotFoundError: 图像文件不存在
            Exception: 图像处理或编码失败
        """
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # L2归一化，确保余弦相似性计算准确
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.squeeze().cpu().tolist()
            
        except FileNotFoundError:
            logging.error(f"图像文件不存在: {image_path}")
            raise
        except Exception as e:
            logging.error(f"图像编码失败 {image_path}: {e}")
            raise
    
    def encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量编码图像
        
        Args:
            image_paths: 图像文件路径列表
            batch_size: 批处理大小
            
        Returns:
            图像特征向量列表
        """
        features = []
        total_images = len(image_paths)
        
        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # 预处理批次图像
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.preprocess(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    logging.warning(f"跳过无效图像 {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # 批量编码
            try:
                batch_tensor = torch.stack(batch_images).to(self.device)
                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_tensor)
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                
                features.extend(batch_features.cpu().tolist())
                logging.info(f"已处理 {min(i + batch_size, total_images)}/{total_images} 张图像")
                
            except Exception as e:
                logging.error(f"批量编码失败: {e}")
                continue
        
        return features
    
    def encode_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """编码文本
        
        Args:
            text: 单个文本字符串或文本列表
            
        Returns:
            归一化的文本特征向量或特征向量列表
        """
        try:
            # 处理单个文本
            if isinstance(text, str):
                text_tokens = clip.tokenize([text]).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.squeeze().cpu().tolist()
            
            # 处理文本列表
            else:
                text_tokens = clip.tokenize(text).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().tolist()
                
        except Exception as e:
            logging.error(f"文本编码失败: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """获取模型信息
        
        Returns:
            包含模型名称、设备、特征维度等信息的字典
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "feature_dimension": config.clip.feature_dimension,
            "is_loaded": self.model is not None
        }