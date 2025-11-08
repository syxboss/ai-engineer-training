"""
图像处理模块
提供图像加载、批量处理、搜索结果可视化等功能
"""

import os
import logging
from glob import glob
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from IPython.display import display
from config import config


class ImageProcessor:
    """图像处理器
    
    负责图像文件的加载、预处理和搜索结果的可视化展示
    """
    
    def __init__(self, image_directory: str = None):
        """初始化图像处理器
        
        Args:
            image_directory: 图像目录路径，默认使用配置中的路径
        """
        self.image_directory = image_directory or config.image.image_directory
        self.image_extensions = config.image.image_extensions
        self.thumbnail_size = config.image.thumbnail_size
        
    def get_image_paths(self, recursive: bool = True) -> List[str]:
        """获取目录中所有图像文件路径
        
        Args:
            recursive: 是否递归搜索子目录
            
        Returns:
            图像文件路径列表
        """
        image_paths = []
        
        try:
            for extension in self.image_extensions:
                pattern = os.path.join(self.image_directory, "**", extension) if recursive else \
                         os.path.join(self.image_directory, extension)
                paths = glob(pattern, recursive=recursive)
                image_paths.extend(paths)
            
            # 去重并排序
            image_paths = sorted(list(set(image_paths)))
            logging.info(f"找到 {len(image_paths)} 张图像文件")
            return image_paths
            
        except Exception as e:
            logging.error(f"获取图像路径失败: {e}")
            return []
    
    def validate_image(self, image_path: str) -> bool:
        """验证图像文件是否有效
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像是否有效
        """
        try:
            with Image.open(image_path) as img:
                img.verify()  # 验证图像完整性
            return True
        except Exception as e:
            logging.warning(f"无效图像文件 {image_path}: {e}")
            return False
    
    def filter_valid_images(self, image_paths: List[str]) -> List[str]:
        """过滤出有效的图像文件
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            有效的图像文件路径列表
        """
        valid_paths = []
        total_count = len(image_paths)
        
        for i, path in enumerate(image_paths):
            if self.validate_image(path):
                valid_paths.append(path)
            
            # 每处理100张图像输出一次进度
            if (i + 1) % 100 == 0:
                logging.info(f"已验证 {i + 1}/{total_count} 张图像")
        
        logging.info(f"有效图像: {len(valid_paths)}/{total_count}")
        return valid_paths
    
    def prepare_image_data(self, image_paths: List[str], features: List[List[float]]) -> List[Dict[str, Any]]:
        """准备用于插入数据库的图像数据
        
        Args:
            image_paths: 图像文件路径列表
            features: 对应的特征向量列表
            
        Returns:
            格式化的数据列表
        """
        if len(image_paths) != len(features):
            raise ValueError("图像路径数量与特征向量数量不匹配")
        
        data = []
        for path, feature in zip(image_paths, features):
            data.append({
                "vector": feature,
                "filepath": path,
                "filename": os.path.basename(path)
            })
        
        return data
    
    def create_result_grid(self, 
                          search_results: List[List[Dict]], 
                          query_text: str = None,
                          max_images: int = None) -> Optional[Image.Image]:
        """创建搜索结果的网格图像
        
        Args:
            search_results: 搜索结果列表
            query_text: 查询文本，用于显示
            max_images: 最大显示图像数量
            
        Returns:
            拼接后的网格图像，失败时返回None
        """
        try:
            # 提取图像路径
            image_paths = []
            for result_group in search_results:
                for hit in result_group:
                    filepath = hit.get("entity", {}).get("filepath")
                    if filepath:
                        image_paths.append(filepath)
            
            if not image_paths:
                logging.warning("没有找到有效的搜索结果")
                return None
            
            # 限制显示数量
            display_config = config.get_display_config()
            max_display = max_images or (display_config["columns"] * display_config["rows"])
            image_paths = image_paths[:max_display]
            
            # 加载和调整图像大小
            result_images = []
            for path in image_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = img.resize(self.thumbnail_size, Image.Resampling.LANCZOS)
                    result_images.append(img)
                except Exception as e:
                    logging.warning(f"加载图像失败 {path}: {e}")
                    # 创建占位图像
                    placeholder = Image.new('RGB', self.thumbnail_size, color='gray')
                    result_images.append(placeholder)
            
            if not result_images:
                return None
            
            # 创建网格图像
            grid_image = self._create_image_grid(result_images, display_config)
            
            # 显示查询信息
            if query_text:
                print(f"查询文本: {query_text}")
                print(f"找到 {len(result_images)} 张相似图像")
            
            return grid_image
            
        except Exception as e:
            logging.error(f"创建结果网格失败: {e}")
            return None
    
    def _create_image_grid(self, images: List[Image.Image], display_config: Dict[str, int]) -> Image.Image:
        """创建图像网格
        
        Args:
            images: 图像列表
            display_config: 显示配置
            
        Returns:
            网格图像
        """
        width = display_config["width"]
        height = display_config["height"]
        columns = display_config["columns"]
        thumb_width = display_config["thumbnail_width"]
        thumb_height = display_config["thumbnail_height"]
        
        # 创建空白画布
        grid_image = Image.new("RGB", (width, height), color='white')
        
        # 粘贴图像到网格
        for idx, img in enumerate(images):
            if idx >= columns * display_config["rows"]:
                break
                
            col = idx % columns
            row = idx // columns
            x = col * thumb_width
            y = row * thumb_height
            
            grid_image.paste(img, (x, y))
        
        return grid_image
    
    def display_results(self, 
                       search_results: List[List[Dict]], 
                       query_text: str = None,
                       save_path: str = None) -> bool:
        """显示搜索结果
        
        Args:
            search_results: 搜索结果
            query_text: 查询文本
            save_path: 保存路径（可选）
            
        Returns:
            显示是否成功
        """
        try:
            # 提取图像路径和相似度分数
            image_paths = []
            scores = []
            for result_group in search_results:
                for hit in result_group:
                    filepath = hit.get("entity", {}).get("filepath")
                    score = hit.get("distance", 0.0)
                    if filepath:
                        image_paths.append(filepath)
                        scores.append(score)
            
            if not image_paths:
                logging.warning("没有找到有效的搜索结果")
                return False
            
            # 限制显示数量
            display_config = config.get_display_config()
            max_display = display_config["columns"] * display_config["rows"]
            image_paths = image_paths[:max_display]
            scores = scores[:max_display]
            
            # 使用matplotlib显示结果
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                # 设置中文字体支持
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                
                # 计算网格布局
                cols = display_config["columns"]
                rows = min(display_config["rows"], (len(image_paths) + cols - 1) // cols)
                
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                if rows == 1:
                    axes = [axes] if cols == 1 else axes
                elif cols == 1:
                    axes = [[ax] for ax in axes]
                else:
                    axes = axes
                
                # 设置标题
                if query_text:
                    fig.suptitle(f'Search Results: "{query_text}"', fontsize=16, fontweight='bold')
                
                # 显示每张图片
                for i in range(rows * cols):
                    row = i // cols
                    col = i % cols
                    ax = axes[row][col] if rows > 1 else axes[col]
                    
                    if i < len(image_paths):
                        try:
                            # 加载并显示图片
                            img = Image.open(image_paths[i]).convert('RGB')
                            ax.imshow(img)
                            
                            # 添加文件名和相似度分数
                            filename = os.path.basename(image_paths[i])
                            title = f"{filename}\nSimilarity: {scores[i]:.3f}"
                            ax.set_title(title, fontsize=10, pad=5)
                            
                        except Exception as e:
                            logging.warning(f"加载图像失败 {image_paths[i]}: {e}")
                            ax.text(0.5, 0.5, 'Image Load Failed', ha='center', va='center', 
                                   transform=ax.transAxes, fontsize=12)
                    else:
                        # 空白区域
                        ax.axis('off')
                        continue
                    
                    ax.axis('off')
                
                plt.tight_layout()
                
                # 保存图像（如果指定了路径）
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    logging.info(f"结果已保存到: {save_path}")
                
                # 显示图像
                plt.show()
                
                print(f"找到 {len(image_paths)} 张相似图像")
                return True
                
            except ImportError:
                # 如果没有matplotlib，回退到原来的方法
                logging.warning("matplotlib未安装，使用PIL显示")
                grid_image = self.create_result_grid(search_results, query_text)
                
                if grid_image is None:
                    logging.error("无法创建结果网格")
                    return False
                
                # 保存图像（如果指定了路径）
                if save_path:
                    grid_image.save(save_path)
                    logging.info(f"结果已保存到: {save_path}")
                
                # 在Jupyter环境中显示
                try:
                    display(grid_image)
                except NameError:
                    # 如果不在Jupyter环境中，显示图像
                    grid_image.show()
                
                return True
            
        except Exception as e:
            logging.error(f"显示结果失败: {e}")
            return False
    
    def get_image_info(self, image_path: str) -> Optional[Dict[str, Any]]:
        """获取图像信息
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像信息字典
        """
        try:
            with Image.open(image_path) as img:
                return {
                    "filepath": image_path,
                    "filename": os.path.basename(image_path),
                    "size": img.size,
                    "mode": img.mode,
                    "format": img.format,
                    "file_size": os.path.getsize(image_path)
                }
        except Exception as e:
            logging.error(f"获取图像信息失败 {image_path}: {e}")
            return None
    
    def batch_resize_images(self, 
                           image_paths: List[str], 
                           output_dir: str, 
                           size: Tuple[int, int] = None) -> List[str]:
        """批量调整图像大小
        
        Args:
            image_paths: 输入图像路径列表
            output_dir: 输出目录
            size: 目标尺寸，默认使用缩略图尺寸
            
        Returns:
            输出图像路径列表
        """
        size = size or self.thumbnail_size
        output_paths = []
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        for path in image_paths:
            try:
                filename = os.path.basename(path)
                output_path = os.path.join(output_dir, filename)
                
                with Image.open(path) as img:
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)
                    resized_img.save(output_path)
                    output_paths.append(output_path)
                    
            except Exception as e:
                logging.warning(f"调整图像大小失败 {path}: {e}")
        
        logging.info(f"成功调整 {len(output_paths)} 张图像大小")
        return output_paths