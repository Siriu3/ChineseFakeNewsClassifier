import torch
import numpy as np
import jieba
from typing import Dict, Tuple, List, Any
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .model import ChineseFakeNewsModel
from .config import Config
from .data_utils import get_top_suspicious_words, calculate_word_scores, clean_text


class ChineseFakeNewsPredictor:
    def __init__(self):
        self.models: List[ChineseFakeNewsModel] = []
        self.tokenizers: List[AutoTokenizer] = []
        self.model_configs: List[Dict[str, Any]] = []
        self.initialized = False
    
    def initialize(self):
        """加载所有模型和tokenizer。如果本地不存在，则从Hugging Face下载。"""
        if self.initialized:
            return
        
        # 验证并创建所需目录
        Config.verify_paths()
        
        # 按顺序加载每个模型和对应的tokenizer
        for fold, model_config in enumerate(Config.MODEL_CONFIGS):
            model_weight_path = Config.MODEL_DIR / f"model_fold{fold}.bin"
            
            # 从模型配置中获取 Hugging Face 模型的名称，例如 "bert-base-chinese"
            # 这个名称用于从 Hugging Face Hub 下载
            hf_model_name = model_config["name"]
            
            # 获取模型在本地的完整存储路径，例如 "D:\...\original\bert-base-chinese"
            # 这个路径用于从本地加载，以及下载后保存
            local_storage_path = Config.get_model_path(model_config)

            try:
                # 1. 优先尝试从本地路径加载 Tokenizer
                # 使用 local_storage_path 和 local_files_only=True 来加载本地文件
                tokenizer = AutoTokenizer.from_pretrained(
                    local_storage_path, 
                    local_files_only=True
                )
                print(f"Tokenizer for '{hf_model_name}' loaded from local path: {local_storage_path}")
            except (OSError, IOError):
                # 2. 如果本地加载失败（文件不存在），则从 Hugging Face Hub 下载并缓存到本地
                print(f"Local tokenizer for '{hf_model_name}' not found. Downloading from Hugging Face Hub...")
                # cache_dir 指向 Config.ORIGINAL_DIR，用于指定下载缓存的位置
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name, # 使用 Hugging Face 模型名称下载
                    cache_dir=Config.ORIGINAL_DIR 
                )
                # 下载成功后，将 tokenizer 保存到本地存储路径
                tokenizer.save_pretrained(local_storage_path)
                print(f"Tokenizer for '{hf_model_name}' downloaded and saved to {local_storage_path}")

            # 初始化模型 (ChineseFakeNewsModel 内部会处理其自身的 AutoModel 加载和下载逻辑)
            model = ChineseFakeNewsModel(model_config) 
            
            # 检查微调后的权重文件是否存在（这部分逻辑与下载无关，保持不变）
            if not model_weight_path.exists():
                raise FileNotFoundError(
                    f"Fine-tuned model weight not found: {model_weight_path}. "
                    f"This file cannot be downloaded and must be provided."
                )

            model.load_state_dict(
                torch.load(model_weight_path, map_location=Config.DEVICE)
            )
            model.to(Config.DEVICE)
            model.eval()
            
            self.models.append(model)
            self.tokenizers.append(tokenizer)
            self.model_configs.append(model_config)

        self.initialized = True
    
    def _process_text(self, text: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
        """处理输入文本"""
        # 清理文本
        text = clean_text(text)
        
        # 使用tokenizer处理
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        # 将输入移动到设备上
        input_ids = inputs["input_ids"].to(Config.DEVICE)
        attention_mask = inputs["attention_mask"].to(Config.DEVICE)
        offset_mapping = inputs["offset_mapping"]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping
        }
    
    def predict(self, text: str, analyze_words: bool = False) -> Tuple[int, Dict[str, float]]:
        """
        预测单条新闻
        
        参数:
            text: 待分析文本
            analyze_words: 是否分析可疑词汇
            
        返回:
            (预测标签, 可疑词字典) 如果analyze_words=False，字典为空
        """
        if not self.initialized:
            self.initialize()
        
        # 1. 分类预测
        all_logits = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = self._process_text(text, tokenizer)
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                all_logits.append(outputs.cpu().numpy())
        
        avg_logits = np.mean(all_logits, axis=0)
        pred_label = int(np.argmax(avg_logits, axis=1)[0])
        
        # 2. 按需分析可疑词
        suspicious_words = {}
        if analyze_words:
            all_word_scores = []
            
            for model, tokenizer in zip(self.models, self.tokenizers):
                inputs = self._process_text(text, tokenizer)
                offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
                
                with torch.no_grad():
                    outputs = model.encoder(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        output_attentions=True
                    )
                    
                    # 计算词语分数
                    word_scores = calculate_word_scores(
                        text,
                        tokenizer,
                        offset_mapping,
                        [layer.cpu().numpy() for layer in outputs.attentions],
                        Config.EXTRACT_LAYERS
                    )
                    
                    all_word_scores.append(word_scores)
            
            # 合并多个模型的词语分数
            from .data_utils import merge_word_scores
            merged_scores = merge_word_scores(
                all_word_scores,
                Config.SUSPICIOUS_WORDS_MIN_MODELS
            )
            
            # 获取top可疑词
            suspicious_words = get_top_suspicious_words(merged_scores, len(text))
        
        return pred_label, suspicious_words