import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .config import Config


class ChineseFakeNewsModel(nn.Module):
    def __init__(self, model_config: dict, num_classes: int = None):
        """
        中文假新闻检测模型
        
        参数:
            model_config: 模型配置字典
            num_classes: 分类类别数，默认为Config.NUM_CLASSES
        """
        super().__init__()
        self.model_name = model_config["name"]
        
        # 从模型配置中获取 Hugging Face 模型的名称，例如 "bert-base-chinese"
        # 这个名称用于从 Hugging Face Hub 下载
        hf_model_name = model_config["name"]
        
        # 获取模型在本地的完整存储路径，例如 "D:\...\original\bert-base-chinese"
        # 这个路径用于从本地加载，以及下载后保存
        local_storage_path = Config.get_model_path(model_config)

        try:
            # 1. 优先尝试从本地路径加载配置和模型
            # 使用 local_storage_path 和 local_files_only=True 来加载本地文件
            config = AutoConfig.from_pretrained(
                local_storage_path, # 使用本地路径加载
                attn_implementation="eager",
                local_files_only=True
            )
            self.encoder = AutoModel.from_pretrained(
                local_storage_path, # 使用本地路径加载
                config=config,
                local_files_only=True
            )
            print(f"Model '{hf_model_name}' loaded from local path: {local_storage_path}")
        except (OSError, IOError):
            # 2. 如果本地加载失败（文件不存在），则从 Hugging Face Hub 下载并缓存到本地
            print(f"Local model for '{hf_model_name}' not found. Downloading from Hugging Face Hub...")
            # cache_dir 指向 Config.ORIGINAL_DIR，用于指定下载缓存的位置
            config = AutoConfig.from_pretrained(
                hf_model_name, # 使用 Hugging Face 模型名称下载
                attn_implementation="eager",
                cache_dir=Config.ORIGINAL_DIR 
            )
            self.encoder = AutoModel.from_pretrained(
                hf_model_name, # 使用 Hugging Face 模型名称下载
                config=config,
                cache_dir=Config.ORIGINAL_DIR 
            )
            # 下载成功后，将模型保存到本地存储路径
            self.encoder.save_pretrained(local_storage_path)
            print(f"Model '{hf_model_name}' downloaded and saved to {local_storage_path}")
        
        # 分类器
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_classes or Config.NUM_CLASSES
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        参数:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            
        返回:
            分类logits
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True  # 确保返回注意力权重
        )
        
        # 获取pooled输出
        if hasattr(outputs, 'pooler_output'):
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)