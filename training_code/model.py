### 模型架构模块 (model.py) - PyTorch实现
import torch
import torch.nn as nn
from transformers import BertModel
from config import Config

class ChineseFakeNewsModel(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 
                                  num_classes or Config.NUM_CLASSES)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

def get_optimizer(model, total_steps):
    """带warmup的AdamW优化器"""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    optimizer = AdamW(model.parameters(), 
                     lr=Config.LR, 
                     weight_decay=Config.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    return optimizer, scheduler