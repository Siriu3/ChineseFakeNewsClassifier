### 2. 数据处理模块 (data_utils.py) - 第一部分
import jieba
import torch
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from config import Config


class ChineseNewsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer or Config.TOKENIZER
        self.max_len = max_len or Config.MAX_LEN
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 中文分词处理
        segmented_text = " ".join(jieba.cut(text))
        
        encoding = self.tokenizer.encode_plus(
            segmented_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            output['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return output

def load_data():
    """加载中文数据集示例"""
    train_df = pd.read_csv(Config.DATA_DIR/'train.csv')
    test_df = pd.read_csv(Config.DATA_DIR/'test.csv')
    
    # 标签转换示例（假设原始标签为'真实'/'虚假'）
    label_map = {'真实': 1, '虚假': 0}
    train_df['label'] = train_df['label'].map(label_map)
    
    return train_df, test_df