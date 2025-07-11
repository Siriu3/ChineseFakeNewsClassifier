import torch
from pathlib import Path

class Config:
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 1024
    
    # 数据参数
    DATA_DIR = Path("./data/chinese_fake_news")
    MAX_LEN = 256
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # 模型参数
    MODEL_NAME = "bert-base-chinese"  # 中文BERT模型
    TOKENIZER = None  # 将在初始化时设置
    NUM_CLASSES = 2
    
    # 训练参数
    EPOCHS = 10
    LR = 2e-5
    WARMUP_STEPS = 1000
    WEIGHT_DECAY = 0.01
    
    @classmethod
    def setup_tokenizer(cls):
        from transformers import BertTokenizerFast
        cls.TOKENIZER = BertTokenizerFast.from_pretrained(cls.MODEL_NAME)