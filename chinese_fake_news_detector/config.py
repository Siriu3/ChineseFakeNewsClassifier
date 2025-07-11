import torch
from pathlib import Path
import os

class Config:
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 1024
    
    # 基础路径配置
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / "models"
    ORIGINAL_DIR = BASE_DIR / "original"
    
    # 模型配置 (五折使用不同模型)
    MODEL_CONFIGS = [
        {
            "name": "bert-base-chinese", 
            "type": "bert",
            "dir_name": "bert-base-chinese"
        },
        {
            "name": "nghuyong/ernie-3.0-base-zh", 
            "type": "ernie",
            "dir_name": "ernie-3.0-base-zh"
        },
        {
            "name": "hfl/chinese-roberta-wwm-ext", 
            "type": "roberta",
            "dir_name": "chinese-roberta-wwm-ext"
        },
        {
            "name": "hfl/chinese-xlnet-base", 
            "type": "xlnet",
            "dir_name": "chinese-xlnet-base"
        },
        {
            "name": "hfl/chinese-electra-base-discriminator", 
            "type": "electra",
            "dir_name": "chinese-electra-base-discriminator"
        }
    ]
    
    # 分类数和batch大小
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    
    # 可疑词提取配置
    EXTRACT_LAYERS = [-1, -2, -3]  # 最后三层
    CANDIDATE_MULTIPLIER = 2       # 候选词倍数
    WORD_COUNT_RULES = [            # 字符数区间: 词数
        (0, 40, 3),
        (41, 100, 5),
        (101, float('inf'), 0.06)   # 百分比
    ]
    
    # 若输出可疑词汇，词语最少需要被几个模型识别
    SUSPICIOUS_WORDS_MIN_MODELS = 2
    
    @classmethod
    def get_model_path(cls, model_config):
        """获取模型本地路径"""
        return str(cls.ORIGINAL_DIR / model_config["dir_name"])
    
    @classmethod
    def verify_paths(cls):
        """验证所需路径和文件是否存在"""
        # 检查模型权重文件
        required_weights = [f"model_fold{i}.bin" for i in range(5)]
        missing_weights = [f for f in required_weights if not (cls.MODEL_DIR / f).exists()]
        
        if missing_weights:
            raise FileNotFoundError(
                f"Missing model weights: {', '.join(missing_weights)}. "
                f"Expected files in {cls.MODEL_DIR}"
            )
        