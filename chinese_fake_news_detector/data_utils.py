import re
import jieba
from typing import List, Dict, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np
from .config import Config


def initialize_jieba():
    """初始化jieba分词"""
    # 添加停用词
    jieba.initialize()
    jieba.setLogLevel('ERROR')  # 关闭调试日志


def clean_text(text: str) -> str:
    """
    清理文本
    
    参数:
        text: 原始文本
        
    返回:
        清理后的文本
    """
    # 移除特殊字符和多余空格
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    return text.strip()


def calculate_word_scores(
    text: str,
    tokenizer,
    offset_mapping: list,
    attention_weights: List[np.ndarray],
    layers: List[int] = None
) -> Dict[str, float]:
    """
    计算词语的可疑度分数
    
    参数:
        text: 原始文本
        tokenizer: 使用的tokenizer
        offset_mapping: token到原始文本的位置映射
        attention_weights: 各层的注意力权重
        layers: 使用的层索引列表
        
    返回:
        词语到分数的字典
    """
    if layers is None:
        layers = Config.EXTRACT_LAYERS
        
    word_scores = defaultdict(float)
    token_count = len(offset_mapping)
    
    for layer_idx in layers:
        if layer_idx >= len(attention_weights):
            continue
            
        # 获取当前层的注意力权重 (平均所有注意力头)
        layer_attn = attention_weights[layer_idx].mean(axis=1)[0]  # [seq_len, seq_len]
        
        # 获取[CLS] token对其他token的注意力
        cls_attention = layer_attn[0]
        
        for token_idx, score in enumerate(cls_attention):
            if token_idx == 0:  # 跳过[CLS] token
                continue
                
            if token_idx >= token_count:
                break
                
            # 获取原始文本中的词语
            start, end = offset_mapping[token_idx]
            if start == end:  # 特殊token
                continue
                
            word = text[start:end]
            if word.strip():
                word_scores[word] += float(score)
    
    return word_scores


def merge_word_scores(
    all_scores: List[Dict[str, float]],
    min_models: int = 2
) -> Dict[str, float]:
    """
    合并多个模型的词语分数
    
    参数:
        all_scores: 各模型的词语分数列表
        min_models: 词语至少需要出现在几个模型中
        
    返回:
        合并后的词语分数字典
    """
    merged_scores = defaultdict(float)
    word_counts = defaultdict(int)
    
    for scores in all_scores:
        for word, score in scores.items():
            merged_scores[word] += score
            word_counts[word] += 1
    
    # 只保留被足够多模型识别的词语
    filtered_scores = {
        word: (score / word_counts[word])
        for word, score in merged_scores.items()
        if word_counts[word] >= min_models
    }
    
    return filtered_scores


def get_top_suspicious_words(
    word_scores: Dict[str, float],
    text_length: int
) -> Dict[str, float]:
    """
    根据文本长度获取top可疑词
    
    参数:
        word_scores: 词语分数字典
        text_length: 文本长度
        
    返回:
        前N个可疑词及其分数
    """
    # 确定要提取的词数
    top_n = 0
    for start, end, count in Config.WORD_COUNT_RULES:
        if start <= text_length <= end:
            top_n = count if isinstance(count, int) else int(text_length * count)
            break
    
    if top_n <= 0:
        top_n = 5
        
    # 排序并取前N个
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_words[:top_n])
    
    # 归一化分数
    if top_words:
        max_score = max(top_words.values()) or 1.0
        top_words = {word: score/max_score for word, score in top_words.items()}
    
    return top_words