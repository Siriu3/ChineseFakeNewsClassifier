### 5. 评估与集成模块 (eval.py)
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from model import ChineseFakeNewsModel
from data_utils import ChineseNewsDataset
from config import Config
from sklearn.metrics import f1_score


def predict(model, dataloader, device):
    """生成模型预测结果"""
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            outputs = model(**inputs)
            all_logits.append(outputs.cpu().numpy())
    
    return np.concatenate(all_logits, axis=0)

def ensemble_predictions(model_paths, test_dataset):
    """多模型集成预测"""
    all_preds = []
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=SequentialSampler(test_dataset),
        num_workers=Config.NUM_WORKERS
    )
    
    for path in model_paths:
        model = ChineseFakeNewsModel()
        model.load_state_dict(torch.load(path))
        model.to(Config.DEVICE)
        logits = predict(model, test_loader, Config.DEVICE)
        all_preds.append(logits)
    
    # 平均集成
    avg_logits = np.mean(all_preds, axis=0)
    return np.argmax(avg_logits, axis=1)

def evaluate_ensemble(model_paths, test_df):
    """评估集成模型"""
    test_dataset = ChineseNewsDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values
    )
    preds = ensemble_predictions(model_paths, test_dataset)
    return f1_score(test_df['label'], preds, average='weighted')