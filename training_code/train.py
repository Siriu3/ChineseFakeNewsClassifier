### 4. 训练循环模块 (train.py) - 第一部分
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from tqdm import tqdm
from model import ChineseFakeNewsModel, get_optimizer
from data_utils import ChineseNewsDataset, load_data
from config import Config
from sklearn.model_selection import KFold

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device)
        }
        labels = batch['labels'].to(device)
        
        with autocast():
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, f1

def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, f1

### 4. 训练循环模块 (train.py) - 第二部分（K折交叉验证）
def prepare_dataloaders(train_df, fold=0):
    """准备五折交叉验证的数据加载器"""
    kfold = KFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    train_indices, val_indices = list(kfold.split(train_df))[fold]
    
    train_dataset = ChineseNewsDataset(
        texts=train_df.iloc[train_indices]['content'].values,
        labels=train_df.iloc[train_indices]['label'].values
    )
    val_dataset = ChineseNewsDataset(
        texts=train_df.iloc[val_indices]['content'].values,
        labels=train_df.iloc[val_indices]['label'].values
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=RandomSampler(train_dataset),
        num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=SequentialSampler(val_dataset),
        num_workers=Config.NUM_WORKERS
    )
    
    return train_loader, val_loader

def run_training(fold=0, save_model=True):
    """单折训练流程"""
    train_df, _ = load_data()
    train_loader, val_loader = prepare_dataloaders(train_df, fold)
    
    model = ChineseFakeNewsModel().to(Config.DEVICE)
    total_steps = len(train_loader) * Config.EPOCHS
    optimizer, scheduler = get_optimizer(model, total_steps)
    scaler = GradScaler()
    
    best_f1 = 0
    for epoch in range(Config.EPOCHS):
        print(f"\nFold {fold + 1} | Epoch {epoch + 1}/{Config.EPOCHS}")
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, scaler, Config.DEVICE)
        val_loss, val_f1 = eval_epoch(model, val_loader, Config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1 and save_model:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"model_fold{fold}.bin")
            print(f"Saved new best model with F1: {best_f1:.4f}")
    
    return best_f1