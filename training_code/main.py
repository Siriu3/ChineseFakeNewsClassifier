### 6. 主程序入口 (main.py)
import argparse
from train import run_training
from eval import evaluate_ensemble
from config import Config
from data_utils import load_data
import os

def main():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--fold", type=int, help="Fold number (0-4)")
    args = parser.parse_args()
    
    Config.setup_tokenizer()
    
    if args.mode == "train":
        if args.fold is not None:
            run_training(fold=args.fold)
        else:
            # 五折交叉训练
            for fold in range(5):
                run_training(fold=fold)
    else:
        model_paths = [f"model_fold{i}.bin" for i in range(5)]
        _, test_df = load_data()
        f1 = evaluate_ensemble(model_paths, test_df)
        print(f"\nEnsemble Model F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
"""
# 训练所有五折模型
python main.py --mode train

# 评估集成模型
python main.py --mode eval
"""
