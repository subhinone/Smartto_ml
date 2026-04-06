"""
모델 최종 평가 스크립트
-----------------------
학습된 모델을 DAiSEE 테스트셋으로 평가합니다.

실행:
    python src/evaluate.py
    python src/evaluate.py --finetuned   # fine-tuned 모델 평가
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix, classification_report
)

# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "features")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ──────────────────────────────────────────────
# 설정 (step3_train.py와 동일)
# ──────────────────────────────────────────────
SEQ_LEN     = 90
N_FEATURES  = 14
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
N_CLASSES   = 2

# ──────────────────────────────────────────────
# 디바이스
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ──────────────────────────────────────────────
# 스케일러 (pickle 로드용)
# ──────────────────────────────────────────────
class RobustFeatureScaler:
    def __init__(self, binary_col_idx=7):
        self.binary_col_idx = binary_col_idx
        self.mean_ = None
        self.std_  = None

    def fit(self, X_flat):
        self.mean_ = np.mean(X_flat, axis=0).astype(np.float32)
        self.std_  = np.std(X_flat,  axis=0).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0
        self.mean_[self.binary_col_idx] = 0.0
        self.std_[self.binary_col_idx]  = 1.0
        return self

    def transform(self, X_flat):
        return ((X_flat - self.mean_) / self.std_).astype(np.float32)


# ──────────────────────────────────────────────
# 모델 정의 (step3_train.py와 동일 구조)
# ──────────────────────────────────────────────
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class ConcentrationGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            N_FEATURES, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.0
        )
        self.pool      = AttentionPooling(HIDDEN_SIZE)
        self.dropout   = nn.Dropout(0.4)
        self.fc        = nn.Linear(HIDDEN_SIZE, N_CLASSES)

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = self.pool(out)
        return self.fc(self.dropout(pooled))


# ──────────────────────────────────────────────
# 테스트 데이터 로드
# ──────────────────────────────────────────────
def load_test_data():
    path = os.path.join(DATA_DIR, "test_ready.npz")
    data = np.load(path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    print(f"테스트 데이터: {X.shape[0]}개")
    print(f"  집중(1)  : {(y==1).sum()}개 ({(y==1).mean()*100:.1f}%)")
    print(f"  비집중(0): {(y==0).sum()}개 ({(y==0).mean()*100:.1f}%)")
    return X, y


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────
def evaluate(model, X, y, threshold):
    model.eval()
    all_probs = []

    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size]).to(DEVICE)
            logits = model(xb)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    probs = np.array(all_probs)
    preds = (probs >= threshold).astype(int)

    acc       = accuracy_score(y, preds)
    f1_macro  = f1_score(y, preds, average="macro")
    f1_0      = f1_score(y, preds, pos_label=0)
    f1_1      = f1_score(y, preds, pos_label=1)
    rec_0     = recall_score(y, preds, pos_label=0)
    rec_1     = recall_score(y, preds, pos_label=1)
    prec_0    = precision_score(y, preds, pos_label=0, zero_division=0)
    prec_1    = precision_score(y, preds, pos_label=1, zero_division=0)
    cm        = confusion_matrix(y, preds)

    return {
        "accuracy"  : acc,
        "f1_macro"  : f1_macro,
        "f1_0"      : f1_0,
        "f1_1"      : f1_1,
        "recall_0"  : rec_0,
        "recall_1"  : rec_1,
        "precision_0": prec_0,
        "precision_1": prec_1,
        "confusion_matrix": cm,
        "probs"     : probs,
    }


def print_results(results, model_name, threshold):
    cm = results["confusion_matrix"]
    print(f"\n{'='*50}")
    print(f"  {model_name}  (threshold={threshold:.2f})")
    print(f"{'='*50}")
    print(f"  Accuracy    : {results['accuracy']*100:.1f}%")
    print(f"  F1 Macro    : {results['f1_macro']:.4f}")
    print()
    print(f"  {'':12s}  Precision  Recall    F1")
    print(f"  {'비집중(0)':12s}  {results['precision_0']:.3f}      {results['recall_0']:.3f}     {results['f1_0']:.3f}")
    print(f"  {'집중(1)':12s}  {results['precision_1']:.3f}      {results['recall_1']:.3f}     {results['f1_1']:.3f}")
    print()
    print(f"  혼동 행렬:")
    print(f"                예측 비집중  예측 집중")
    print(f"  실제 비집중      {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"  실제 집중        {cm[1,0]:5d}       {cm[1,1]:5d}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", action="store_true",
                        help="fine-tuned 모델로 평가")
    args = parser.parse_args()

    print(f"디바이스: {DEVICE}\n")

    # 테스트 데이터 로드
    X_test, y_test = load_test_data()

    # 스케일러 로드
    scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # 스케일링 (2D로 reshape → 스케일 → 복원)
    N, T, F = X_test.shape
    X_flat  = X_test.reshape(-1, F)
    X_scaled = scaler.transform(X_flat).reshape(N, T, F)

    # ── 원본 모델 평가 ──
    model_path = os.path.join(MODEL_DIR, "best_model.pt")
    thresh_path = os.path.join(MODEL_DIR, "threshold.json")

    model = ConcentrationGRU().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

    with open(thresh_path) as f:
        threshold = json.load(f)["focused_threshold"]

    results = evaluate(model, X_scaled, y_test, threshold)
    print_results(results, "원본 모델 (DAiSEE 학습)", threshold)

    # ── Fine-tuned 모델 평가 ──
    ft_path = os.path.join(MODEL_DIR, "best_model_finetuned.pt")
    ft_thresh_path = os.path.join(MODEL_DIR, "threshold_finetuned.json")

    if os.path.exists(ft_path):
        ft_model = ConcentrationGRU().to(DEVICE)
        ft_model.load_state_dict(torch.load(ft_path, map_location=DEVICE, weights_only=True))

        with open(ft_thresh_path) as f:
            ft_threshold = json.load(f)["focused_threshold"]

        ft_results = evaluate(ft_model, X_scaled, y_test, ft_threshold)
        print_results(ft_results, "Fine-tuned 모델", ft_threshold)

        # 비교
        print(f"\n{'='*50}")
        print("  변화 요약")
        print(f"{'='*50}")
        delta_f1  = ft_results["f1_macro"] - results["f1_macro"]
        delta_acc = ft_results["accuracy"]  - results["accuracy"]
        delta_r0  = ft_results["recall_0"]  - results["recall_0"]
        print(f"  F1 Macro  : {results['f1_macro']:.4f} → {ft_results['f1_macro']:.4f}  ({delta_f1:+.4f})")
        print(f"  Accuracy  : {results['accuracy']*100:.1f}% → {ft_results['accuracy']*100:.1f}%  ({delta_acc*100:+.1f}%)")
        print(f"  Recall(비집중): {results['recall_0']:.3f} → {ft_results['recall_0']:.3f}  ({delta_r0:+.3f})")
    else:
        print("\nfine-tuned 모델 없음. 먼저 python src/finetune.py 실행하세요.")


if __name__ == "__main__":
    main()
