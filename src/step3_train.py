"""
Step 3: LSTM 모델 학습
------------------------------------------------------
Google Colab 또는 로컬에서 실행 가능합니다.

Colab 사용 시 사전 준비:
  1. Google Drive에 아래 파일 3개 업로드
       data/features/train_ready.npz
       data/features/val_ready.npz
       data/features/scaler.pkl
  2. Colab 상단에서 아래 셀 먼저 실행:
       from google.colab import drive
       drive.mount('/content/drive')
  3. DATA_DIR 경로를 Drive 경로로 수정 후 실행

로컬 실행:
    python src/step3_train.py

출력:
    models/best_model.pt   ← 가장 성능 좋은 모델
    models/train_log.csv   ← 에포크별 loss/accuracy 기록
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Colab 사용 시 아래 주석 해제 후 경로 수정
# DATA_DIR  = "/content/drive/MyDrive/Smartto_ml/data/features"
# MODEL_DIR = "/content/drive/MyDrive/Smartto_ml/models"

# ──────────────────────────────────────────────
# 하이퍼파라미터
# ──────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
SEQ_LEN     = 150
N_FEATURES  = 8
N_CLASSES   = 2

# ──────────────────────────────────────────────
# 디바이스 설정 (GPU / MPS / CPU 자동 선택)
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")   # M4 Pro Mac
else:
    DEVICE = torch.device("cpu")
print(f"사용 디바이스: {DEVICE}")

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class ConcentrationDataset(Dataset):
    def __init__(self, split: str):
        data = np.load(os.path.join(DATA_DIR, f"{split}_ready.npz"))
        self.X = torch.tensor(data["X"], dtype=torch.float32)  # (N, T, F)
        self.y = torch.tensor(data["y"], dtype=torch.long)     # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────

class ConcentrationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])   # 마지막 레이어의 hidden state 사용

# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total

# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    print("=== Step 3: LSTM 모델 학습 시작 ===\n")

    # 데이터 로드
    train_ds = ConcentrationDataset("train")
    val_ds   = ConcentrationDataset("val")
    test_ds  = ConcentrationDataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)}개 | Val: {len(val_ds)}개 | Test: {len(test_ds)}개\n")

    # 모델 초기화
    model = ConcentrationLSTM(
        input_size=N_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=N_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {total_params:,}\n")

    # 클래스 불균형 보정
    y_train = train_ds.y.numpy()
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # 학습
    best_val_acc = 0.0
    log = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)

        log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
        })

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
            flag = " ← 저장!"

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}{flag}")

    # 로그 저장
    pd.DataFrame(log).to_csv(os.path.join(MODEL_DIR, "train_log.csv"), index=False)

    # 최종 테스트 평가
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"\n=== 최종 결과 ===")
    print(f"Best Val Acc : {best_val_acc:.4f}")
    print(f"Test Acc     : {test_acc:.4f}")
    print(f"\n모델 저장 → models/best_model.pt")
    print(f"로그  저장 → models/train_log.csv")
    print("\n=== Step 3 완료! ===")


if __name__ == "__main__":
    main()
