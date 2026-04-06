"""
Fine-tuning 스크립트
------------------------------------------------------
collect_realdata.py로 수집한 실사용 데이터를
DAiSEE 학습 데이터와 합쳐 모델을 fine-tuning합니다.

전략:
  - DAiSEE 학습 데이터 + 실사용 데이터 혼합
  - 실사용 데이터에 더 높은 가중치 부여 (반복 횟수로 조절)
  - 낮은 학습률로 기존 지식 유지하면서 적응

실행 방법:
    python src/finetune.py

출력:
    models/best_model_finetuned.pt
    models/threshold_finetuned.json
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
from sklearn.metrics import f1_score, recall_score

# ──────────────────────────────────────────────
# step2와 동일한 스케일러 정의 (pickle 로드용)
# ──────────────────────────────────────────────
class RobustFeatureScaler:
    """face_detected 이진열(idx=7)은 정규화하지 않는 StandardScaler"""
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

    def inverse_transform(self, X_flat):
        return (X_flat * self.std_ + self.mean_).astype(np.float32)

# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "features")
REAL_DIR    = os.path.join(BASE_DIR, "data", "realdata")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
SEQ_LEN     = 90
N_FEATURES  = 14
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
DROPOUT     = 0.4
N_CLASSES   = 2

# fine-tuning 하이퍼파라미터
FT_LR        = 1e-4     # 기존 학습률(5e-4)보다 낮게
FT_EPOCHS    = 30
BATCH_SIZE   = 32
REAL_REPEAT  = 5        # 실사용 데이터를 5배 반복해서 강조
ES_PATIENCE  = 10

# ──────────────────────────────────────────────
# 디바이스
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"사용 디바이스: {DEVICE}")

# ──────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0):
        super().__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma

    def forward(self, logits, targets):
        self.alpha = self.alpha.to(logits.device)
        ce   = F.cross_entropy(logits, targets, reduction='none')
        pt   = torch.exp(-ce)
        loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce
        return loss.mean()

# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────

class ConcentrationGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS,
                          batch_first=True,
                          dropout=DROPOUT if NUM_LAYERS > 1 else 0)
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32), nn.Tanh(), nn.Linear(32, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 32), nn.GELU(),
            nn.Dropout(DROPOUT * 0.5), nn.Linear(32, N_CLASSES))

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn = torch.softmax(self.attention(gru_out), dim=1)
        ctx  = (gru_out * attn).sum(dim=1)
        return self.classifier(ctx)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class SequenceDataset(Dataset):
    """(N, SEQ_LEN, F) 형태의 이미 처리된 데이터"""
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx].clone(), self.y[idx]
        if self.augment:
            x += torch.randn_like(x) * 0.01
        return x, y


class RealDataset(Dataset):
    """
    collect_realdata.py로 수집한 프레임 단위 데이터를
    SEQ_LEN 길이 시퀀스로 변환합니다.
    """
    def __init__(self, npz_paths, scaler, seq_len=SEQ_LEN, augment=False):
        self.seq_len = seq_len
        self.augment = augment
        sequences, labels = [], []

        for path in npz_paths:
            data = np.load(path)
            X = data["X"].astype(np.float32)  # (N_frames, 13)
            y = data["y"].astype(np.int32)    # (N_frames,)

            # 정규화
            X = scaler.transform(X)

            # SEQ_LEN 길이 슬라이딩 윈도우로 시퀀스 생성
            # stride=15 (0.5초 간격 @ 30fps)
            stride = 15
            for start in range(0, len(X) - seq_len + 1, stride):
                seg_x = X[start:start+seq_len]
                seg_y = y[start:start+seq_len]
                # 시퀀스의 다수 레이블을 해당 시퀀스의 레이블로
                label = int(np.bincount(seg_y).argmax())
                sequences.append(seg_x)
                labels.append(label)

        if len(sequences) == 0:
            raise ValueError("수집된 데이터가 없습니다. collect_realdata.py를 먼저 실행하세요.")

        self.X = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        print(f"  실사용 데이터: {len(self.y)}개 시퀀스  "
              f"(집중: {(self.y==1).sum().item()}  비집중: {(self.y==0).sum().item()})")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx].clone(), self.y[idx]
        if self.augment:
            x += torch.randn_like(x) * 0.01
        return x, y

# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_targets = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        probs = torch.softmax(model(X), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(y.numpy())
    probs   = np.array(all_probs)
    targets = np.array(all_targets)
    preds   = (probs >= 0.5).astype(int)
    acc  = (preds == targets).mean()
    r0   = recall_score(targets, preds, pos_label=0, zero_division=0)
    r1   = recall_score(targets, preds, pos_label=1, zero_division=0)
    f1   = f1_score(targets, preds, average="macro", zero_division=0)
    return acc, r0, r1, f1, probs, targets


def find_best_threshold(probs, targets):
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        r0 = recall_score(targets, preds, pos_label=0, zero_division=0)
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        if r0 >= 0.40 and f1 > best_f1:
            best_f1, best_thresh = f1, t
    return round(float(best_thresh), 2), round(float(best_f1), 4)

# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    print("=== Fine-tuning 시작 ===\n")

    import pickle
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # 실사용 데이터 파일 탐색
    real_files = sorted(glob.glob(os.path.join(REAL_DIR, "session_*.npz")))
    if not real_files:
        print("❌ 수집된 실사용 데이터가 없습니다.")
        print("   먼저 python src/collect_realdata.py 를 실행하세요.")
        return

    print(f"실사용 데이터 파일 {len(real_files)}개 발견:")
    for f in real_files:
        print(f"  {os.path.basename(f)}")

    # DAiSEE 학습 데이터
    daisee_data = np.load(os.path.join(DATA_DIR, "train_ready.npz"))
    X_daisee = daisee_data["X"].astype(np.float32)
    y_daisee = daisee_data["y"].astype(np.int32)
    print(f"\nDAiSEE 학습 데이터: {len(y_daisee)}개")

    # Val 데이터 (평가용)
    val_data = np.load(os.path.join(DATA_DIR, "val_ready.npz"))
    val_ds   = SequenceDataset(val_data["X"], val_data["y"])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 실사용 Dataset 생성
    print("\n실사용 데이터 시퀀스 변환 중...")
    real_ds = RealDataset(real_files, scaler, augment=True)

    # 실사용 데이터를 REAL_REPEAT배 반복
    from torch.utils.data import ConcatDataset as CD
    real_repeated = CD([real_ds] * REAL_REPEAT)

    # DAiSEE + 실사용 혼합
    daisee_ds = SequenceDataset(X_daisee, y_daisee, augment=True)
    combined  = CD([daisee_ds, real_repeated])
    train_loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    y_all = np.concatenate([y_daisee] + [real_ds.y.numpy()] * REAL_REPEAT)
    n0 = (y_all == 0).sum()
    n1 = (y_all == 1).sum()
    print(f"\n혼합 데이터: {len(y_all)}개  "
          f"(비집중: {n0} {n0/len(y_all)*100:.1f}%  집중: {n1} {n1/len(y_all)*100:.1f}%)")

    # 모델 로드
    model = ConcentrationGRU().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "best_model.pt"),
        map_location=DEVICE, weights_only=True
    ))
    print(f"\n기존 모델 로드 완료 → fine-tuning 시작 (LR={FT_LR})")

    criterion = FocalLoss(alpha=0.6, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-3)

    best_f1    = 0.0
    best_path  = os.path.join(MODEL_DIR, "best_model_finetuned.pt")
    es_counter = 0

    print(f"\n{'Ep':>3} | {'TrLoss':>7} | {'ValAcc':>6} {'Rec0':>6} {'Rec1':>6} {'F1Mac':>6}")
    print("-" * 50)

    for epoch in range(1, FT_EPOCHS + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
        tr_loss = total_loss / len(combined)

        acc, r0, r1, f1, probs_val, targets_val = evaluate(model, val_loader)
        flag = ""
        if f1 > best_f1 + 0.001:
            best_f1 = f1
            es_counter = 0
            torch.save(model.state_dict(), best_path)
            flag = " *"
        else:
            es_counter += 1
            flag = f" ({es_counter}/{ES_PATIENCE})"

        print(f"{epoch:3d} | {tr_loss:7.4f} | {acc:6.4f} {r0:6.4f} {r1:6.4f} {f1:6.4f}{flag}")

        if es_counter >= ES_PATIENCE:
            print(f"\n[Early Stopping] epoch {epoch}에서 중단")
            break

    # Best 모델로 threshold 재최적화
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    _, _, _, _, probs_val, targets_val = evaluate(model, val_loader)
    best_thresh, best_val_f1 = find_best_threshold(probs_val, targets_val)

    thresh_path = os.path.join(MODEL_DIR, "threshold_finetuned.json")
    with open(thresh_path, "w") as f:
        json.dump({"focused_threshold": best_thresh, "val_f1_macro": best_val_f1}, f)

    print(f"\n=== Fine-tuning 완료 ===")
    print(f"  모델   → models/best_model_finetuned.pt")
    print(f"  threshold → {best_thresh}  (val F1: {best_val_f1:.4f})")
    print(f"\nstep4_realtime.py에서 MODEL_PATH를 best_model_finetuned.pt로 변경하면 됩니다.")


if __name__ == "__main__":
    main()
