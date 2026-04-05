"""
Step 3: GRU 모델 학습 + threshold 최적화
------------------------------------------------------
이전 버전 대비 변경 사항:
  - LSTM → GRU (파라미터 감소, 수렴 속도 향상)
  - hidden_size 128 → 64, num_layers 2 → 1 (과대적합 방지)
  - Focal Loss 도입 (클래스 불균형에 강건)
  - Cosine Annealing LR 스케줄러
  - 데이터 증강 개선 (time warp, feature masking)
  - 학습률 warmup 적용

출력:
    models/best_model.pt
    models/threshold.json
    models/train_log.csv
    models/train_curve.png
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import recall_score, f1_score, confusion_matrix, precision_score

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 하이퍼파라미터
# ──────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 80
LR           = 5e-4
HIDDEN_SIZE  = 64
NUM_LAYERS   = 1
DROPOUT      = 0.4
SEQ_LEN      = 90
N_FEATURES   = 14     # 원시 8 + 시간적 6
N_CLASSES    = 2
ES_PATIENCE  = 20
ES_MIN_DELTA = 0.002
WARMUP_EPOCHS = 5

# Focal Loss 파라미터
FOCAL_ALPHA = 0.6     # 비집중(0) 클래스에 더 높은 가중치
FOCAL_GAMMA = 2.0     # 쉬운 샘플의 기여를 줄임

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
    """
    클래스 불균형에 강건한 Focal Loss.
    어렵게 분류되는 샘플에 더 높은 가중치를 부여합니다.
    """
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        # alpha: 클래스별 가중치 [비집중, 집중]
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma

    def forward(self, logits, targets):
        self.alpha = self.alpha.to(logits.device)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 정답 클래스의 확률
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class ConcentrationDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        data = np.load(os.path.join(DATA_DIR, f"{split}_ready.npz"))
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx].clone(), self.y[idx]
        if self.augment:
            x = self._augment(x)
        return x, y

    def _augment(self, x):
        """
        시계열 데이터에 적합한 증강:
        1. 가우시안 노이즈 추가
        2. 시간 구간 마스킹 (time masking)
        3. 특징 마스킹 (feature masking)
        4. 스케일 변동
        """
        # 1) 가우시안 노이즈 (매우 작게)
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * 0.015

        # 2) 시간 구간 마스킹 — 연속 5~15 프레임을 0으로
        if torch.rand(1).item() < 0.4:
            mask_len = torch.randint(5, 15, (1,)).item()
            start = torch.randint(0, max(1, x.shape[0] - mask_len), (1,)).item()
            x[start:start + mask_len] = 0

        # 3) 특징 마스킹 — 1~2개 특징 채널을 0으로
        if torch.rand(1).item() < 0.3:
            n_mask = torch.randint(1, 3, (1,)).item()
            mask_idx = torch.randperm(x.shape[1])[:n_mask]
            x[:, mask_idx] = 0

        # 4) 스케일 변동 (±10%)
        if torch.rand(1).item() < 0.4:
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            x = x * scale

        return x


# ──────────────────────────────────────────────
# Early Stopping (val F1-macro 기준)
# ──────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience, min_delta, save_path):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_path  = save_path
        self.best_score = 0.0
        self.counter    = 0

    def step(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


# ──────────────────────────────────────────────
# 모델 — 경량 GRU
# ──────────────────────────────────────────────

class ConcentrationGRU(nn.Module):
    """
    경량 GRU 모델.
    - GRU 1-layer (LSTM보다 파라미터 25% 적음)
    - Attention pooling (마지막 hidden state만 쓰는 것보다 효과적)
    - Dropout 강화
    """
    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention: 어떤 시간 스텝이 집중도 판단에 중요한지 학습
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        gru_out, _ = self.gru(x)  # (B, T, H)

        # Attention pooling
        attn_weights = self.attention(gru_out)     # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (gru_out * attn_weights).sum(dim=1)  # (B, H)

        return self.classifier(context)


# ──────────────────────────────────────────────
# 학습 / 평가 루프
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def get_probs_and_targets(model, loader):
    """소프트맥스 확률과 정답 레이블을 반환"""
    model.eval()
    all_probs, all_targets = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(y.numpy())
    return np.array(all_probs), np.array(all_targets)


def evaluate_with_threshold(probs, targets, threshold=0.5):
    """P(집중) >= threshold → 집중(1), else → 비집중(0)"""
    preds = (probs >= threshold).astype(int)
    acc      = (preds == targets).mean()
    recall_0 = recall_score(targets, preds, pos_label=0, zero_division=0)
    recall_1 = recall_score(targets, preds, pos_label=1, zero_division=0)
    prec_0   = precision_score(targets, preds, pos_label=0, zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    cm       = confusion_matrix(targets, preds, labels=[0, 1])
    return acc, recall_0, recall_1, prec_0, f1_macro, cm


def find_best_threshold(probs_val, targets_val):
    """
    val set에서 최적 threshold 탐색.
    조건: Recall(비집중) >= 40%
    목표: F1-macro 최대화
    """
    best_thresh = 0.5
    best_f1     = 0.0

    for thresh in np.arange(0.10, 0.91, 0.01):
        preds    = (probs_val >= thresh).astype(int)
        r0       = recall_score(targets_val, preds, pos_label=0, zero_division=0)
        f1_macro = f1_score(targets_val, preds, average="macro", zero_division=0)

        if r0 >= 0.40 and f1_macro > best_f1:
            best_f1     = f1_macro
            best_thresh = thresh

    return round(float(best_thresh), 2), round(float(best_f1), 4)


@torch.no_grad()
def evaluate(model, loader, criterion):
    probs, targets = get_probs_and_targets(model, loader)
    model.eval()
    total_loss, total = 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        total      += len(y)
    loss = total_loss / total
    acc, r0, r1, p0, f1_macro, _ = evaluate_with_threshold(probs, targets, 0.5)
    return loss, acc, r0, r1, f1_macro


# ──────────────────────────────────────────────
# Warmup + Cosine Annealing 스케줄러
# ──────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    print("=== Step 3: GRU 모델 학습 시작 ===\n")

    train_ds = ConcentrationDataset("train", augment=True)
    val_ds   = ConcentrationDataset("val",   augment=False)
    test_ds  = ConcentrationDataset("test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 실제 특징 수 확인
    actual_n_features = train_ds.X.shape[2]
    print(f"특징 수: {actual_n_features}")

    y_train = train_ds.y.numpy()
    n_neg   = (y_train == 0).sum()
    n_pos   = (y_train == 1).sum()
    print(f"Train: {len(train_ds)}개 | Val: {len(val_ds)}개 | Test: {len(test_ds)}개")
    print(f"Train 클래스 분포 → 비집중(0): {n_neg} ({n_neg/len(y_train)*100:.1f}%)  "
          f"집중(1): {n_pos} ({n_pos/len(y_train)*100:.1f}%)\n")

    model = ConcentrationGRU(
        input_size=actual_n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=N_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {n_params:,}\n")

    # Focal Loss
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    print(f"Loss: FocalLoss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, LR)
    es = EarlyStopping(ES_PATIENCE, ES_MIN_DELTA,
                       save_path=os.path.join(MODEL_DIR, "best_model.pt"))

    log_records = []
    print(f"\n{'Ep':>3} | {'LR':>8} | {'TrLoss':>7} {'TrAcc':>6} | "
          f"{'VaLoss':>7} {'VaAcc':>6} {'Rec0':>6} {'Rec1':>6} {'F1Mac':>6}")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        lr = scheduler.step(epoch - 1)
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_r0, val_r1, val_f1 = evaluate(model, val_loader, criterion)

        log_records.append({
            "epoch": epoch, "lr": round(lr, 6),
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
            "val_recall0": round(val_r0, 4), "val_recall1": round(val_r1, 4),
            "val_f1_macro": round(val_f1, 4),
        })

        stop = es.step(val_f1, model)
        flag = " *" if es.counter == 0 else f" ({es.counter}/{ES_PATIENCE})"
        print(f"{epoch:3d} | {lr:8.6f} | {tr_loss:7.4f} {tr_acc:6.4f} | "
              f"{val_loss:7.4f} {val_acc:6.4f} {val_r0:6.4f} {val_r1:6.4f} {val_f1:6.4f}{flag}")

        if stop:
            print(f"\n[Early Stopping] epoch {epoch}에서 중단 (best F1={es.best_score:.4f})")
            break

    # 로그 저장
    pd.DataFrame(log_records).to_csv(os.path.join(MODEL_DIR, "train_log.csv"), index=False)

    # ── Best model 로드 후 threshold 최적화 ──
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE, weights_only=True
    ))

    print("\n" + "=" * 50)
    print("threshold 최적화 중 (val set 기준)...")
    probs_val, targets_val = get_probs_and_targets(model, val_loader)

    neg_mask = targets_val == 0
    pos_mask = targets_val == 1
    if neg_mask.sum() > 0 and pos_mask.sum() > 0:
        print(f"  val P(집중) 분포 → 비집중 샘플: {probs_val[neg_mask].mean():.3f} (avg)  "
              f"집중 샘플: {probs_val[pos_mask].mean():.3f} (avg)")

    best_thresh, best_val_f1 = find_best_threshold(probs_val, targets_val)
    print(f"  최적 threshold: {best_thresh}  (val F1-macro: {best_val_f1:.4f})")

    # threshold 저장
    threshold_path = os.path.join(MODEL_DIR, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"focused_threshold": best_thresh, "val_f1_macro": best_val_f1}, f)
    print(f"  저장 → models/threshold.json")

    # ── 최종 테스트 평가 ──
    probs_test, targets_test = get_probs_and_targets(model, test_loader)

    print("\n" + "=" * 50)
    print("=== 최종 테스트 결과 ===\n")

    acc50, r0_50, r1_50, p0_50, f1_50, cm50 = evaluate_with_threshold(
        probs_test, targets_test, 0.5)
    print(f"[threshold=0.50] Acc: {acc50:.4f} | Recall(0): {r0_50:.4f}  Recall(1): {r1_50:.4f} "
          f"| Prec(0): {p0_50:.4f} | F1-Macro: {f1_50:.4f}")
    print(f"  혼동행렬:\n{cm50}\n")

    acc_t, r0_t, r1_t, p0_t, f1_t, cm_t = evaluate_with_threshold(
        probs_test, targets_test, best_thresh)
    print(f"[threshold={best_thresh:.2f}] Acc: {acc_t:.4f} | Recall(0): {r0_t:.4f}  Recall(1): {r1_t:.4f} "
          f"| Prec(0): {p0_t:.4f} | F1-Macro: {f1_t:.4f}")
    print(f"  혼동행렬:\n{cm_t}")
    print(f"\n  앱에서 사용: P(집중) < {best_thresh:.2f} → 비집중 판정")

    print(f"\n모델 저장 → models/best_model.pt")
    print(f"Threshold → models/threshold.json  (focused_threshold={best_thresh})")

    # ── 학습 곡선 그래프 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_log  = pd.DataFrame(log_records)
    best_ep = df_log["val_f1_macro"].idxmax() + 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"GRU | threshold={best_thresh} | Test Acc: {acc_t:.4f} | "
        f"Recall(unfocused): {r0_t:.4f} | F1-Macro: {f1_t:.4f}",
        fontsize=11, fontweight="bold"
    )

    ax = axes[0]
    ax.plot(df_log["epoch"], df_log["train_loss"], "b-o", ms=2, label="Train Loss")
    ax.plot(df_log["epoch"], df_log["val_loss"],   "r-o", ms=2, label="Val Loss")
    ax.axvline(x=best_ep, color="gray", ls=":", lw=1.2, label=f"Best ep {best_ep}")
    ax.set_title("Loss"); ax.set_xlabel("Epoch")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df_log["epoch"], df_log["train_acc"], "b-o", ms=2, label="Train Acc")
    ax.plot(df_log["epoch"], df_log["val_acc"],   "r-o", ms=2, label="Val Acc")
    ax.axvline(x=best_ep, color="gray", ls=":", lw=1.2, label=f"Best ep {best_ep}")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch")
    ax.set_ylim(0.3, 1.0); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(df_log["epoch"], df_log["val_recall0"], "r-o", ms=2, label="Val Recall(unfocused)")
    ax.plot(df_log["epoch"], df_log["val_recall1"], "b-o", ms=2, label="Val Recall(focused)")
    ax.plot(df_log["epoch"], df_log["val_f1_macro"], "g-s", ms=2, label="Val F1-macro")
    ax.axvline(x=best_ep, color="gray", ls=":", lw=1.2, label=f"Best ep {best_ep}")
    ax.set_title("Recall & F1"); ax.set_xlabel("Epoch")
    ax.set_ylim(0.0, 1.05); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "train_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"그래프 저장 → models/train_curve.png")
    print("\n=== Step 3 완료! ===")


if __name__ == "__main__":
    main()
