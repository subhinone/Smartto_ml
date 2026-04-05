"""
Step 2: 특징 데이터를 학습용 데이터셋으로 가공
------------------------------------------------------
Step 1에서 추출한 .npz 파일을 불러와서:
  1. 시퀀스 길이를 고정 (패딩 / 트런케이션)
  2. 특징값 정규화 (StandardScaler)
  3. 클래스 불균형 대응 (비집중 클래스 오버샘플링)
  4. 저장

레이블은 step1에서 이미 다중 조건으로 이진화되었으므로
.npz의 y 값을 그대로 사용합니다.

실행 방법:
    python src/step2_prepare_dataset.py

출력:
    data/features/train_ready.npz
    data/features/val_ready.npz
    data/features/test_ready.npz
    data/features/scaler.pkl
"""

import numpy as np
import pickle
import os

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE_DIR, "data", "features")
SEQ_LEN  = 90       # 고정 시퀀스 길이 (약 3초 @ 30fps)

# 오버샘플링 목표 비율 — 비집중 클래스가 전체의 약 30~40%가 되도록
OVERSAMPLE_TARGET_RATIO = 0.35


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def load_npz(split: str):
    """
    .npz에서 X, y, clip_ids를 로드합니다.
    step1에서 저장한 레이블(다중 조건 이진화)을 그대로 사용합니다.
    """
    path = os.path.join(FEAT_DIR, f"{split}_features.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}\n  → step1을 먼저 실행하세요.")

    data = np.load(path, allow_pickle=True)
    X        = data["X"]           # object array of (T_i, N_FEATURES)
    y        = data["y"].astype(np.int32)
    clip_ids = data["clip_ids"].astype(str)

    print(f"[{split}] 로드: {len(X)}개 시퀀스  "
          f"(y 범위: {y.min()}~{y.max()}, unique={np.unique(y).tolist()})")
    print(f"  클래스 분포 → 비집중(0): {(y==0).sum()}  집중(1): {(y==1).sum()}")
    return X, y, clip_ids


def pad_or_truncate(seq: np.ndarray, seq_len: int) -> np.ndarray:
    """시퀀스를 고정 길이로 맞춤 (뒤쪽 = 최근 프레임 보존)"""
    T, F = seq.shape
    if T >= seq_len:
        return seq[-seq_len:]
    else:
        pad = np.zeros((seq_len - T, F), dtype=np.float32)
        return np.concatenate([pad, seq], axis=0)


def build_array(X_list, seq_len: int) -> np.ndarray:
    """가변 길이 시퀀스 리스트를 (N, seq_len, F) 배열로 변환"""
    return np.stack([pad_or_truncate(seq, seq_len) for seq in X_list], axis=0)


def oversample_minority(X: np.ndarray, y: np.ndarray,
                         target_ratio: float = OVERSAMPLE_TARGET_RATIO):
    """
    비집중(0) 클래스를 오버샘플링하여 클래스 불균형을 완화합니다.

    - 단순 복제 + 약간의 노이즈 추가 (SMOTE 대신 시퀀스 데이터에 적합한 방식)
    - target_ratio: 오버샘플링 후 비집중 클래스가 전체에서 차지할 비율

    Returns:
        X_balanced, y_balanced
    """
    mask_0 = y == 0
    mask_1 = y == 1

    X_0, X_1 = X[mask_0], X[mask_1]
    n_0, n_1 = len(X_0), len(X_1)

    if n_0 == 0:
        print("  ⚠ 비집중 샘플이 0개입니다. 오버샘플링 불가.")
        return X, y

    # 목표 개수 계산: n_0_target / (n_0_target + n_1) = target_ratio
    n_0_target = int(n_1 * target_ratio / (1 - target_ratio))
    n_0_target = max(n_0_target, n_0)  # 최소한 원본 유지

    print(f"  오버샘플링: 비집중 {n_0} → {n_0_target}  (집중: {n_1})")

    if n_0_target <= n_0:
        print("  오버샘플링 불필요 (이미 충분)")
        return X, y

    # 복제 + 미세 노이즈 추가
    n_extra = n_0_target - n_0
    rng = np.random.default_rng(42)
    indices = rng.choice(n_0, size=n_extra, replace=True)
    X_extra = X_0[indices].copy()

    # 미세 가우시안 노이즈 (원본 특징의 1~3% 수준)
    noise_scale = 0.02
    X_extra += rng.normal(0, noise_scale, size=X_extra.shape).astype(np.float32)

    X_balanced = np.concatenate([X_1, X_0, X_extra], axis=0)
    y_balanced = np.concatenate([
        np.ones(n_1, dtype=np.int32),
        np.zeros(n_0, dtype=np.int32),
        np.zeros(n_extra, dtype=np.int32),
    ])

    # 셔플
    perm = rng.permutation(len(y_balanced))
    return X_balanced[perm], y_balanced[perm]


# ──────────────────────────────────────────────
# 정규화
# ──────────────────────────────────────────────

class RobustFeatureScaler:
    """
    StandardScaler와 유사하지만, face_detected 열(이진값)은 정규화하지 않습니다.
    """
    def __init__(self, binary_col_idx=7):
        self.binary_col_idx = binary_col_idx
        self.mean_ = None
        self.std_ = None

    def fit(self, X_flat):
        """X_flat: (N*T, F)"""
        self.mean_ = np.mean(X_flat, axis=0).astype(np.float32)
        self.std_  = np.std(X_flat, axis=0).astype(np.float32)
        self.std_[self.std_ < 1e-8] = 1.0  # 0으로 나누기 방지

        # face_detected 열은 정규화 안 함
        self.mean_[self.binary_col_idx] = 0.0
        self.std_[self.binary_col_idx]  = 1.0
        return self

    def transform(self, X_flat):
        return ((X_flat - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, X_flat):
        return (X_flat * self.std_ + self.mean_).astype(np.float32)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    print("=== Step 2: 데이터셋 준비 시작 ===\n")

    X_train_raw, y_train, _ = load_npz("train")
    X_val_raw,   y_val,   _ = load_npz("validation")
    X_test_raw,  y_test,  _ = load_npz("test")

    # 고정 길이 배열로 변환
    print(f"\n시퀀스 길이 고정 (SEQ_LEN={SEQ_LEN})...")
    X_train = build_array(X_train_raw, SEQ_LEN)
    X_val   = build_array(X_val_raw,   SEQ_LEN)
    X_test  = build_array(X_test_raw,  SEQ_LEN)
    N_FEATURES = X_train.shape[2]
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")

    # 정규화 (Train 기준 fit)
    print("\n특징 정규화 중...")
    N_train, T, F = X_train.shape
    scaler = RobustFeatureScaler(binary_col_idx=7)
    scaler.fit(X_train.reshape(-1, F))
    X_train = scaler.transform(X_train.reshape(-1, F)).reshape(N_train, T, F)
    X_val   = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
    print("  완료!")

    # 오버샘플링 (Train만)
    print(f"\n클래스 불균형 대응 (오버샘플링)...")
    print(f"  오버샘플링 전 → 비집중(0): {(y_train==0).sum()}  집중(1): {(y_train==1).sum()}")
    X_train, y_train = oversample_minority(X_train, y_train)
    print(f"  오버샘플링 후 → 비집중(0): {(y_train==0).sum()}  집중(1): {(y_train==1).sum()}")
    print(f"  최종 Train shape: {X_train.shape}")

    # 저장
    print("\n저장 중...")
    np.savez(os.path.join(FEAT_DIR, "train_ready.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(FEAT_DIR, "val_ready.npz"),   X=X_val,   y=y_val)
    np.savez(os.path.join(FEAT_DIR, "test_ready.npz"),  X=X_test,  y=y_test)
    with open(os.path.join(FEAT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n=== Step 2 완료! ===")
    print(f"  data/features/train_ready.npz  {X_train.shape}")
    print(f"  data/features/val_ready.npz    {X_val.shape}")
    print(f"  data/features/test_ready.npz   {X_test.shape}")
    print(f"  data/features/scaler.pkl")
    print(f"\n다음 단계: python src/step3_train.py")


if __name__ == "__main__":
    main()
