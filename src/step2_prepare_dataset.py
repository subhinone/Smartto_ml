"""
Step 2: 특징 데이터를 LSTM 학습용 데이터셋으로 가공
------------------------------------------------------
Step 1에서 추출한 .npz 파일을 불러와서:
  1. 시퀀스 길이를 고정 (패딩 / 트런케이션)
  2. 특징값 정규화 (StandardScaler)
  3. 학습/검증/테스트 데이터 저장

실행 방법:
    python src/step2_prepare_dataset.py

출력:
    data/features/train_ready.npz
    data/features/val_ready.npz
    data/features/test_ready.npz
    data/features/scaler.pkl   ← 앱 추론 시에도 동일하게 사용
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from collections import Counter

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR   = os.path.join(BASE_DIR, "data", "features")
SEQ_LEN    = 150   # 고정 시퀀스 길이 (약 5초 @ 30fps)
N_CLASSES  = 2     # 이진 분류: 0=비집중(Engagement 0,1) / 1=집중(Engagement 2,3)

# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

def load_npz(split: str):
    path = os.path.join(FEAT_DIR, f"{split}_features.npz")
    data = np.load(path, allow_pickle=True)
    X = data["X"]          # object array: 각 원소가 (T, 8) float32
    y = data["y"].astype(np.int32)
    print(f"[{split}] 로드: {len(X)}개 시퀀스, 레이블 분포: {dict(Counter(y.tolist()))}")
    return X, y


def binarize_labels(y: np.ndarray) -> np.ndarray:
    """Engagement 0,1 → 0 (비집중) / 2,3 → 1 (집중)"""
    return (y >= 2).astype(np.int32)


def pad_or_truncate(seq: np.ndarray, seq_len: int) -> np.ndarray:
    """
    시퀀스를 고정 길이로 맞춤.
    - 짧으면 앞을 0으로 패딩 (최근 프레임이 끝에 오도록)
    - 길면 마지막 seq_len 프레임만 사용
    """
    T, F = seq.shape
    if T >= seq_len:
        return seq[-seq_len:]          # 뒤쪽(최근) 프레임 사용
    else:
        pad = np.zeros((seq_len - T, F), dtype=np.float32)
        return np.concatenate([pad, seq], axis=0)


def build_array(X_list, seq_len: int) -> np.ndarray:
    """(N, seq_len, F) 배열로 변환"""
    return np.stack([pad_or_truncate(seq, seq_len) for seq in X_list], axis=0)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    print("=== Step 2: 데이터셋 준비 시작 ===\n")

    # 1. 로드
    X_train_raw, y_train = load_npz("train")
    X_val_raw,   y_val   = load_npz("validation")
    X_test_raw,  y_test  = load_npz("test")

    # 2. 레이블 이진화
    y_train = binarize_labels(y_train)
    y_val   = binarize_labels(y_val)
    y_test  = binarize_labels(y_test)
    print(f"\n이진 레이블 변환 완료")
    print(f"  Train  → 비집중: {(y_train==0).sum()}, 집중: {(y_train==1).sum()}")
    print(f"  Val    → 비집중: {(y_val==0).sum()}, 집중: {(y_val==1).sum()}")
    print(f"  Test   → 비집중: {(y_test==0).sum()}, 집중: {(y_test==1).sum()}")

    # 3. 고정 길이 배열로 변환 (N, SEQ_LEN, 8)
    print(f"\n시퀀스 길이 고정 중 (SEQ_LEN={SEQ_LEN})...")
    X_train = build_array(X_train_raw, SEQ_LEN)
    X_val   = build_array(X_val_raw,   SEQ_LEN)
    X_test  = build_array(X_test_raw,  SEQ_LEN)
    print(f"  Train shape : {X_train.shape}")
    print(f"  Val shape   : {X_val.shape}")
    print(f"  Test shape  : {X_test.shape}")

    # 4. 정규화 (Train 기준으로 fit → 모두 transform)
    print("\n특징 정규화 중 (StandardScaler)...")
    N_train, T, F = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, F)
    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train.reshape(-1, F)).reshape(N_train, T, F).astype(np.float32)
    X_val   = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape).astype(np.float32)
    X_test  = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape).astype(np.float32)
    print("  완료!")

    # 5. 저장
    print("\n저장 중...")
    np.savez(os.path.join(FEAT_DIR, "train_ready.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(FEAT_DIR, "val_ready.npz"),   X=X_val,   y=y_val)
    np.savez(os.path.join(FEAT_DIR, "test_ready.npz"),  X=X_test,  y=y_test)

    scaler_path = os.path.join(FEAT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n✅ 저장 완료!")
    print(f"   data/features/train_ready.npz  ({X_train.shape})")
    print(f"   data/features/val_ready.npz    ({X_val.shape})")
    print(f"   data/features/test_ready.npz   ({X_test.shape})")
    print(f"   data/features/scaler.pkl")
    print("\n=== Step 2 완료! ===")
    print("다음 단계: Google Colab에서 src/step3_train.py 실행")


if __name__ == "__main__":
    main()
