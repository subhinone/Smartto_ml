"""
특징 유효성 검증 스크립트
------------------------------------------------------
step1에서 추출한 특징이 집중/비집중을 실제로 구분하는지 확인합니다.

확인 항목:
  1. 클래스별 특징 분포 (박스플롯)
  2. 특징-레이블 상관관계
  3. 클래스 간 평균 차이 통계 (t-test)

실행 방법:
    python src/verify_features.py

출력:
    models/feature_distribution.png   ← 특징 분포 비교
    models/feature_correlation.png    ← 상관관계 히트맵
    (콘솔) t-test 결과 및 판단
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR  = os.path.join(BASE_DIR, "data", "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_NAMES = [
    "ear_avg", "ear_left", "ear_right", "mar",
    "pitch", "yaw", "roll", "face_detected",
    "ear_std", "mar_std", "pitch_std", "yaw_std",
    "blink_rate", "head_move_mag",
]


def load_flat_features(split: str):
    """
    .npz에서 특징을 로드하고 시퀀스 평균으로 압축합니다.
    (N, T, F) → (N, F): 각 클립의 프레임 평균값 사용
    """
    path = os.path.join(FEAT_DIR, f"{split}_features.npz")
    data = np.load(path, allow_pickle=True)
    X_raw = data["X"]   # object array of (T, F)
    y     = data["y"].astype(np.int32)

    # 각 클립을 프레임 평균으로 압축
    X_mean = np.array([seq.mean(axis=0) for seq in X_raw], dtype=np.float32)
    return X_mean, y


def main():
    print("=== 특징 유효성 검증 시작 ===\n")

    # train 데이터 로드
    X, y = load_flat_features("train")
    n_features = X.shape[1]
    feat_names = FEATURE_NAMES[:n_features]

    mask_0 = y == 0   # 비집중
    mask_1 = y == 1   # 집중

    print(f"전체: {len(y)}개  |  비집중(0): {mask_0.sum()}  |  집중(1): {mask_1.sum()}\n")

    # ──────────────────────────────────────────
    # 1. t-test: 각 특징의 클래스 간 차이 유의성
    # ──────────────────────────────────────────
    print("=" * 60)
    print(f"{'특징명':<16} {'비집중 평균':>10} {'집중 평균':>10} {'차이':>8} {'p-value':>10}  판정")
    print("-" * 60)

    discriminative = []
    for i, name in enumerate(feat_names):
        vals_0 = X[mask_0, i]
        vals_1 = X[mask_1, i]
        mean_0 = vals_0.mean()
        mean_1 = vals_1.mean()
        diff   = mean_1 - mean_0

        t_stat, p_val = stats.ttest_ind(vals_0, vals_1, equal_var=False)

        # p < 0.05 이고 평균 차이가 0.01 이상이면 유의미
        is_useful = p_val < 0.05 and abs(diff) > 0.01
        flag = "✅ 유의미" if is_useful else "❌ 무의미"
        if is_useful:
            discriminative.append(name)

        print(f"{name:<16} {mean_0:>10.4f} {mean_1:>10.4f} {diff:>8.4f} {p_val:>10.4f}  {flag}")

    print()
    print(f"유의미한 특징: {len(discriminative)}/{len(feat_names)}개")
    if discriminative:
        print(f"  → {', '.join(discriminative)}")
    else:
        print("  → 유의미한 특징 없음! 특징 자체를 재설계해야 합니다.")

    # ──────────────────────────────────────────
    # 2. 박스플롯: 특징 분포 시각화
    # ──────────────────────────────────────────
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()
    fig.suptitle("특징별 클래스 분포 비교 (비집중=0, 집중=1)", fontsize=13, fontweight="bold")

    for i, name in enumerate(feat_names):
        ax = axes[i]
        data_0 = X[mask_0, i]
        data_1 = X[mask_1, i]

        bp = ax.boxplot(
            [data_0, data_1],
            labels=["비집중(0)", "집중(1)"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#FF7F7F")
        bp["boxes"][1].set_facecolor("#7FB2FF")

        # 유의미한 특징에 별표
        title_flag = " ✅" if name in discriminative else " ❌"
        ax.set_title(f"{name}{title_flag}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # 남는 subplot 숨기기
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, "feature_distribution.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n박스플롯 저장 → models/feature_distribution.png")

    # ──────────────────────────────────────────
    # 3. 상관관계 히트맵
    # ──────────────────────────────────────────
    # 레이블과 각 특징의 상관계수
    corr_with_label = []
    for i in range(n_features):
        r, p = stats.pearsonr(X[:, i], y)
        corr_with_label.append(r)

    fig, ax = plt.subplots(figsize=(12, 3))
    colors = ["#d73027" if c < 0 else "#4575b4" for c in corr_with_label]
    bars = ax.bar(feat_names, corr_with_label, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=0.1,  color="green", linewidth=1, linestyle="--", alpha=0.5, label="r=±0.1 기준선")
    ax.axhline(y=-0.1, color="green", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("특징-레이블 상관계수 (절대값이 클수록 집중도 판별에 유용)", fontsize=12)
    ax.set_ylabel("Pearson r")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 막대 위에 수치 표시
    for bar, val in zip(bars, corr_with_label):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (0.003 if val >= 0 else -0.008),
                f"{val:.3f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path2 = os.path.join(MODEL_DIR, "feature_correlation.png")
    plt.savefig(out_path2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"상관관계 그래프 저장 → models/feature_correlation.png")

    # ──────────────────────────────────────────
    # 4. 최종 판단
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("=== 최종 판단 ===")
    ratio = len(discriminative) / len(feat_names)

    if ratio >= 0.5:
        print("특징 품질 양호 — 모델/학습 전략 개선으로 성능 향상 가능")
    elif ratio >= 0.3:
        print("특징 부분 유효 — 유의미한 특징만 선택하거나 추가 특징 설계 필요")
    else:
        print("특징 품질 불량 — EAR/MAR/HeadPose가 이 데이터셋 레이블과 상관관계 낮음")
        print("  → 규칙 기반 접근 또는 실사용 데이터 수집을 우선 권장")

    print("\n=== 검증 완료 ===")


if __name__ == "__main__":
    main()
