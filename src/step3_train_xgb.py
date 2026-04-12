"""
Step 3: XGBoost 모델 학습 (Baseline)

특징:
  - Clip-level tabular feature로 학습 (시계열 → 통계량 집계 후)
  - SMOTE로 클래스 균형 맞춤
  - Optuna로 하이퍼파라미터 자동 튜닝
  - Feature importance 시각화
  - 학습 결과 저장: 모델, threshold, 메트릭, 그래프

Usage:
  python step3_train_xgb.py
  python step3_train_xgb.py --no-tune   # 튜닝 없이 기본값으로 빠르게 학습
"""

import argparse
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report, f1_score, recall_score,
    precision_recall_curve, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

# ── 선택적 import ────────────────────────────────────────────────

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[WARN] optuna not installed. Using default hyperparameters.")


# ── 데이터 로드 ─────────────────────────────────────────────────

def load_data(features_dir):
    """train/val/test 데이터 로드"""
    splits = {}
    for name in ['train', 'val', 'test']:
        path = features_dir / f'{name}_ready.npz'
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run step2 first.")
        data = np.load(path, allow_pickle=True)
        splits[name] = {
            'X': data['X'],
            'y': data['y'],
            'feature_names': data['feature_names'],
        }
    return splits


# ── Optuna 하이퍼파라미터 튜닝 ──────────────────────────────────

def tune_hyperparams(X_train, y_train, X_val, y_val, n_trials=50):
    """Optuna로 XGBoost 하이퍼파라미터 탐색"""
    if not HAS_OPTUNA:
        return _default_params()

    print(f"\n[Tuning] Running {n_trials} trials...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
        }

        model = XGBClassifier(
            **params,
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False,
        )
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_val)
        # F1-macro를 최적화하되, recall(drowsy) >= 50% 제약
        f1 = f1_score(y_val, y_pred, average='macro')
        recall_drowsy = recall_score(y_val, y_pred, pos_label=0)

        # 패널티: recall(drowsy)가 50% 미만이면 점수 감소
        if recall_drowsy < 0.5:
            f1 *= 0.5

        return f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[Tuning] Best F1-macro: {study.best_value:.4f}")
    print(f"[Tuning] Best params: {study.best_params}")
    return study.best_params


def _default_params():
    """기본 하이퍼파라미터"""
    return {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 3,
        'gamma': 0.1,
        'scale_pos_weight': 1.5,
    }


# ── 학습 & 평가 ─────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, params):
    """XGBoost 모델 학습"""


    model = XGBClassifier(
        **params,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def find_optimal_threshold(model, X_val, y_val):
    """
    F1-macro를 최대화하면서 recall(drowsy) >= 50%인 최적 threshold 탐색
    """
    y_proba = model.predict_proba(X_val)[:, 1]  # P(alert)

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.3, 0.8, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, average='macro')
        recall_d = recall_score(y_val, y_pred, pos_label=0)

        if recall_d >= 0.5 and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n[Threshold] Optimal: {best_threshold:.2f} (F1-macro: {best_f1:.4f})")
    return best_threshold


def evaluate_model(model, X, y, threshold, split_name="Test"):
    """모델 평가 및 리포트 출력"""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"  {split_name} Set Evaluation (threshold={threshold:.2f})")
    print(f"{'='*50}")
    print(classification_report(
        y, y_pred,
        target_names=['Drowsy(0)', 'Alert(1)'],
        digits=4
    ))

    # AUC
    try:
        auc = roc_auc_score(y, y_proba)
        print(f"  AUC-ROC: {auc:.4f}")
    except ValueError:
        auc = 0.0

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    Predicted →  Drowsy  Alert")
    print(f"    Actual Drowsy  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"    Actual Alert   {cm[1][0]:5d}  {cm[1][1]:5d}")

    metrics = {
        'f1_macro': float(f1_score(y, y_pred, average='macro')),
        'recall_drowsy': float(recall_score(y, y_pred, pos_label=0)),
        'recall_alert': float(recall_score(y, y_pred, pos_label=1)),
        'auc_roc': float(auc),
        'accuracy': float(np.mean(y_pred == y)),
    }
    return metrics


# ── 시각화 ──────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, output_path, top_k=20):
    """Feature importance 시각화"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(top_k),
        importances[indices][::-1],
        color='steelblue'
    )
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_k} Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] Feature importance saved to {output_path}")


def plot_metrics_summary(val_metrics, test_metrics, output_path):
    """Val/Test 주요 지표를 한눈에 보여주는 요약 이미지"""
    labels = ['Accuracy', 'AUC-ROC', 'F1-macro', 'Recall\n(Drowsy)', 'Recall\n(Alert)']
    val_vals = [
        val_metrics['accuracy'],
        val_metrics['auc_roc'],
        val_metrics['f1_macro'],
        val_metrics['recall_drowsy'],
        val_metrics['recall_alert'],
    ]
    test_vals = [
        test_metrics['accuracy'],
        test_metrics['auc_roc'],
        test_metrics['f1_macro'],
        test_metrics['recall_drowsy'],
        test_metrics['recall_alert'],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_val  = ax.bar(x - width/2, val_vals,  width, label='Validation', color='steelblue',  alpha=0.85)
    bars_test = ax.bar(x + width/2, test_vals, width, label='Test',       color='darkorange', alpha=0.85)

    # 막대 위에 숫자 표시
    for bar in bars_val + bars_test:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Summary (XGBoost + UTA-RLDD)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] Metrics summary saved to {output_path}")


def plot_precision_recall(model, X_test, y_test, output_path):
    """Precision-Recall 커브"""
    y_proba = model.predict_proba(X_test)[:, 1]

    # Drowsy class (label=0)에 대한 PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, 1 - y_proba, pos_label=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.set_xlabel('Recall (Drowsy)')
    ax.set_ylabel('Precision (Drowsy)')
    ax.set_title('Precision-Recall Curve (Drowsy Class)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Plot] PR curve saved to {output_path}")


# ── 메인 실행 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip Optuna tuning, use defaults')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    features_dir = base / "data" / "features"
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    data = load_data(features_dir)
    X_train = data['train']['X']
    y_train = data['train']['y']
    X_val = data['val']['X']
    y_val = data['val']['y']
    X_test = data['test']['X']
    y_test = data['test']['y']
    feature_names = list(data['train']['feature_names'])

    print(f"Train: {X_train.shape}, Alert={np.sum(y_train==1)}, Drowsy={np.sum(y_train==0)}")
    print(f"Val:   {X_val.shape},  Alert={np.sum(y_val==1)},  Drowsy={np.sum(y_val==0)}")
    print(f"Test:  {X_test.shape}, Alert={np.sum(y_test==1)}, Drowsy={np.sum(y_test==0)}")

    # 하이퍼파라미터
    if args.no_tune:
        params = _default_params()
        print("\n[Config] Using default hyperparameters")
    else:
        params = tune_hyperparams(X_train, y_train, X_val, y_val, n_trials=args.trials)

    # 학습
    model = train_model(X_train, y_train, X_val, y_val, params)

    # 최적 threshold
    threshold = find_optimal_threshold(model, X_val, y_val)

    # 평가
    val_metrics = evaluate_model(model, X_val, y_val, threshold, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, threshold, "Test")

    # 저장
    model_path = models_dir / "xgb_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n[Save] Model → {model_path}")

    threshold_path = models_dir / "threshold.json"
    with open(threshold_path, 'w') as f:
        json.dump({
            'threshold': threshold,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'params': params,
        }, f, indent=2)
    print(f"[Save] Threshold & metrics → {threshold_path}")

    # 시각화
    plot_feature_importance(
        model, feature_names,
        models_dir / "feature_importance.png"
    )
    plot_precision_recall(
        model, X_test, y_test,
        models_dir / "pr_curve.png"
    )
    plot_metrics_summary(
        val_metrics, test_metrics,
        models_dir / "metrics_summary.png"
    )

    print("\n✓ Training complete!")
    print(f"  Test F1-macro: {test_metrics['f1_macro']:.4f}")
    print(f"  Test Recall(drowsy): {test_metrics['recall_drowsy']:.4f}")
    print(f"  Test Recall(alert):  {test_metrics['recall_alert']:.4f}")


if __name__ == "__main__":
    main()
