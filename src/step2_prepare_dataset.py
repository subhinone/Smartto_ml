"""
Step 2: 프레임 단위 feature → 클립 단위 통계량으로 집계 (XGBoost용 tabular 데이터)

각 클립의 시계열 feature를 통계량으로 변환:
  - 기본 통계: mean, std, min, max, median
  - 눈 관련: blink_rate, long_closure_rate, ear_range
  - 입 관련: yawn_rate (MAR > threshold 비율)
  - 머리 관련: head_movement_magnitude, pose_stability
  - 시선 관련: gaze_dispersion, off_screen_rate
  - 얼굴 관련: face_detection_rate

Usage:
  python step2_prepare_dataset.py
"""

import numpy as np
from pathlib import Path


# ── Feature 이름 (step1에서 추출한 순서) ────────────────────────
RAW_FEATURE_NAMES = [
    'ear_left', 'ear_right', 'ear_avg', 'mar',
    'pitch', 'yaw', 'roll',
    'gaze_h', 'gaze_v',
    'face_detected'
]

EAR_BLINK_THRESHOLD = 0.21   # EAR 아래면 눈 감은 것으로 판단
EAR_LONG_CLOSE_FRAMES = 10   # 연속 N프레임 이상 눈감으면 long closure
MAR_YAWN_THRESHOLD = 0.65    # MAR 위면 하품으로 판단
GAZE_OFF_THRESHOLD = 0.35    # 시선 오프셋이 이 이상이면 화면 밖


def compute_clip_features(frames):
    """
    프레임별 feature 배열 (N, 10) → 클립 통계량 벡터 (1D)

    Args:
        frames: np.ndarray of shape (num_frames, 10)
    Returns:
        feature_vector: np.ndarray (1D), feature_names: list[str]
    """
    features = {}
    n_frames = len(frames)

    if n_frames == 0:
        return None, None

    # object array로 저장된 경우 float32로 명시 변환
    frames = np.array(frames, dtype=np.float32)

    # 얼굴 감지된 프레임만 필터링
    face_mask = frames[:, 9] > 0.5
    n_face = np.sum(face_mask)
    features['face_detection_rate'] = n_face / n_frames

    if n_face < 5:
        # 얼굴이 거의 안 잡히면 비집중으로 간주될 feature 생성
        return _empty_features(features), None

    face_frames = frames[face_mask]

    # ── 1. 기본 통계 (얼굴 감지된 프레임 기준) ──
    for i, name in enumerate(RAW_FEATURE_NAMES[:9]):  # face_detected 제외
        col = face_frames[:, i]
        features[f'{name}_mean'] = np.mean(col)
        features[f'{name}_std'] = np.std(col)
        features[f'{name}_min'] = np.min(col)
        features[f'{name}_max'] = np.max(col)
        features[f'{name}_median'] = np.median(col)

    # ── 2. 눈 관련 feature ──
    ear_avg = face_frames[:, 2]

    # Blink rate: EAR < threshold인 프레임 비율
    blink_frames = ear_avg < EAR_BLINK_THRESHOLD
    features['blink_rate'] = np.mean(blink_frames)

    # Long closure rate: 연속으로 눈 감은 구간의 비율
    long_closures = _count_consecutive_runs(blink_frames, EAR_LONG_CLOSE_FRAMES)
    features['long_closure_count'] = long_closures
    features['long_closure_rate'] = (long_closures * EAR_LONG_CLOSE_FRAMES) / n_face

    # EAR 변동성 (눈 깜빡임 패턴)
    features['ear_range'] = np.max(ear_avg) - np.min(ear_avg)

    # Blink 간격 분산 (규칙적 vs 불규칙적)
    blink_intervals = _compute_blink_intervals(blink_frames)
    if len(blink_intervals) >= 2:
        features['blink_interval_mean'] = np.mean(blink_intervals)
        features['blink_interval_std'] = np.std(blink_intervals)
    else:
        features['blink_interval_mean'] = 0.0
        features['blink_interval_std'] = 0.0

    # ── 3. 입 관련 feature ──
    mar = face_frames[:, 3]
    features['yawn_rate'] = np.mean(mar > MAR_YAWN_THRESHOLD)
    features['mar_range'] = np.max(mar) - np.min(mar)

    # ── 4. 머리 움직임 feature ──
    pitch = face_frames[:, 4]
    yaw = face_frames[:, 5]
    roll = face_frames[:, 6]

    # 연속 프레임 간 머리 움직임 크기
    if len(face_frames) > 1:
        d_pitch = np.diff(pitch)
        d_yaw = np.diff(yaw)
        d_roll = np.diff(roll)
        head_movement = np.abs(d_pitch) + np.abs(d_yaw) + np.abs(d_roll)
        features['head_move_mean'] = np.mean(head_movement)
        features['head_move_max'] = np.max(head_movement)
        features['head_move_std'] = np.std(head_movement)
    else:
        features['head_move_mean'] = 0.0
        features['head_move_max'] = 0.0
        features['head_move_std'] = 0.0

    # 고개 숙임 감지 (pitch가 큰 음수 = 아래를 봄)
    features['head_down_rate'] = np.mean(pitch < -15)

    # ── 5. 시선 관련 feature ──
    gaze_h = face_frames[:, 7]
    gaze_v = face_frames[:, 8]

    # 시선 분산 (집중하면 시선이 안정적)
    features['gaze_h_std'] = np.std(gaze_h)
    features['gaze_v_std'] = np.std(gaze_v)
    gaze_dist = np.sqrt(gaze_h**2 + gaze_v**2)
    features['gaze_dispersion'] = np.std(gaze_dist)

    # 화면 밖 시선 비율
    features['off_screen_rate'] = np.mean(gaze_dist > GAZE_OFF_THRESHOLD)

    return features, None


def _empty_features(base_features):
    """얼굴 미감지 시 기본값"""
    # 모든 feature를 0으로 채우되, face_detection_rate만 유지
    all_names = _get_feature_names()
    result = {name: 0.0 for name in all_names}
    result['face_detection_rate'] = base_features.get('face_detection_rate', 0.0)
    return result


def _count_consecutive_runs(binary_array, min_length):
    """연속 True 구간 중 min_length 이상인 것의 개수"""
    count = 0
    run_length = 0
    for val in binary_array:
        if val:
            run_length += 1
        else:
            if run_length >= min_length:
                count += 1
            run_length = 0
    if run_length >= min_length:
        count += 1
    return count


def _compute_blink_intervals(blink_frames):
    """눈 감은 구간 사이의 간격(프레임 수) 계산"""
    intervals = []
    in_blink = False
    gap = 0
    for val in blink_frames:
        if val:
            if not in_blink and gap > 0:
                intervals.append(gap)
            in_blink = True
            gap = 0
        else:
            in_blink = False
            gap += 1
    return intervals


def _get_feature_names():
    """모든 clip-level feature 이름 반환 (순서 보장)"""
    names = ['face_detection_rate']

    for raw_name in RAW_FEATURE_NAMES[:9]:
        for stat in ['mean', 'std', 'min', 'max', 'median']:
            names.append(f'{raw_name}_{stat}')

    names.extend([
        'blink_rate', 'long_closure_count', 'long_closure_rate',
        'ear_range', 'blink_interval_mean', 'blink_interval_std',
        'yawn_rate', 'mar_range',
        'head_move_mean', 'head_move_max', 'head_move_std',
        'head_down_rate',
        'gaze_h_std', 'gaze_v_std', 'gaze_dispersion', 'off_screen_rate',
    ])
    return names


# ── 메인 실행 ───────────────────────────────────────────────────

def process_split(input_path, output_path):
    """한 split의 frame features → clip features 변환"""
    print(f"\nProcessing {input_path}...")
    data = np.load(input_path, allow_pickle=True)

    frame_features = data['features']  # array of arrays (variable length)
    labels = data['labels']
    clip_ids = data['clip_ids']

    feature_names = _get_feature_names()
    n_features = len(feature_names)

    X = np.zeros((len(frame_features), n_features), dtype=np.float32)
    valid_mask = np.ones(len(frame_features), dtype=bool)

    for i, frames in enumerate(frame_features):
        clip_feats, _ = compute_clip_features(frames)
        if clip_feats is None:
            valid_mask[i] = False
            continue

        for j, name in enumerate(feature_names):
            X[i, j] = clip_feats.get(name, 0.0)

    # 유효한 샘플만 저장
    X = X[valid_mask]
    y = labels[valid_mask]
    ids = clip_ids[valid_mask]

    print(f"  Valid samples: {len(X)} / {len(frame_features)}")
    print(f"  Alert: {np.sum(y==1)}, Drowsy: {np.sum(y==0)}")
    print(f"  Features: {n_features}")

    # NaN/Inf 처리
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        clip_ids=ids,
        feature_names=np.array(feature_names),
    )
    print(f"  Saved to {output_path}")


def merge_utarldd(features_dir, train_X, train_y, train_ids, feature_names):
    """
    UTA-RLDD feature를 train/val/test로 분리해서 각 split에 병합.
    DAiSEE val/test는 라벨이 EAR/MAR 신호와 안 맞으므로,
    UTA-RLDD를 70/15/15로 나눠서 val/test 평가도 UTA-RLDD 기준으로 변경.
    """
    utarldd_path = features_dir / "utarldd_features.npz"
    if not utarldd_path.exists():
        print("[SKIP] utarldd_features.npz not found. Run step1b first.")
        return train_X, train_y, train_ids, None, None, None, None, None, None

    print("\n[UTA-RLDD] 병합 중...")
    data = np.load(utarldd_path, allow_pickle=True)
    frame_features = data['features']
    labels = data['labels']
    clip_ids = data['clip_ids']

    X_extra = np.zeros((len(frame_features), len(feature_names)), dtype=np.float32)
    valid_mask = np.ones(len(frame_features), dtype=bool)

    for i, frames in enumerate(frame_features):
        clip_feats, _ = compute_clip_features(frames)
        if clip_feats is None:
            valid_mask[i] = False
            continue
        for j, name in enumerate(feature_names):
            X_extra[i, j] = clip_feats.get(name, 0.0)

    X_extra = X_extra[valid_mask]
    y_extra = labels[valid_mask]
    ids_extra = clip_ids[valid_mask]

    X_extra = np.nan_to_num(X_extra, nan=0.0, posinf=0.0, neginf=0.0)

    n_alert = np.sum(y_extra == 1)
    n_drowsy = np.sum(y_extra == 0)
    print(f"[UTA-RLDD] 유효 클립: {len(X_extra)} "
          f"(Alert={n_alert}, Drowsy={n_drowsy})")

    # UTA-RLDD를 70/15/15로 분리 (재현성을 위해 seed 고정)
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(len(X_extra))
    n = len(idx)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    idx_train = idx[:n_train]
    idx_val   = idx[n_train:n_train + n_val]
    idx_test  = idx[n_train + n_val:]

    Xu_train, yu_train, idu_train = X_extra[idx_train], y_extra[idx_train], ids_extra[idx_train]
    Xu_val,   yu_val,   idu_val   = X_extra[idx_val],   y_extra[idx_val],   ids_extra[idx_val]
    Xu_test,  yu_test,  idu_test  = X_extra[idx_test],  y_extra[idx_test],  ids_extra[idx_test]

    print(f"[UTA-RLDD Split] Train={len(Xu_train)}, Val={len(Xu_val)}, Test={len(Xu_test)}")

    # Train: DAiSEE train + UTA-RLDD train
    merged_X = np.concatenate([train_X, Xu_train], axis=0)
    merged_y = np.concatenate([train_y, yu_train], axis=0)
    merged_ids = np.concatenate([train_ids, idu_train], axis=0)

    print(f"[Merged] Train: {len(merged_X)} "
          f"(Alert={np.sum(merged_y==1)}, Drowsy={np.sum(merged_y==0)})")

    return (merged_X, merged_y, merged_ids,
            Xu_val, yu_val, idu_val,
            Xu_test, yu_test, idu_test)


def main():
    base = Path(__file__).resolve().parent.parent
    features_dir = base / "data" / "features"
    feature_names = _get_feature_names()

    splits = {
        'train': features_dir / 'train_features.npz',
        'val': features_dir / 'validation_features.npz',
        'test': features_dir / 'test_features.npz',
    }

    processed = {}
    for split_name, input_path in splits.items():
        if not input_path.exists():
            print(f"[SKIP] {input_path} not found. Run step1 first.")
            continue
        output_path = features_dir / f'{split_name}_ready.npz'
        process_split(input_path, output_path)

        # train은 나중에 UTA-RLDD 병합을 위해 로드
        if split_name == 'train':
            d = np.load(output_path, allow_pickle=True)
            processed['X'] = d['X']
            processed['y'] = d['y']
            processed['ids'] = d['clip_ids']

    # UTA-RLDD 병합 (utarldd_features.npz가 있는 경우)
    if 'X' in processed:
        result = merge_utarldd(
            features_dir,
            processed['X'], processed['y'], processed['ids'],
            feature_names
        )
        (merged_X, merged_y, merged_ids,
         Xu_val, yu_val, idu_val,
         Xu_test, yu_test, idu_test) = result

        # 병합된 train 저장
        train_path = features_dir / 'train_ready.npz'
        np.savez_compressed(
            train_path,
            X=merged_X, y=merged_y, clip_ids=merged_ids,
            feature_names=np.array(feature_names),
        )
        print(f"[Merged] train_ready.npz 업데이트 완료 → {train_path}")

        # val/test를 UTA-RLDD 기준으로 교체 (라벨-신호 정합성 확보)
        if Xu_val is not None:
            val_path = features_dir / 'val_ready.npz'
            np.savez_compressed(
                val_path,
                X=Xu_val, y=yu_val, clip_ids=idu_val,
                feature_names=np.array(feature_names),
            )
            print(f"[UTA-RLDD] val_ready.npz  → {val_path} "
                  f"(Alert={np.sum(yu_val==1)}, Drowsy={np.sum(yu_val==0)})")

            test_path = features_dir / 'test_ready.npz'
            np.savez_compressed(
                test_path,
                X=Xu_test, y=yu_test, clip_ids=idu_test,
                feature_names=np.array(feature_names),
            )
            print(f"[UTA-RLDD] test_ready.npz → {test_path} "
                  f"(Alert={np.sum(yu_test==1)}, Drowsy={np.sum(yu_test==0)})")

    print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    main()
