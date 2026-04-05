"""
Step 1: DAiSEE 데이터셋에서 MediaPipe로 특징 추출
------------------------------------------------------
프레임별 원시 특징 + 윈도우 기반 시간적 통계 특징을 추출합니다.

원시 특징 (per-frame):
    EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio),
    Head Pose (pitch, yaw, roll), face_detected

시간적 통계 특징 (per-window):
    ear_std, mar_std, pitch_std, yaw_std,
    blink_rate, head_movement_magnitude

레이블 전략 (다중 조건 이진화):
    집중(1):  Engagement >= 2 AND Boredom <= 1 AND Confusion <= 1 AND Frustration <= 1
    비집중(0): 그 외 모든 경우

실행 방법:
    python src/step1_extract_features.py

출력:
    data/features/train_features.npz
    data/features/val_features.npz
    data/features/test_features.npz
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAISEE_ROOT = os.path.join(BASE_DIR, "data", "DAiSEE", "DAiSEE")
LABELS_DIR = os.path.join(DAISEE_ROOT, "Labels")
DATASET_DIR = os.path.join(DAISEE_ROOT, "DataSet")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# MediaPipe FaceMesh 랜드마크 인덱스 정의
# ──────────────────────────────────────────────
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]
MOUTH_MAR = [61, 291, 0, 17]

NOSE_TIP     = 1
CHIN         = 152
LEFT_EYE_C   = 263
RIGHT_EYE_C  = 33
LEFT_MOUTH   = 287
RIGHT_MOUTH  = 57

# ──────────────────────────────────────────────
# 시간적 특징 설정
# ──────────────────────────────────────────────
TEMPORAL_WINDOW = 30          # 통계 계산 윈도우 (프레임)
EAR_BLINK_THRESHOLD = 0.2    # EAR이 이 값 이하면 눈 감김(blink) 판정

# ──────────────────────────────────────────────
# 특징 계산 함수
# ──────────────────────────────────────────────

def compute_ear(landmarks, indices):
    """Eye Aspect Ratio: 눈 감김 정도 (낮을수록 졸림)"""
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_mar(landmarks):
    """Mouth Aspect Ratio: 입 벌림 정도 (하품 감지)"""
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in MOUTH_MAR])
    vertical   = np.linalg.norm(pts[2] - pts[3])
    horizontal = np.linalg.norm(pts[0] - pts[1])
    return vertical / (horizontal + 1e-6)


def compute_head_pose(landmarks, img_w, img_h):
    """
    Head Pose 추정 (solvePnP 사용)
    반환: pitch(고개 끄덕임), yaw(좌우 회전), roll(기울임)  단위: 도(degree)
    """
    face_3d = np.array([
        [0.0,    0.0,    0.0  ],
        [0.0,   -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0,  170.0, -135.0],
        [-150.0,-150.0, -125.0],
        [150.0, -150.0, -125.0],
    ], dtype=np.float64)

    key_pts = [NOSE_TIP, CHIN, LEFT_EYE_C, RIGHT_EYE_C, LEFT_MOUTH, RIGHT_MOUTH]
    face_2d = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in key_pts],
        dtype=np.float64
    )

    focal = img_w
    cam_matrix = np.array([
        [focal, 0,     img_w / 2],
        [0,     focal, img_h / 2],
        [0,     0,     1        ]
    ], dtype=np.float64)
    dist = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist)
    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rot_mat)
    pitch = angles[0] * 360
    yaw   = angles[1] * 360
    roll  = angles[2] * 360
    return pitch, yaw, roll


# ──────────────────────────────────────────────
# 시간적 통계 특징 계산
# ──────────────────────────────────────────────

def compute_temporal_features(raw_features: np.ndarray, window: int = TEMPORAL_WINDOW) -> np.ndarray:
    """
    프레임별 원시 특징 시퀀스로부터 시간적 통계 특징을 계산합니다.

    입력: raw_features (T, 8) — [ear_avg, ear_l, ear_r, mar, pitch, yaw, roll, face_detected]
    출력: temporal (T, 6)    — [ear_std, mar_std, pitch_std, yaw_std, blink_rate, head_move_mag]

    각 프레임 t에서 [max(0, t-window+1) : t+1] 윈도우의 통계를 계산합니다.
    """
    T = raw_features.shape[0]
    temporal = np.zeros((T, 6), dtype=np.float32)

    ear_avg = raw_features[:, 0]
    mar     = raw_features[:, 3]
    pitch   = raw_features[:, 4]
    yaw     = raw_features[:, 5]

    for t in range(T):
        start = max(0, t - window + 1)
        w_ear   = ear_avg[start:t+1]
        w_mar   = mar[start:t+1]
        w_pitch = pitch[start:t+1]
        w_yaw   = yaw[start:t+1]

        # 표준편차 — 변동이 클수록 비집중 가능성
        temporal[t, 0] = np.std(w_ear)
        temporal[t, 1] = np.std(w_mar)
        temporal[t, 2] = np.std(w_pitch)
        temporal[t, 3] = np.std(w_yaw)

        # 눈 깜빡임 빈도 (윈도우 내 blink 횟수 / 윈도우 길이)
        blinks = np.sum(w_ear < EAR_BLINK_THRESHOLD)
        temporal[t, 4] = blinks / len(w_ear)

        # 머리 움직임 크기 (pitch + yaw 변화의 누적)
        if len(w_pitch) > 1:
            d_pitch = np.abs(np.diff(w_pitch))
            d_yaw   = np.abs(np.diff(w_yaw))
            temporal[t, 5] = (np.mean(d_pitch) + np.mean(d_yaw))
        else:
            temporal[t, 5] = 0.0

    return temporal


# ──────────────────────────────────────────────
# 최종 특징 이름 (원시 8 + 시간적 6 = 14)
# ──────────────────────────────────────────────
RAW_FEATURE_NAMES = [
    "ear_avg", "ear_left", "ear_right", "mar",
    "pitch", "yaw", "roll", "face_detected",
]
TEMPORAL_FEATURE_NAMES = [
    "ear_std", "mar_std", "pitch_std", "yaw_std",
    "blink_rate", "head_move_mag",
]
FEATURE_NAMES = RAW_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
N_RAW_FEATURES = len(RAW_FEATURE_NAMES)
N_FEATURES = len(FEATURE_NAMES)
MAX_FRAMES = 300


# ──────────────────────────────────────────────
# 영상 1개에서 프레임별 특징 추출
# ──────────────────────────────────────────────

def extract_features_from_video(video_path: str, face_mesh) -> np.ndarray:
    """
    단일 영상에서 원시 특징 + 시간적 특징을 추출합니다.

    Returns:
        features: (T, N_FEATURES) float32 배열
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"영상을 열 수 없습니다: {video_path}")
        return np.zeros((1, N_FEATURES), dtype=np.float32)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_features = []

    while cap.isOpened() and len(raw_features) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            ear_l = compute_ear(lm, LEFT_EYE_EAR)
            ear_r = compute_ear(lm, RIGHT_EYE_EAR)
            ear_avg = (ear_l + ear_r) / 2.0
            mar = compute_mar(lm)
            pitch, yaw, roll = compute_head_pose(lm, w, h)
            raw_features.append([ear_avg, ear_l, ear_r, mar, pitch, yaw, roll, 1.0])
        else:
            raw_features.append([0.0] * (N_RAW_FEATURES - 1) + [0.0])

    cap.release()

    if len(raw_features) == 0:
        return np.zeros((1, N_FEATURES), dtype=np.float32)

    raw_arr = np.array(raw_features, dtype=np.float32)  # (T, 8)
    temporal_arr = compute_temporal_features(raw_arr)     # (T, 6)

    return np.concatenate([raw_arr, temporal_arr], axis=1)  # (T, 14)


# ──────────────────────────────────────────────
# 다중 조건 레이블 이진화
# ──────────────────────────────────────────────

def compute_label(row) -> int:
    """
    DAiSEE 4가지 감정 레이블을 종합하여 이진 집중도 레이블을 산출합니다.

    집중(1):  Engagement >= 2 AND 부정적 감정(Boredom, Confusion, Frustration) 모두 <= 1
    비집중(0): Engagement <= 1 OR 부정적 감정 중 하나라도 >= 2

    이 전략은 단순히 Engagement만 보는 것보다 '비집중' 클래스를 확대하여
    극심한 클래스 불균형(95% vs 5%)을 완화합니다.
    """
    eng = int(row["Engagement"])
    bor = int(row["Boredom"])
    con = int(row["Confusion"])
    fru = int(row["Frustration"])

    if eng <= 1:
        return 0
    if bor >= 2 or con >= 2 or fru >= 2:
        return 0
    return 1


# ──────────────────────────────────────────────
# DAiSEE split 전체 처리
# ──────────────────────────────────────────────

def build_file_index(split_dir: str) -> dict:
    """split 폴더 내 모든 .avi 파일을 {파일명: 전체경로}로 인덱싱"""
    index = {}
    log.info(f"파일 인덱스 구축 중: {split_dir}")
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.endswith(".avi"):
                index[f] = os.path.join(root, f)
    log.info(f"  → {len(index)}개 .avi 파일 발견")
    return index


def process_split(split: str, labels_df: pd.DataFrame):
    """
    한 split(Train/Validation/Test)을 처리합니다.

    Returns:
        X: list of (T, N_FEATURES) arrays
        y: list of int (0 or 1)
        clip_ids: list of str
    """
    split_dir = os.path.join(DATASET_DIR, split)
    if not os.path.isdir(split_dir):
        log.error(f"폴더가 없습니다: {split_dir}")
        return [], [], []

    file_index = build_file_index(split_dir)

    X, y, clip_ids = [], [], []
    id_col = "ClipID" if "ClipID" in labels_df.columns else labels_df.columns[0]

    failed = 0
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"[{split}]"):
            clip_name = str(row[id_col])
            label = compute_label(row)

            video_path = file_index.get(clip_name)
            if video_path is None:
                video_path = file_index.get(os.path.basename(clip_name))
            if video_path is None:
                failed += 1
                continue

            feat = extract_features_from_video(video_path, face_mesh)
            X.append(feat)
            y.append(label)
            clip_ids.append(clip_name)

    log.info(f"[{split}] 완료: {len(X)}개 추출, {failed}개 파일 없음")

    # 클래스 분포 출력
    y_arr = np.array(y)
    n0 = (y_arr == 0).sum()
    n1 = (y_arr == 1).sum()
    log.info(f"[{split}] 클래스 분포 → 비집중(0): {n0} ({n0/max(len(y),1)*100:.1f}%)  "
             f"집중(1): {n1} ({n1/max(len(y),1)*100:.1f}%)")

    return X, y, clip_ids


def main():
    log.info("=== Step 1: DAiSEE 특징 추출 시작 ===")
    log.info(f"DAiSEE 경로: {DAISEE_ROOT}")
    log.info(f"출력 경로  : {OUTPUT_DIR}")
    log.info(f"특징 수    : {N_FEATURES}개 (원시 {N_RAW_FEATURES} + 시간적 {len(TEMPORAL_FEATURE_NAMES)})")

    if not os.path.isdir(DAISEE_ROOT):
        log.error(
            f"\n❌ DAiSEE 경로를 찾을 수 없습니다: {DAISEE_ROOT}\n"
            "   step1_extract_features.py 상단의 DAISEE_ROOT 경로를 확인하세요."
        )
        return

    split_map = {
        "Train":      "TrainLabels.csv",
        "Validation": "ValidationLabels.csv",
        "Test":       "TestLabels.csv",
    }

    for split, label_file in split_map.items():
        label_path = os.path.join(LABELS_DIR, label_file)
        if not os.path.isfile(label_path):
            log.warning(f"레이블 파일 없음, 건너뜀: {label_path}")
            continue

        labels_df = pd.read_csv(label_path)
        labels_df.columns = labels_df.columns.str.strip()
        log.info(f"[{split}] 레이블 로드: {len(labels_df)}개 클립")

        X, y, clip_ids = process_split(split, labels_df)
        if len(X) == 0:
            log.warning(f"[{split}] 추출된 데이터 없음, 건너뜀")
            continue

        # 저장 (가변 길이 시퀀스 → object array)
        out_path = os.path.join(OUTPUT_DIR, f"{split.lower()}_features.npz")
        X_arr = np.empty(len(X), dtype=object)
        for i, seq in enumerate(X):
            X_arr[i] = seq

        np.savez(
            out_path,
            X=X_arr,
            y=np.array(y, dtype=np.int32),
            clip_ids=np.array(clip_ids),
            feature_names=np.array(FEATURE_NAMES),
        )
        log.info(f"[{split}] 저장 완료 → {out_path}")

    log.info("=== Step 1 완료! ===")
    log.info("다음 단계: python src/step2_prepare_dataset.py")


if __name__ == "__main__":
    main()
