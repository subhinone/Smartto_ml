"""
Step 1: DAiSEE 데이터셋에서 MediaPipe로 특징 추출
------------------------------------------------------
EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio),
Head Pose, Blink 등을 각 영상 프레임에서 추출하여
numpy 배열로 저장합니다.

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
# 경로 설정 (본인 환경에 맞게 수정하세요)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAISEE_ROOT = os.path.join(BASE_DIR, "data", "DAiSEE", "DAiSEE")   # 프로젝트 내 경로
LABELS_DIR = os.path.join(DAISEE_ROOT, "Labels")
DATASET_DIR = os.path.join(DAISEE_ROOT, "DataSet")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# MediaPipe FaceMesh 랜드마크 인덱스 정의
# ──────────────────────────────────────────────
# EAR 계산용: [왼쪽끝, 위1, 위2, 오른쪽끝, 아래1, 아래2]
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]

# MAR 계산용: [왼쪽끝, 오른쪽끝, 위, 아래]
MOUTH_MAR = [61, 291, 0, 17]

# Head Pose 계산용 (3D 기준점)
NOSE_TIP     = 1
CHIN         = 152
LEFT_EYE_C   = 263
RIGHT_EYE_C  = 33
LEFT_MOUTH   = 287
RIGHT_MOUTH  = 57

# ──────────────────────────────────────────────
# 특징 계산 함수
# ──────────────────────────────────────────────

def compute_ear(landmarks, indices):
    """Eye Aspect Ratio: 눈 감김 정도 (낮을수록 졸림)"""
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])   # 수직1
    B = np.linalg.norm(pts[2] - pts[4])   # 수직2
    C = np.linalg.norm(pts[0] - pts[3])   # 수평
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
        [0.0,    0.0,    0.0  ],   # 코 끝
        [0.0,   -330.0, -65.0],   # 턱
        [-225.0, 170.0, -135.0],  # 왼쪽 눈
        [225.0,  170.0, -135.0],  # 오른쪽 눈
        [-150.0,-150.0, -125.0],  # 왼쪽 입
        [150.0, -150.0, -125.0],  # 오른쪽 입
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
# 영상 1개에서 프레임별 특징 추출
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    "ear_avg",    # 평균 EAR
    "ear_left",   # 왼쪽 EAR
    "ear_right",  # 오른쪽 EAR
    "mar",        # MAR
    "pitch",      # 고개 상하
    "yaw",        # 고개 좌우
    "roll",       # 고개 기울기
    "face_detected",  # 얼굴 검출 여부 (0/1)
]
N_FEATURES = len(FEATURE_NAMES)
MAX_FRAMES = 300   # 영상당 최대 프레임 수 (약 10초 @ 30fps)


def extract_features_from_video(video_path: str, face_mesh) -> np.ndarray:
    """
    단일 영상에서 프레임별 특징을 추출합니다.

    Returns:
        features: (T, N_FEATURES) float32 배열
                  T는 실제 프레임 수 (최대 MAX_FRAMES)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"영상을 열 수 없습니다: {video_path}")
        return np.zeros((1, N_FEATURES), dtype=np.float32)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    features = []

    while cap.isOpened() and len(features) < MAX_FRAMES:
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
            features.append([ear_avg, ear_l, ear_r, mar, pitch, yaw, roll, 1.0])
        else:
            # 얼굴 미검출 → 0으로 채움
            features.append([0.0] * (N_FEATURES - 1) + [0.0])

    cap.release()

    if len(features) == 0:
        return np.zeros((1, N_FEATURES), dtype=np.float32)

    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────
# DAiSEE split 전체 처리
# ──────────────────────────────────────────────

def build_file_index(split_dir: str) -> dict:
    """
    split 폴더 내 모든 .avi 파일을 재귀 탐색하여
    {파일명: 전체경로} 딕셔너리를 반환합니다.
    (CSV의 ClipID가 파일명만 있는 경우에 대응)
    """
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
    'Train' / 'Validation' / 'Test' 한 split을 처리합니다.

    Returns:
        X: list of (T, N_FEATURES) arrays
        y: list of int  (Engagement 레이블 0~3)
        clip_ids: list of str
    """
    split_dir = os.path.join(DATASET_DIR, split)
    if not os.path.isdir(split_dir):
        log.error(f"폴더가 없습니다: {split_dir}")
        return [], [], []

    # 파일명 → 전체경로 인덱스 (하위 폴더까지 탐색)
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
            clip_name = str(row[id_col])   # 예: "1100011002.avi"
            engagement = int(row["Engagement"])

            # 1) 파일명 직접 인덱스에서 찾기
            video_path = file_index.get(clip_name)

            # 2) 못 찾으면 basename으로 재시도 (경로 포함된 경우 대비)
            if video_path is None:
                video_path = file_index.get(os.path.basename(clip_name))

            if video_path is None:
                failed += 1
                continue

            feat = extract_features_from_video(video_path, face_mesh)
            X.append(feat)
            y.append(engagement)
            clip_ids.append(clip_name)

    log.info(f"[{split}] 완료: {len(X)}개 추출, {failed}개 파일 없음")
    return X, y, clip_ids


def main():
    log.info("=== Step 1: DAiSEE 특징 추출 시작 ===")
    log.info(f"DAiSEE 경로: {DAISEE_ROOT}")
    log.info(f"출력 경로  : {OUTPUT_DIR}")

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
        log.info(f"[{split}] 레이블 로드: {len(labels_df)}개 클립, 컬럼: {list(labels_df.columns)}")

        X, y, clip_ids = process_split(split, labels_df)
        if len(X) == 0:
            log.warning(f"[{split}] 추출된 데이터 없음, 건너뜀")
            continue

        # ── 저장 ──────────────────────────────────────────
        out_path = os.path.join(OUTPUT_DIR, f"{split.lower()}_features.npz")
        # X는 가변 길이 시퀀스라 object array로 저장
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
    log.info("다음 단계: src/step2_prepare_dataset.py 실행")


if __name__ == "__main__":
    main()
