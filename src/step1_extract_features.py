"""
Step 1: DAiSEE 영상에서 MediaPipe 기반 feature 추출

Features (프레임 단위):
  - EAR (Eye Aspect Ratio): left, right, avg
  - MAR (Mouth Aspect Ratio)
  - Head Pose: pitch, yaw, roll
  - Gaze Direction: iris 기반 좌우/상하 시선 오프셋
  - Face Detected: 얼굴 감지 여부 (0/1)

Labeling (Drowsiness-oriented):
  - Alert (1): Engagement >= 2 AND Boredom <= 1
  - Drowsy/Unfocused (0): Engagement <= 1 OR Boredom >= 2

Usage:
  python step1_extract_features.py
"""

import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# ── MediaPipe 설정 ──────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# EAR 랜드마크 인덱스 (MediaPipe FaceMesh 468점 기준)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# MAR 랜드마크 인덱스
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

# Iris 랜드마크 (MediaPipe FaceMesh refine_landmarks=True)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Head pose 추정용 3D 모델 포인트
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # 코끝 (1)
    (0.0, -330.0, -65.0),     # 턱 (152)
    (-225.0, 170.0, -135.0),  # 왼쪽 눈 끝 (263)
    (225.0, 170.0, -135.0),   # 오른쪽 눈 끝 (33)
    (-150.0, -150.0, -125.0), # 왼쪽 입 끝 (287)
    (150.0, -150.0, -125.0),  # 오른쪽 입 끝 (57)
], dtype=np.float64)

FACE_2D_IDX = [1, 152, 263, 33, 287, 57]

# ── Feature 계산 함수들 ─────────────────────────────────────────

def compute_ear(landmarks, eye_indices):
    """Eye Aspect Ratio 계산"""
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    # 수직 거리 2개
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # 수평 거리 1개
    h = np.linalg.norm(pts[0] - pts[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_mar(landmarks):
    """Mouth Aspect Ratio 계산"""
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in MOUTH])
    # 수직 거리 3개 (입의 상하 열림)
    v1 = np.linalg.norm(pts[2] - pts[6])  # 39-269
    v2 = np.linalg.norm(pts[3] - pts[7])  # 181-405
    v3 = np.linalg.norm(pts[4] - pts[5])  # 0-17
    # 수평 거리 1개
    h = np.linalg.norm(pts[0] - pts[1])   # 61-291
    if h < 1e-6:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h)


def compute_head_pose(landmarks, img_w, img_h):
    """Head pose (pitch, yaw, roll) 추정 via solvePnP"""
    face_2d = np.array([
        (landmarks[i].x * img_w, landmarks[i].y * img_h)
        for i in FACE_2D_IDX
    ], dtype=np.float64)

    focal_length = img_w
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, face_2d, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0]  # 상하
    yaw = angles[1]    # 좌우
    roll = angles[2]   # 기울임
    return pitch, yaw, roll


def compute_gaze_offset(landmarks, img_w, img_h):
    """
    Iris 중심과 눈 중심 사이의 오프셋으로 시선 방향 근사.
    반환: (horizontal_offset, vertical_offset)
    - 양수 = 오른쪽/아래, 음수 = 왼쪽/위
    - 눈 너비로 정규화 (-1 ~ 1 범위)
    """
    try:
        # 왼쪽 iris 중심
        l_iris = np.mean([(landmarks[i].x, landmarks[i].y) for i in LEFT_IRIS], axis=0)
        # 오른쪽 iris 중심
        r_iris = np.mean([(landmarks[i].x, landmarks[i].y) for i in RIGHT_IRIS], axis=0)

        # 왼쪽 눈 양 끝
        l_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        l_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        l_eye_center = (l_eye_inner + l_eye_outer) / 2
        l_eye_width = np.linalg.norm(l_eye_inner - l_eye_outer)

        # 오른쪽 눈 양 끝
        r_eye_inner = np.array([landmarks[33].x, landmarks[33].y])
        r_eye_outer = np.array([landmarks[133].x, landmarks[133].y])
        r_eye_center = (r_eye_inner + r_eye_outer) / 2
        r_eye_width = np.linalg.norm(r_eye_inner - r_eye_outer)

        if l_eye_width < 1e-6 or r_eye_width < 1e-6:
            return 0.0, 0.0

        # 각 눈별 오프셋 계산 후 평균
        l_offset = (l_iris - l_eye_center) / l_eye_width
        r_offset = (r_iris - r_eye_center) / r_eye_width

        avg_h = (l_offset[0] + r_offset[0]) / 2  # 좌우
        avg_v = (l_offset[1] + r_offset[1]) / 2  # 상하

        return float(avg_h), float(avg_v)
    except (IndexError, ValueError):
        return 0.0, 0.0


# ── 라벨 로드 ───────────────────────────────────────────────────

def load_labels(label_csv):
    """
    DAiSEE 라벨 CSV → {clip_id: label} 딕셔너리
    Drowsiness-oriented labeling:
      Alert (1): Engagement >= 2 AND Boredom <= 1
      Drowsy (0): Engagement <= 1 OR Boredom >= 2
    """
    labels = {}
    with open(label_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row['ClipID'].replace('.avi', '')
            engagement = int(row['Engagement'])
            boredom = int(row['Boredom'])

            if engagement <= 1 or boredom >= 2:
                labels[clip_id] = 0  # Drowsy/Unfocused
            else:
                labels[clip_id] = 1  # Alert/Focused
    return labels


# ── 영상에서 feature 추출 ───────────────────────────────────────

def extract_video_features(video_path, max_frames=300):
    """
    한 영상에서 프레임별 feature 벡터 추출.
    반환: np.ndarray (num_frames, 10)
      [ear_left, ear_right, ear_avg, mar, pitch, yaw, roll,
       gaze_h, gaze_v, face_detected]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    features = []
    frame_count = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # iris 랜드마크 활성화
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                ear_l = compute_ear(lm, LEFT_EYE)
                ear_r = compute_ear(lm, RIGHT_EYE)
                ear_avg = (ear_l + ear_r) / 2
                mar = compute_mar(lm)
                pitch, yaw, roll = compute_head_pose(lm, img_w, img_h)
                gaze_h, gaze_v = compute_gaze_offset(lm, img_w, img_h)

                features.append([
                    ear_l, ear_r, ear_avg, mar,
                    pitch, yaw, roll,
                    gaze_h, gaze_v,
                    1.0  # face_detected
                ])
            else:
                # 얼굴 미감지 → 0 벡터
                features.append([0.0] * 9 + [0.0])

    cap.release()

    if len(features) == 0:
        return None

    return np.array(features, dtype=np.float32)


# ── 메인 실행 ───────────────────────────────────────────────────

def process_split(split_name, data_root, label_csv, output_path):
    """한 split (Train/Validation/Test) 전체 처리"""
    labels = load_labels(label_csv)
    split_dir = data_root / "DataSet" / split_name

    if not split_dir.exists():
        print(f"[WARN] {split_dir} does not exist, skipping.")
        return

    all_features = []
    all_labels = []
    all_clip_ids = []
    skipped = 0

    # 모든 클립 디렉토리 수집
    clip_dirs = []
    for user_dir in sorted(split_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        for clip_dir in sorted(user_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            clip_dirs.append(clip_dir)

    print(f"\n[{split_name}] Processing {len(clip_dirs)} clips...")

    for clip_dir in tqdm(clip_dirs, desc=split_name):
        clip_id = clip_dir.name

        # 라벨 확인
        if clip_id not in labels:
            skipped += 1
            continue

        # 영상 파일 찾기
        video_file = clip_dir / f"{clip_id}.avi"
        if not video_file.exists():
            # 다른 확장자 시도
            for ext in ['.mp4', '.avi', '.mov']:
                alt = clip_dir / f"{clip_id}{ext}"
                if alt.exists():
                    video_file = alt
                    break
            else:
                skipped += 1
                continue

        # feature 추출
        feats = extract_video_features(video_file)
        if feats is None or len(feats) < 10:
            skipped += 1
            continue

        all_features.append(feats)
        all_labels.append(labels[clip_id])
        all_clip_ids.append(clip_id)

    print(f"[{split_name}] Extracted: {len(all_features)}, Skipped: {skipped}")

    # 클래스 분포
    labels_arr = np.array(all_labels)
    n_alert = np.sum(labels_arr == 1)
    n_drowsy = np.sum(labels_arr == 0)
    print(f"[{split_name}] Alert: {n_alert}, Drowsy: {n_drowsy} "
          f"(ratio {n_drowsy/(n_alert+n_drowsy)*100:.1f}%)")

    # 저장
    np.savez_compressed(
        output_path,
        features=np.array(all_features, dtype=object),
        labels=labels_arr,
        clip_ids=np.array(all_clip_ids),
    )
    print(f"[{split_name}] Saved to {output_path}")


def main():
    base = Path(__file__).resolve().parent.parent
    data_root = base / "data" / "DAiSEE" / "DAiSEE"
    labels_dir = data_root / "Labels"
    output_dir = base / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "Train": labels_dir / "TrainLabels.csv",
        "Validation": labels_dir / "ValidationLabels.csv",
        "Test": labels_dir / "TestLabels.csv",
    }

    for split_name, label_csv in splits.items():
        output_path = output_dir / f"{split_name.lower()}_features.npz"
        process_split(split_name, data_root, label_csv, output_path)

    print("\n✓ Feature extraction complete!")


if __name__ == "__main__":
    main()
