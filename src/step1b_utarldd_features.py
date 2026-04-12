"""
Step 1b: UTA-RLDD 데이터셋에서 MediaPipe 기반 feature 추출

UTA-RLDD 구조:
  - 60명 피험자
  - 각 피험자당 3개 영상: 0(alert), 1(low vigilance), 2(drowsy)
  - 영상당 수 분 길이 → 3초 클립으로 분할해서 추출

라벨링:
  - Alert (1):  클래스 0 (alert)
  - Drowsy (0): 클래스 1 (low vigilance) + 클래스 2 (drowsy)
    → 실제 졸음 신호(EAR↓, MAR↑)가 확실히 있는 데이터

다운로드:
  https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset
  → data/UTA-RLDD/ 폴더에 압축 해제

Usage:
  python src/step1b_utarldd_features.py
  python src/step1b_utarldd_features.py --data-dir data/UTA-RLDD
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# step1의 함수 재사용
from step1_extract_features import (
    compute_ear, compute_mar, compute_head_pose, compute_gaze_offset,
    LEFT_EYE, RIGHT_EYE, mp_face_mesh
)

# 클립 길이 설정
CLIP_FPS = 30
CLIP_SEC = 3
CLIP_FRAMES = CLIP_FPS * CLIP_SEC  # 90 프레임


def detect_dataset_structure(data_dir: Path):
    """
    UTA-RLDD 폴더 구조 자동 감지.
    여러 Kaggle 업로드 버전의 구조가 다를 수 있어서 유연하게 처리.

    지원 구조:
      A) data_dir/0/, data_dir/1/, data_dir/2/   (클래스별 폴더)
      B) data_dir/alert/, data_dir/low_vigilance/, data_dir/drowsy/
      C) data_dir/subject_XX/0.avi, 1.avi, 2.avi  (피험자별 폴더)
      D) data_dir/Fold*/Fold*/subject_XX/0.mov, 5.mov, 10.mov  (실제 UTA-RLDD Kaggle 구조)
    """
    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".MOV", ".MP4", ".AVI"}

    # 구조 A: 숫자 폴더
    if (data_dir / "0").exists() or (data_dir / "0_Alert").exists():
        class_dirs = {}
        for label, names in [(1, ["0", "0_Alert", "alert", "Alert"]),
                              (0, ["1", "2", "1_Low_Vigilance", "2_Drowsy",
                                   "low_vigilance", "drowsy"])]:
            for name in names:
                d = data_dir / name
                if d.exists():
                    class_dirs.setdefault(label, []).append(d)
        if class_dirs:
            return "class_dirs", class_dirs

    # 구조 B: 영어 이름 폴더
    alert_names = ["alert", "Alert", "ALERT"]
    drowsy_names = ["drowsy", "Drowsy", "DROWSY", "low_vigilance",
                    "low vigilance", "sleepy"]
    found_alert = [data_dir / n for n in alert_names if (data_dir / n).exists()]
    found_drowsy = [data_dir / n for n in drowsy_names if (data_dir / n).exists()]
    if found_alert or found_drowsy:
        return "class_dirs", {1: found_alert, 0: found_drowsy}

    # 구조 D: Fold*/Fold*/subject/ 구조 (실제 Kaggle UTA-RLDD)
    # 영상 파일명이 숫자 (0=alert, 5=low vigilance, 10=drowsy)
    fold_dirs = [d for d in data_dir.iterdir()
                 if d.is_dir() and d.name.lower().startswith("fold")]
    if fold_dirs:
        # 재귀적으로 비디오 파일 검색하여 subject 폴더들 수집
        subject_dirs = set()
        for fd in fold_dirs:
            for vf in fd.rglob("*"):
                if vf.is_file() and vf.suffix in VIDEO_EXTS:
                    subject_dirs.add(vf.parent)
        if subject_dirs:
            return "subject_dirs", list(subject_dirs)

    # 구조 C: 피험자별 폴더 (subject_XX/0.avi, 1.avi, 2.avi)
    subject_dirs = [d for d in data_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('.')]
    if subject_dirs:
        sample = subject_dirs[0]
        video_files = [f for f in sample.iterdir()
                       if f.is_file() and f.suffix in VIDEO_EXTS]
        if video_files:
            return "subject_dirs", subject_dirs

    return "unknown", None


def collect_videos_class_dirs(class_dirs: dict):
    """클래스별 폴더 구조에서 (video_path, label) 수집"""
    videos = []
    for label, dirs in class_dirs.items():
        for d in dirs:
            for ext in ["*.avi", "*.mp4", "*.mov", "*.MP4", "*.AVI"]:
                for vf in d.rglob(ext):
                    videos.append((vf, label))
    return videos


def collect_videos_subject_dirs(subject_dirs: list):
    """
    피험자별 폴더 구조에서 (video_path, label) 수집.
    파일명의 숫자로 클래스 판단 (UTA-RLDD 실제 파일명: 0, 5, 10):
      0       → alert         (label=1)
      5, 10   → low vigilance / drowsy (label=0)
      1, 2    → low vigilance / drowsy (label=0) — 구버전 호환
    """
    VIDEO_EXTS = ["*.avi", "*.mp4", "*.mov", "*.MP4", "*.AVI", "*.MOV"]
    videos = []
    seen = set()  # 중복 방지

    for subject_dir in subject_dirs:
        for ext in VIDEO_EXTS:
            for vf in subject_dir.glob(ext):
                if vf in seen:
                    continue
                seen.add(vf)

                stem = vf.stem  # e.g. "0", "5", "10"
                try:
                    class_num = int(stem)
                    # 0 = alert → label 1 (집중)
                    # 5, 10 = low vigilance / drowsy → label 0
                    # 1, 2 = 구버전 UTA-RLDD → label 0
                    label = 1 if class_num == 0 else 0
                    videos.append((vf, label))
                except ValueError:
                    # 파일명이 숫자가 아닌 경우 → 폴더명으로 판단
                    parent = vf.parent.name.lower()
                    if any(k in parent for k in ["alert"]):
                        videos.append((vf, 1))
                    elif any(k in parent for k in ["drowsy", "vigilance", "sleepy"]):
                        videos.append((vf, 0))
    return videos


def extract_clips_from_video(video_path, label, clip_frames=CLIP_FRAMES, frame_skip=2):
    """
    긴 영상을 clip_frames 길이의 클립으로 분할해서 feature 추출.
    frame_skip: N프레임마다 1개만 처리 (기본값 2 → 처리 속도 2배)
    반환: list of (feature_array, label)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    clips = []
    current_clip = []
    frame_idx = 0
    # 클립당 실제로 수집할 프레임 수 (skip 적용)
    target_frames = max(1, clip_frames // frame_skip)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            # frame_skip마다 1프레임만 처리
            if frame_idx % frame_skip != 0:
                continue

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

                feat = [ear_l, ear_r, ear_avg, mar,
                        pitch, yaw, roll, gaze_h, gaze_v, 1.0]
            else:
                feat = [0.0] * 9 + [0.0]

            current_clip.append(feat)

            # 클립 완성
            if len(current_clip) >= target_frames:
                arr = np.array(current_clip[:target_frames], dtype=np.float32)
                clips.append((arr, label))
                current_clip = []

    cap.release()
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/UTA-RLDD",
                        help="UTA-RLDD 데이터 폴더 경로")
    parser.add_argument("--clip-sec", type=int, default=3,
                        help="클립 길이 (초)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = base / args.data_dir
    output_path = base / "data" / "features" / "utarldd_features.npz"

    if not data_dir.exists():
        print(f"[ERROR] {data_dir} 폴더가 없어.")
        print("  Kaggle에서 UTA-RLDD를 다운받아서 해당 폴더에 압축 해제해줘:")
        print("  https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset")
        return

    # 데이터셋 구조 감지
    print(f"[Info] 데이터 폴더: {data_dir}")
    structure, info = detect_dataset_structure(data_dir)
    print(f"[Info] 감지된 구조: {structure}")

    if structure == "class_dirs":
        videos = collect_videos_class_dirs(info)
    elif structure == "subject_dirs":
        videos = collect_videos_subject_dirs(info)
    else:
        # 폴더 구조 출력 후 종료
        print("[ERROR] 폴더 구조를 인식 못했어. 실제 폴더 구조를 알려줘:")
        for item in sorted(data_dir.iterdir())[:20]:
            print(f"  {item}")
        return

    if not videos:
        print("[ERROR] 영상 파일을 찾지 못했어. 폴더 구조를 확인해줘.")
        return

    n_alert = sum(1 for _, l in videos if l == 1)
    n_drowsy = sum(1 for _, l in videos if l == 0)
    print(f"[Info] 영상 수: {len(videos)} (Alert={n_alert}, Drowsy={n_drowsy})")

    # feature 추출
    all_features = []
    all_labels = []
    skipped = 0

    global_clip_frames = CLIP_FPS * args.clip_sec

    for video_path, label in tqdm(videos, desc="Extracting"):
        try:
            clips = extract_clips_from_video(video_path, label,
                                             clip_frames=global_clip_frames)
            if not clips:
                skipped += 1
                continue
            for feat_arr, lbl in clips:
                all_features.append(feat_arr)
                all_labels.append(lbl)
        except Exception as e:
            print(f"[WARN] {video_path.name}: {e}")
            skipped += 1
            continue

    print(f"\n[Result] 총 클립: {len(all_features)}, 스킵: {skipped}")
    labels_arr = np.array(all_labels)
    n_alert_clips = np.sum(labels_arr == 1)
    n_drowsy_clips = np.sum(labels_arr == 0)
    print(f"[Result] Alert: {n_alert_clips}, Drowsy: {n_drowsy_clips} "
          f"(Drowsy {n_drowsy_clips/(len(labels_arr))*100:.1f}%)")

    if len(all_features) == 0:
        print("[ERROR] 추출된 클립이 없어.")
        return

    np.savez_compressed(
        output_path,
        features=np.array(all_features, dtype=object),
        labels=labels_arr,
        clip_ids=np.array([f"utarldd_{i}" for i in range(len(all_features))]),
    )
    print(f"[Save] {output_path}")
    print("\n다음 단계: python src/step2_prepare_dataset.py")


if __name__ == "__main__":
    main()
