"""
실사용 데이터 수집 스크립트 (타이머 자동 모드)
------------------------------------------------------
스크립트가 타이머에 따라 자동으로 집중/비집중 구간을 안내합니다.
키 없이 화면 지시만 따라가면 됩니다.

진행 순서:
    [준비] 5초 카운트다운
    [집중] 10분 — 실제 작업하세요
    [전환] 1분  — 딴짓 준비
    [비집중] 5분 — 의도적으로 딴짓 (폰, 멍, 고개 돌리기)
    [전환] 1분  — 복귀 준비
    → 위 집중/전환/비집중 사이클 2회 반복

조작법:
    Q / ESC → 중간에 저장 후 종료 (수집된 데이터까지 저장됨)

출력:
    data/realdata/session_YYYYMMDD_HHMMSS.npz

실행 방법:
    python src/collect_realdata.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "realdata")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 타이머 세션 구성 (초 단위)
# ──────────────────────────────────────────────
SESSIONS = [
    ("ready",      5,  None, "잠깐! 준비해주세요",                  (180, 180, 180)),
    ("focus",    600,  1,    "집중하세요 — 평소처럼 작업하면 됩니다", (80, 200, 80)),
    ("transit",   60,  None, "잠깐 쉬세요 — 곧 딴짓 구간입니다",    (200, 200, 80)),
    ("unfocus",  300,  0,    "딴짓하세요 — 폰 보기, 멍 때리기, 고개 돌리기", (80, 80, 240)),
    ("transit",   60,  None, "다시 집중 준비! 화면 앞으로 돌아오세요", (200, 200, 80)),
    ("focus",    600,  1,    "집중하세요 — 평소처럼 작업하면 됩니다", (80, 200, 80)),
    ("transit",   60,  None, "잠깐 쉬세요 — 곧 딴짓 구간입니다",    (200, 200, 80)),
    ("unfocus",  300,  0,    "딴짓하세요 — 폰 보기, 멍 때리기, 고개 돌리기", (80, 80, 240)),
    ("done",       0,  None, "수집 완료! 잠시 후 저장됩니다.",       (100, 220, 100)),
]
# label: None이면 해당 구간 데이터 저장 안 함 (전환/준비 구간)

# ──────────────────────────────────────────────
# 특징 설정 (step1과 동일)
# ──────────────────────────────────────────────
N_RAW_FEATURES  = 8   # ear_avg, ear_l, ear_r, mar, pitch, yaw, roll, face_detected
N_FEATURES      = 14  # 8 raw + 6 temporal
TEMPORAL_WINDOW = 30
EAR_BLINK_THRESHOLD = 0.2

LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]
MOUTH_MAR     = [61, 291, 0, 17]
NOSE_TIP, CHIN           = 1, 152
LEFT_EYE_C, RIGHT_EYE_C  = 263, 33
LEFT_MOUTH, RIGHT_MOUTH  = 287, 57

# ──────────────────────────────────────────────
# 한글 폰트
# ──────────────────────────────────────────────
FONT_PATHS = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
]
_font_path = next((p for p in FONT_PATHS if os.path.exists(p)), None)

def get_font(size=20):
    if _font_path:
        return ImageFont.truetype(_font_path, size)
    return ImageFont.load_default()

def put_kr_text(frame, text, pos, font_size=20, color=(255,255,255), bg=None):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = get_font(font_size)
    x, y   = pos
    if bg is not None:
        bbox = font.getbbox(text)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle([x-4, y-4, x+tw+4, y+th+4], fill=bg)
    draw.text((x, y), text, font=font, fill=color)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ──────────────────────────────────────────────
# 특징 계산 (step1과 동일)
# ──────────────────────────────────────────────

def compute_ear(landmarks, indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def compute_mar(landmarks):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in MOUTH_MAR])
    return np.linalg.norm(pts[2]-pts[3]) / (np.linalg.norm(pts[0]-pts[1]) + 1e-6)

def compute_head_pose(landmarks, img_w, img_h):
    face_3d = np.array([
        [0.,0.,0.], [0.,-330.,-65.],
        [-225.,170.,-135.], [225.,170.,-135.],
        [-150.,-150.,-125.], [150.,-150.,-125.],
    ], dtype=np.float64)
    key_pts = [NOSE_TIP, CHIN, LEFT_EYE_C, RIGHT_EYE_C, LEFT_MOUTH, RIGHT_MOUTH]
    face_2d = np.array(
        [[landmarks[i].x*img_w, landmarks[i].y*img_h] for i in key_pts],
        dtype=np.float64
    )
    focal = img_w
    cam_matrix = np.array([[focal,0,img_w/2],[0,focal,img_h/2],[0,0,1]], dtype=np.float64)
    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4,1)))
    if not success:
        return 0., 0., 0.
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rot_mat)
    pitch = float(np.clip(angles[0]*360, -90., 90.))
    yaw   = float(np.clip(angles[1]*360, -90., 90.))
    roll  = float(np.clip(angles[2]*360, -90., 90.))
    return pitch, yaw, roll

def compute_temporal(raw_buf: deque, mar_buf: deque) -> np.ndarray:
    buf   = np.array(list(raw_buf), dtype=np.float32)
    mar_b = np.array(list(mar_buf), dtype=np.float32)
    start = max(0, len(buf) - TEMPORAL_WINDOW)
    w_raw = buf[start:]
    w_mar = mar_b[start:]
    ear   = w_raw[:, 0]
    pitch = w_raw[:, 4]   # idx: ear_avg=0, ear_l=1, ear_r=2, mar=3, pitch=4
    yaw   = w_raw[:, 5]   # idx: yaw=5
    t = np.zeros(6, dtype=np.float32)
    t[0] = np.std(ear)
    t[1] = np.std(w_mar)
    t[2] = np.std(pitch)
    t[3] = np.std(yaw)
    t[4] = np.sum(ear < EAR_BLINK_THRESHOLD) / max(len(ear), 1)
    if len(pitch) > 1:
        t[5] = np.mean(np.abs(np.diff(pitch))) + np.mean(np.abs(np.diff(yaw)))
    return t

# ──────────────────────────────────────────────
# 진행률 바
# ──────────────────────────────────────────────

def draw_progress_bar(frame, elapsed, total, x, y, w=400, h=14, color=(100,200,100)):
    ratio = min(elapsed / max(total, 1), 1.0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,50), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*ratio), y+h), color, -1)

# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    total_focus_sec   = sum(s[1] for s in SESSIONS if s[2] == 1)
    total_unfocus_sec = sum(s[1] for s in SESSIONS if s[2] == 0)
    total_sec = sum(s[1] for s in SESSIONS)
    print("=== 실사용 데이터 수집 (타이머 자동 모드) ===")
    print(f"  총 소요 시간: 약 {total_sec//60}분")
    print(f"  집중 구간  : {total_focus_sec//60}분")
    print(f"  비집중 구간: {total_unfocus_sec//60}분")
    print("  Q/ESC: 중간 종료 (수집된 데이터까지 저장)\n")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    raw_buf = deque(maxlen=TEMPORAL_WINDOW * 2)
    mar_buf = deque(maxlen=TEMPORAL_WINDOW * 2)

    collected_X   = []
    collected_y   = []
    label_counts  = {0: 0, 1: 0}

    session_idx   = 0
    session_start = time.time()
    early_quit    = False

    print(f"[{SESSIONS[0][3]}]")

    while session_idx < len(SESSIONS):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        now  = time.time()

        # ── 세션 전환 체크 ──────────────────────
        kind, duration, label, message, color = SESSIONS[session_idx]
        elapsed_in_session = now - session_start

        if kind == "done" or (duration > 0 and elapsed_in_session >= duration):
            session_idx  += 1
            session_start = now
            if session_idx < len(SESSIONS):
                kind, duration, label, message, color = SESSIONS[session_idx]
                print(f"[{message}]")
            continue

        # ── 특징 추출 ──────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        face_detected = False
        raw_feat = np.zeros(N_RAW_FEATURES, dtype=np.float32)
        mar_val  = 0.0

        if result.multi_face_landmarks:
            face_detected = True
            lm = result.multi_face_landmarks[0].landmark
            ear_l = compute_ear(lm, LEFT_EYE_EAR)
            ear_r = compute_ear(lm, RIGHT_EYE_EAR)
            mar_val = compute_mar(lm)
            pitch, yaw, roll = compute_head_pose(lm, w, h)
            raw_feat = np.array(
                [(ear_l+ear_r)/2, ear_l, ear_r, mar_val, pitch, yaw, roll, 1.0],
                dtype=np.float32
            )  # 8-dim: ear_avg, ear_l, ear_r, mar, pitch, yaw, roll, face_detected
            for idx in LEFT_EYE_EAR + RIGHT_EYE_EAR:
                cx, cy = int(lm[idx].x*w), int(lm[idx].y*h)
                cv2.circle(frame, (cx,cy), 2, (0,255,180), -1)

        raw_buf.append(raw_feat)
        mar_buf.append(mar_val)

        # 레이블이 있는 구간만 저장
        if label is not None and face_detected and len(raw_buf) >= 5:
            temporal  = compute_temporal(raw_buf, mar_buf)
            full_feat = np.concatenate([raw_feat, temporal])
            collected_X.append(full_feat)
            collected_y.append(label)
            label_counts[label] += 1

        # ── UI ─────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,110), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # 현재 구간 메시지
        put_kr_text(frame, message, (15, 8), font_size=22, color=color, bg=(20,20,20))

        # 남은 시간
        remain = max(0, duration - elapsed_in_session)
        remain_str = f"{int(remain//60)}분 {int(remain%60):02d}초 남음"
        put_kr_text(frame, remain_str, (15, 45), font_size=20,
                    color=(220,220,220), bg=(20,20,20))

        # 진행률 바
        draw_progress_bar(frame, elapsed_in_session, duration,
                          x=15, y=80, color=color)

        # 전체 진행률 (우상단)
        total_collected = len(collected_y)
        info = f"집중: {label_counts[1]}f  비집중: {label_counts[0]}f"
        put_kr_text(frame, info, (w-280, 8), font_size=16,
                    color=(180,180,180), bg=(20,20,20))

        # 세션 번호 (전체 중 몇 번째)
        step_info = f"구간 {session_idx+1}/{len(SESSIONS)}"
        put_kr_text(frame, step_info, (w-130, 35), font_size=16,
                    color=(150,150,150), bg=(20,20,20))

        # 얼굴 미검출 경고
        if not face_detected:
            put_kr_text(frame, "얼굴을 카메라에 맞춰주세요",
                        (w//2-140, h//2), font_size=20,
                        color=(80,80,240), bg=(20,20,20))

        # 전환 구간 안내 (label=None)
        if label is None and kind not in ("ready", "done"):
            put_kr_text(frame, "이 구간은 저장되지 않습니다",
                        (15, h-30), font_size=15,
                        color=(130,130,130), bg=(20,20,20))

        put_kr_text(frame, "Q: 중간 종료 후 저장",
                    (15, h-30), font_size=15,
                    color=(120,120,120), bg=(20,20,20))

        cv2.imshow("Smartto - 데이터 수집", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            early_quit = True
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

    # ── 저장 ──────────────────────────────────
    if len(collected_y) == 0:
        print("수집된 데이터가 없습니다.")
        return

    X_arr = np.array(collected_X, dtype=np.float32)
    y_arr = np.array(collected_y, dtype=np.int32)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"session_{ts}.npz")
    np.savez(out_path, X=X_arr, y=y_arr)

    print(f"\n=== 수집 {'중단 후 ' if early_quit else ''}완료 ===")
    print(f"  저장 경로 : {out_path}")
    print(f"  총 프레임 : {len(y_arr)}")
    print(f"  집중(1)   : {(y_arr==1).sum()} ({(y_arr==1).mean()*100:.1f}%)")
    print(f"  비집중(0) : {(y_arr==0).sum()} ({(y_arr==0).mean()*100:.1f}%)")
    print(f"\n다음 단계: python src/finetune.py")


if __name__ == "__main__":
    main()
