"""
Step 4: 실시간 카메라 집중도 추론
------------------------------------------------------
학습된 LSTM 모델로 웹캠 영상을 분석해
실시간으로 집중도를 판단하고 세션/휴식을 추천합니다.

실행 방법:
    python src/step4_realtime.py

조작법:
    Q 또는 ESC  → 종료
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import time
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ──────────────────────────────────────────────
# 한글 폰트 설정 (macOS 시스템 폰트 사용)
# ──────────────────────────────────────────────
FONT_PATHS = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",       # macOS 기본 한글 폰트
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
]
_font_path = next((p for p in FONT_PATHS if os.path.exists(p)), None)

def get_font(size=22):
    if _font_path:
        return ImageFont.truetype(_font_path, size)
    return ImageFont.load_default()

def put_kr_text(frame, text, pos, font_size=22, color=(255,255,255), bg=None):
    """OpenCV 프레임에 한글 텍스트를 렌더링합니다."""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = get_font(font_size)

    x, y = pos
    # 배경 박스
    if bg is not None:
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x-4, y-4, x+tw+4, y+th+4], fill=bg)

    draw.text((x, y), text, font=font, fill=color)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "data", "features", "scaler.pkl")

# ──────────────────────────────────────────────
# 모델 설정 (step3_train.py와 동일하게)
# ──────────────────────────────────────────────
SEQ_LEN     = 150
N_FEATURES  = 8
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
N_CLASSES   = 2

# 집중도 판정 기준 (연속 몇 초 동안의 평균으로 판단)
DECISION_WINDOW = 30   # 최근 30번의 판정을 평균

# 세션/휴식 추천 기준 (초)
SESSION_FOCUS_THRESHOLD   = 0.6   # 집중도 60% 이상이면 "집중 중"
RECOMMEND_BREAK_AFTER     = 25 * 60   # 25분 집중 → 휴식 추천
RECOMMEND_SESSION_AFTER   = 5  * 60   # 5분 휴식 → 세션 추천

# ──────────────────────────────────────────────
# MediaPipe 랜드마크 인덱스
# ──────────────────────────────────────────────
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]
MOUTH_MAR     = [61, 291, 0, 17]
NOSE_TIP, CHIN, LEFT_EYE_C, RIGHT_EYE_C, LEFT_MOUTH, RIGHT_MOUTH = 1, 152, 263, 33, 287, 57

# ──────────────────────────────────────────────
# 특징 계산
# ──────────────────────────────────────────────

def compute_ear(landmarks, indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_mar(landmarks):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in MOUTH_MAR])
    return np.linalg.norm(pts[2] - pts[3]) / (np.linalg.norm(pts[0] - pts[1]) + 1e-6)


def compute_head_pose(landmarks, img_w, img_h):
    face_3d = np.array([
        [0.0, 0.0, 0.0], [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0],
    ], dtype=np.float64)
    key_pts = [NOSE_TIP, CHIN, LEFT_EYE_C, RIGHT_EYE_C, LEFT_MOUTH, RIGHT_MOUTH]
    face_2d = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in key_pts],
        dtype=np.float64
    )
    focal = img_w
    cam_matrix = np.array([[focal, 0, img_w/2], [0, focal, img_h/2], [0, 0, 1]], dtype=np.float64)
    success, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4,1)))
    if not success:
        return 0.0, 0.0, 0.0
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rot_mat)
    return angles[0]*360, angles[1]*360, angles[2]*360

# ──────────────────────────────────────────────
# LSTM 모델 정의
# ──────────────────────────────────────────────

class ConcentrationLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=DROPOUT if NUM_LAYERS > 1 else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])

# ──────────────────────────────────────────────
# 세션 추천 로직
# ──────────────────────────────────────────────

FACE_ABSENT_THRESHOLD = 30   # 얼굴이 몇 초 이상 없으면 휴식으로 자동 전환

class SessionAdvisor:
    def __init__(self):
        self.state             = "idle"
        self.focus_start       = None
        self.focus_accumulated = 0.0
        self.rest_start        = None
        self.absent_start      = None
        self.recommendation    = ""

    def update(self, is_focused: bool, face_detected: bool, current_time: float):

        # ── 얼굴 미검출 처리 ─────────────────────────
        if not face_detected:
            if self.absent_start is None:
                self.absent_start = current_time

            absent_sec = current_time - self.absent_start

            if absent_sec < FACE_ABSENT_THRESHOLD:
                # 잠깐 자리 비운 것 → 세션 유지
                self.recommendation = f"⏸ 잠시 자리를 비웠어요 ({int(absent_sec)}초) — 세션 유지 중"
                return self.recommendation
            else:
                # 30초 초과 → 휴식으로 자동 전환
                if self.state == "focusing" and self.focus_start:
                    self.focus_accumulated += current_time - self.focus_start
                    self.focus_start = None
                if self.state != "resting":
                    self.state      = "resting"
                    self.rest_start = current_time
                elapsed = current_time - (self.rest_start or current_time)
                self.recommendation = f"자리를 비운 지 {int(elapsed+FACE_ABSENT_THRESHOLD)}초 — 휴식으로 전환됨"
                return self.recommendation
        else:
            # 얼굴 다시 감지 → 부재 타이머 리셋
            self.absent_start = None

        # ── 집중 / 비집중 판정 ───────────────────────
        if is_focused:
            if self.state != "focusing":
                self.state       = "focusing"
                self.focus_start = current_time
                self.rest_start  = None

            elapsed = (current_time - self.focus_start) + self.focus_accumulated
            if elapsed >= RECOMMEND_BREAK_AFTER:
                self.recommendation = f"🛑 {int(elapsed//60)}분 집중! 잠깐 쉬세요 (5분 추천)"
            else:
                remain = RECOMMEND_BREAK_AFTER - elapsed
                self.recommendation = f"집중 중  |  휴식까지 {int(remain//60)}분 {int(remain%60)}초"
        else:
            if self.state == "focusing" and self.focus_start:
                self.focus_accumulated += current_time - self.focus_start
                self.focus_start = None
            if self.state != "resting":
                self.state      = "resting"
                self.rest_start = current_time

            elapsed = current_time - (self.rest_start or current_time)
            if elapsed >= RECOMMEND_SESSION_AFTER:
                self.focus_accumulated = 0.0
                self.recommendation = "▶️  충분히 쉬었어요! 다시 시작해 볼까요?"
            else:
                self.recommendation = f"비집중 / 휴식 중  ({int(elapsed)}초)"

        return self.recommendation

# ──────────────────────────────────────────────
# 화면 렌더링 유틸
# ──────────────────────────────────────────────

def draw_bar(frame, label, value, x, y, w=200, h=18, color=(0,200,100)):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (60,60,60), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*min(value,1.0)), y+h), color, -1)
    put_kr_text(frame, f"{label}: {value:.2f}", (x, y - 22), font_size=18, color=(220,220,220))

# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────

def main():
    # 디바이스
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 모델 & Scaler 로드
    model = ConcentrationLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print("모델 로드 완료! 카메라 시작 중...")

    # MediaPipe
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 버퍼
    frame_buffer  = deque(maxlen=SEQ_LEN)    # 특징 시퀀스
    pred_buffer   = deque(maxlen=DECISION_WINDOW)  # 최근 예측값
    advisor       = SessionAdvisor()

    pred_label    = "대기 중..."
    focus_ratio   = 0.0
    confidence    = 0.0

    print("실행 중... Q 또는 ESC로 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        face_detected = False
        if result.multi_face_landmarks:
            face_detected = True
            lm = result.multi_face_landmarks[0].landmark

            ear_l = compute_ear(lm, LEFT_EYE_EAR)
            ear_r = compute_ear(lm, RIGHT_EYE_EAR)
            mar   = compute_mar(lm)
            pitch, yaw, roll = compute_head_pose(lm, w, h)

            feat = np.array([(ear_l+ear_r)/2, ear_l, ear_r, mar, pitch, yaw, roll, 1.0],
                            dtype=np.float32)
            frame_buffer.append(feat)

            # 얼굴 랜드마크 그리기 (눈, 입 포인트만)
            for idx in LEFT_EYE_EAR + RIGHT_EYE_EAR + MOUTH_MAR:
                cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 180), -1)

        else:
            # 얼굴 미검출 시 0으로 채움
            frame_buffer.append(np.zeros(N_FEATURES, dtype=np.float32))

        # SEQ_LEN 프레임 쌓이면 추론
        if len(frame_buffer) == SEQ_LEN:
            seq = np.array(frame_buffer, dtype=np.float32)          # (150, 8)
            seq_scaled = scaler.transform(seq).astype(np.float32)   # 정규화
            x = torch.tensor(seq_scaled[np.newaxis], dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred       = int(probs.argmax())
            confidence = float(probs[1])   # 집중 확률
            pred_buffer.append(pred)

            focus_ratio = sum(pred_buffer) / len(pred_buffer)
            is_focused  = focus_ratio >= SESSION_FOCUS_THRESHOLD
            pred_label  = "집중 ✅" if is_focused else "비집중 ⚠️"
            advisor.update(is_focused, face_detected, time.time())

        # ── UI 렌더링 ──────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 상태 텍스트
        label_color = (80, 220, 80) if "집중" in pred_label and "비" not in pred_label else (80, 80, 240)
        put_kr_text(frame, pred_label, (15, 8), font_size=26, color=label_color, bg=(20,20,20))

        # 집중도 바
        draw_bar(frame, "집중도", focus_ratio, x=15, y=68,  color=(80,200,80))
        draw_bar(frame, "확률  ", confidence,  x=15, y=105, color=(80,150,220))

        # 세션 추천
        if advisor.recommendation:
            put_kr_text(frame, advisor.recommendation, (15, h-40),
                        font_size=18, color=(255,220,80), bg=(30,30,30))

        # 얼굴 미검출 경고
        if not face_detected:
            put_kr_text(frame, "얼굴을 카메라에 맞춰주세요",
                        (w//2-140, h//2), font_size=22, color=(80,80,240), bg=(20,20,20))

        # 버퍼 수집 진행률 (초반)
        if len(frame_buffer) < SEQ_LEN:
            ratio = len(frame_buffer) / SEQ_LEN
            put_kr_text(frame, f"초기화 중... {int(ratio*100)}%",
                        (15, h-70), font_size=18, color=(180,180,180), bg=(20,20,20))

        cv2.imshow("Smartto - 집중도 모니터", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q 또는 ESC
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("종료됨.")


if __name__ == "__main__":
    main()
