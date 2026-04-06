"""
Step 4: 실시간 카메라 집중도 추론 + 적응형 뽀모도로 세션 관리
------------------------------------------------------
학습된 GRU 모델로 웹캠 영상을 분석해
실시간으로 집중도를 판단하고 다음 세션/휴식을 동적으로 추천합니다.

적응형 세션 로직:
    - 집중도가 높으면 → 다음 세션 +5분 연장, 휴식도 조금 더 부여
    - 집중도가 낮으면 → 다음 세션 단축, 짧은 휴식 후 재시작 유도

실행 방법:
    python src/step4_realtime.py

조작법:
    Q 또는 ESC → 종료
"""

import cv2
import mediapipe as mp_lib
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
import os
import time
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ──────────────────────────────────────────────
# 한글 폰트 설정 (macOS / Linux / Windows)
# ──────────────────────────────────────────────
FONT_PATHS = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
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
    if bg is not None:
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x-4, y-4, x+tw+4, y+th+4], fill=bg)

    draw.text((x, y), text, font=font, fill=color)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "best_model_finetuned.pt")
SCALER_PATH    = os.path.join(BASE_DIR, "data", "features", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.json")

# ──────────────────────────────────────────────
# 모델 설정 (step3_train.py와 일치)
# ──────────────────────────────────────────────
SEQ_LEN     = 90
N_FEATURES  = 14    # 원시 8 (ear_avg,ear_l,ear_r,mar,pitch,yaw,roll,face_det) + 시간적 6
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
DROPOUT     = 0.4
N_CLASSES   = 2

# 집중도 판정
DECISION_WINDOW = 30

# 적응형 세션 기본값 (초)
DEFAULT_SESSION_MINUTES = 25
DEFAULT_BREAK_MINUTES   = 5
MAX_SESSION_MINUTES     = 40
MIN_SESSION_MINUTES     = 15
MAX_BREAK_MINUTES       = 10
MIN_BREAK_MINUTES       = 3

# ──────────────────────────────────────────────
# MediaPipe 랜드마크 인덱스
# ──────────────────────────────────────────────
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]
MOUTH_MAR     = [61, 291, 0, 17]
NOSE_TIP, CHIN, LEFT_EYE_C, RIGHT_EYE_C, LEFT_MOUTH, RIGHT_MOUTH = 1, 152, 263, 33, 287, 57

# 시간적 특징 설정
TEMPORAL_WINDOW = 30
EAR_BLINK_THRESHOLD = 0.2

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
    pitch = float(np.clip(angles[0] * 360, -90.0, 90.0))
    yaw   = float(np.clip(angles[1] * 360, -90.0, 90.0))
    roll  = float(np.clip(angles[2] * 360, -90.0, 90.0))
    return pitch, yaw, roll


def compute_temporal_features_realtime(raw_buffer: deque, mar_buffer: deque,
                                        window: int = TEMPORAL_WINDOW) -> np.ndarray:
    """
    실시간 버퍼에서 현재 프레임의 시간적 통계 특징을 계산합니다.

    입력:
        raw_buffer — deque of (7,) arrays [ear_avg, ear_l, ear_r, pitch, yaw, roll, face_detected]
        mar_buffer — deque of float (MAR 값, mar_std 계산용)
    출력: (6,) array — [ear_std, mar_std, pitch_std, yaw_std, blink_rate, head_move_mag]
    """
    buf = list(raw_buffer)
    n = len(buf)
    start = max(0, n - window)
    window_data = np.array(buf[start:], dtype=np.float32)

    ear_avg = window_data[:, 0]
    mar     = np.array(list(mar_buffer)[start:], dtype=np.float32)
    pitch   = window_data[:, 4]   # idx: ear_avg=0,ear_l=1,ear_r=2,mar=3,pitch=4
    yaw     = window_data[:, 5]   # idx: yaw=5

    temporal = np.zeros(6, dtype=np.float32)
    temporal[0] = np.std(ear_avg)
    temporal[1] = np.std(mar)
    temporal[2] = np.std(pitch)
    temporal[3] = np.std(yaw)
    temporal[4] = np.sum(ear_avg < EAR_BLINK_THRESHOLD) / len(ear_avg)

    if len(pitch) > 1:
        temporal[5] = np.mean(np.abs(np.diff(pitch))) + np.mean(np.abs(np.diff(yaw)))

    return temporal


# ──────────────────────────────────────────────
# GRU 모델 정의 (step3_train.py와 동일 구조)
# ──────────────────────────────────────────────

class ConcentrationGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            N_FEATURES, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 32),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = (gru_out * attn_weights).sum(dim=1)
        return self.classifier(context)


# ──────────────────────────────────────────────
# 적응형 세션 추천 로직
# ──────────────────────────────────────────────

FACE_ABSENT_THRESHOLD = 30   # 초

class AdaptiveSessionAdvisor:
    """
    실시간 집중도를 기반으로 다음 세션과 휴식 시간을 동적으로 조절합니다.

    핵심 로직:
      - 세션 중 평균 집중도를 추적
      - 세션 종료 시 집중도에 따라 다음 세션/휴식 시간을 조절
        - 집중도 >= 70%: 다음 세션 +5분, 휴식 +1분
        - 집중도 40~70%: 기본 세션/휴식 유지
        - 집중도 < 40%: 다음 세션 -5분, 휴식 +2분
    """
    def __init__(self):
        self.state = "idle"               # idle / focusing / resting
        self.session_start = None
        self.rest_start    = None
        self.absent_start  = None

        # 적응형 세션 시간 (초)
        self.current_session_sec = DEFAULT_SESSION_MINUTES * 60
        self.current_break_sec   = DEFAULT_BREAK_MINUTES * 60

        # 세션 중 집중도 추적
        self.session_focus_samples = []

        self.recommendation = ""
        self.completed_sessions = 0

    def _adapt_times(self, avg_focus: float):
        """세션 종료 후 다음 세션/휴식 시간을 조절합니다."""
        if avg_focus >= 0.70:
            # 고집중 → 세션 연장, 휴식 약간 추가
            self.current_session_sec = min(
                self.current_session_sec + 5 * 60,
                MAX_SESSION_MINUTES * 60
            )
            self.current_break_sec = min(
                self.current_break_sec + 1 * 60,
                MAX_BREAK_MINUTES * 60
            )
        elif avg_focus < 0.40:
            # 저집중 → 세션 단축, 휴식 추가
            self.current_session_sec = max(
                self.current_session_sec - 5 * 60,
                MIN_SESSION_MINUTES * 60
            )
            self.current_break_sec = min(
                self.current_break_sec + 2 * 60,
                MAX_BREAK_MINUTES * 60
            )
        # 40~70%: 현재 설정 유지

    def update(self, is_focused: bool, face_detected: bool, current_time: float) -> str:

        # ── 얼굴 미검출 처리 ─────────────────────────
        if not face_detected:
            if self.absent_start is None:
                self.absent_start = current_time

            absent_sec = current_time - self.absent_start
            if absent_sec < FACE_ABSENT_THRESHOLD:
                self.recommendation = f"잠시 자리를 비웠어요 ({int(absent_sec)}초) - 세션 유지 중"
                return self.recommendation
            else:
                if self.state == "focusing":
                    self._end_session()
                if self.state != "resting":
                    self.state      = "resting"
                    self.rest_start = current_time
                elapsed = current_time - (self.rest_start or current_time)
                self.recommendation = (
                    f"자리 비움 {int(elapsed + FACE_ABSENT_THRESHOLD)}초 - 휴식으로 전환\n"
                    f"  다음 세션: {self.current_session_sec // 60}분"
                )
                return self.recommendation
        else:
            self.absent_start = None

        # ── 집중 / 비집중 판정 ───────────────────────
        if is_focused:
            if self.state != "focusing":
                self.state = "focusing"
                self.session_start = current_time
                self.session_focus_samples = []

            self.session_focus_samples.append(1.0)
            elapsed = current_time - self.session_start

            if elapsed >= self.current_session_sec:
                avg_focus = np.mean(self.session_focus_samples)
                self._end_session()
                self._adapt_times(avg_focus)
                self.state      = "resting"
                self.rest_start = current_time
                self.recommendation = (
                    f"세션 완료! (집중도: {avg_focus*100:.0f}%)\n"
                    f"  휴식 {self.current_break_sec // 60}분 추천\n"
                    f"  다음 세션: {self.current_session_sec // 60}분"
                )
            else:
                remain = self.current_session_sec - elapsed
                avg_focus = np.mean(self.session_focus_samples) if self.session_focus_samples else 0
                self.recommendation = (
                    f"집중 중 | 남은 시간: {int(remain//60)}분 {int(remain%60)}초\n"
                    f"  현재 세션 집중도: {avg_focus*100:.0f}%"
                )
        else:
            if self.state == "focusing":
                self.session_focus_samples.append(0.0)
                elapsed = current_time - self.session_start

                # 세션 도중 비집중이면 기록은 하되 세션 유지
                if elapsed >= self.current_session_sec:
                    avg_focus = np.mean(self.session_focus_samples)
                    self._end_session()
                    self._adapt_times(avg_focus)
                    self.state      = "resting"
                    self.rest_start = current_time
                    self.recommendation = (
                        f"세션 완료! (집중도: {avg_focus*100:.0f}%)\n"
                        f"  휴식 {self.current_break_sec // 60}분 추천\n"
                        f"  다음 세션: {self.current_session_sec // 60}분"
                    )
                else:
                    remain = self.current_session_sec - elapsed
                    avg_focus = np.mean(self.session_focus_samples)
                    self.recommendation = (
                        f"집중이 흐트러지고 있어요 | 남은 시간: {int(remain//60)}분 {int(remain%60)}초\n"
                        f"  현재 세션 집중도: {avg_focus*100:.0f}%"
                    )
            else:
                # 휴식 상태
                if self.state != "resting":
                    self.state      = "resting"
                    self.rest_start = current_time

                elapsed = current_time - (self.rest_start or current_time)
                if elapsed >= self.current_break_sec:
                    self.recommendation = (
                        f"충분히 쉬었어요! 다시 시작해 볼까요?\n"
                        f"  다음 세션: {self.current_session_sec // 60}분"
                    )
                else:
                    remain = self.current_break_sec - elapsed
                    self.recommendation = f"휴식 중 | 남은 시간: {int(remain//60)}분 {int(remain%60)}초"

        return self.recommendation

    def _end_session(self):
        self.completed_sessions += 1
        self.session_start = None
        self.session_focus_samples = []

    def get_status_summary(self) -> str:
        session_min = self.current_session_sec // 60
        break_min   = self.current_break_sec // 60
        return f"세션: {session_min}분 | 휴식: {break_min}분 | 완료: {self.completed_sessions}회"


# ──────────────────────────────────────────────
# 화면 렌더링
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
    model = ConcentrationGRU().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # threshold 로드
    FOCUSED_THRESHOLD = 0.5
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            thresh_data = json.load(f)
        FOCUSED_THRESHOLD = thresh_data.get("focused_threshold", 0.5)
        print(f"threshold 로드: P(집중) >= {FOCUSED_THRESHOLD} → 집중 판정")
    else:
        print(f"threshold.json 없음 → 기본값 {FOCUSED_THRESHOLD} 사용")

    print("모델 로드 완료! 카메라 시작 중...")

    # MediaPipe
    face_mesh = mp_lib.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 버퍼
    raw_buffer    = deque(maxlen=SEQ_LEN + TEMPORAL_WINDOW)  # 원시 특징 (시간적 계산용 여유)
    mar_buffer    = deque(maxlen=SEQ_LEN + TEMPORAL_WINDOW)  # MAR 값 (mar_std 계산용)
    full_buffer   = deque(maxlen=SEQ_LEN)    # 13-dim 전체 특징 시퀀스
    pred_buffer   = deque(maxlen=DECISION_WINDOW)
    advisor       = AdaptiveSessionAdvisor()

    pred_label    = "대기 중..."
    focus_ratio   = 0.0
    confidence    = 0.0

    SESSION_FOCUS_THRESHOLD = 0.6

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

            # [ear_avg, ear_l, ear_r, pitch, yaw, roll, face_detected] — MAR raw 제외
            raw_feat = np.array([(ear_l+ear_r)/2, ear_l, ear_r, mar, pitch, yaw, roll, 1.0],
                                dtype=np.float32)
            raw_buffer.append(raw_feat)
            mar_buffer.append(mar)

            # 시간적 특징 계산
            temporal_feat = compute_temporal_features_realtime(raw_buffer, mar_buffer)
            full_feat = np.concatenate([raw_feat, temporal_feat])  # (14,)
            full_buffer.append(full_feat)

            # 얼굴 랜드마크 그리기
            for idx in LEFT_EYE_EAR + RIGHT_EYE_EAR + MOUTH_MAR:
                cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 180), -1)
        else:
            raw_buffer.append(np.zeros(8, dtype=np.float32))
            mar_buffer.append(0.0)
            full_buffer.append(np.zeros(N_FEATURES, dtype=np.float32))

        # SEQ_LEN 프레임 쌓이면 추론
        if len(full_buffer) == SEQ_LEN:
            seq = np.array(full_buffer, dtype=np.float32)          # (90, 14)
            seq_scaled = scaler.transform(seq).astype(np.float32)
            x = torch.tensor(seq_scaled[np.newaxis], dtype=torch.float32).to(device)

            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            confidence = float(probs[1])
            pred       = 1 if confidence >= FOCUSED_THRESHOLD else 0
            pred_buffer.append(pred)

            focus_ratio = sum(pred_buffer) / len(pred_buffer)
            is_focused  = focus_ratio >= SESSION_FOCUS_THRESHOLD
            pred_label  = "집중" if is_focused else "비집중"
            advisor.update(is_focused, face_detected, time.time())

        # ── UI 렌더링 ──────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 상태 텍스트
        label_color = (80, 220, 80) if pred_label == "집중" else (80, 80, 240)
        put_kr_text(frame, pred_label, (15, 8), font_size=26, color=label_color, bg=(20,20,20))

        # 세션 요약 (우상단)
        status = advisor.get_status_summary()
        put_kr_text(frame, status, (w - 350, 8), font_size=16, color=(180, 180, 180), bg=(20,20,20))

        # 집중도 바
        draw_bar(frame, "집중도", focus_ratio, x=15, y=68,  color=(80,200,80))
        draw_bar(frame, "확률  ", confidence,  x=15, y=105, color=(80,150,220))

        # 세션 추천 (하단)
        if advisor.recommendation:
            lines = advisor.recommendation.split('\n')
            for i, line in enumerate(lines):
                put_kr_text(frame, line, (15, h - 60 + i * 25),
                            font_size=18, color=(255, 220, 80), bg=(30, 30, 30))

        # 얼굴 미검출 경고
        if not face_detected:
            put_kr_text(frame, "얼굴을 카메라에 맞춰주세요",
                        (w//2-140, h//2), font_size=22, color=(80,80,240), bg=(20,20,20))

        # 버퍼 수집 진행률
        if len(full_buffer) < SEQ_LEN:
            ratio = len(full_buffer) / SEQ_LEN
            put_kr_text(frame, f"초기화 중... {int(ratio*100)}%",
                        (15, h-70), font_size=18, color=(180,180,180), bg=(20,20,20))

        cv2.imshow("Smartto - 적응형 뽀모도로", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("종료됨.")


if __name__ == "__main__":
    main()
