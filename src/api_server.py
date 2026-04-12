"""
FastAPI 집중도 측정 서버
Flutter 앱에서 2초마다 이미지를 전송하면 집중도 점수와 상태를 반환합니다.

실행 방법:
  pip install fastapi uvicorn python-multipart --break-system-packages
  cd Smartto_ml
  uvicorn src.api_server:app --host 0.0.0.0 --port 8000

Flutter에서 접속 주소:
  실기기(같은 Wi-Fi): http://<맥 IP>:8000
  맥 IP 확인: 터미널에서 ifconfig | grep "inet " | grep -v 127
"""

import base64
import time
import joblib
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# step4 내부 로직 재사용
import sys
sys.path.insert(0, str(Path(__file__).parent))
from step1_extract_features import (
    compute_ear, compute_mar, compute_head_pose,
    LEFT_EYE, RIGHT_EYE, mp_face_mesh,
)
from step2_prepare_dataset import compute_clip_features
from step4_realtime import (
    Config, RuleBasedDetector, DistractionDetector, compute_focus_score,
)
import mediapipe as mp

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Smartto Concentration API")

# Flutter에서 HTTP 요청을 허용하기 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 모델 & 상태 초기화 (서버 시작 시 1회)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"[API] 모델 로드 완료: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[API] 모델 로드 실패 (Rule-based만 동작): {e}")

# MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 실시간 상태 (세션 단위로 유지)
rule_detector       = RuleBasedDetector()
distraction_detector = DistractionDetector()
feature_buffer: deque = deque(maxlen=150)   # 최근 5초 × 30fps
ml_history: deque   = deque(maxlen=3)
stable_ml_focused   = True

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response 모델
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image_b64: str  # base64 인코딩된 JPEG 이미지


class AnalyzeResponse(BaseModel):
    focus_score: float      # 0 ~ 100
    status: str             # 'Focused' | 'Drowsy' | 'Distracted'
    ear: float
    mar: float
    yaw: float
    rule_reasons: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# 메인 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    global stable_ml_focused

    # 1. base64 → OpenCV 이미지
    img_bytes = base64.b64decode(req.image_b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return AnalyzeResponse(
            focus_score=0, status="Error",
            ear=0, mar=0, yaw=0, rule_reasons=["이미지 디코딩 실패"],
        )

    # 2. MediaPipe FaceMesh 추론
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    now = time.time()
    ear = mar = yaw = 0.0
    face_detected = False

    if result.multi_face_landmarks:
        face_detected = True
        lm = result.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in range(468)])
        ear = (compute_ear(pts, LEFT_EYE) + compute_ear(pts, RIGHT_EYE)) / 2.0
        mar = compute_mar(pts)
        yaw, _, _ = compute_head_pose(pts, (h, w))

        feature_buffer.append([ear, mar, yaw, now])

    # 3. Rule-based 판단
    rule_result = rule_detector.update(ear, mar, face_detected, now)
    distraction_signals = distraction_detector.update(face_detected, yaw, ear, now)

    rule_reasons = []
    if rule_result.get("drowsy"):
        rule_reasons.append(rule_result.get("reason", "Drowsy"))
    for k, v in distraction_signals.items():
        if v:
            rule_reasons.append(k)

    # 4. ML 판단 (모델이 있고 버퍼가 충분할 때)
    ml_focused = True
    if model is not None and len(feature_buffer) >= 30:
        window_features = np.array([[f[0], f[1], f[2]] for f in list(feature_buffer)[-90:]])
        clip_feat = compute_clip_features(window_features).reshape(1, -1)
        try:
            pred = model.predict(clip_feat)[0]
            ml_focused = bool(pred == 1)
            ml_history.append(ml_focused)
            if len(ml_history) == 3 and len(set(ml_history)) == 1:
                stable_ml_focused = ml_history[0]
        except Exception:
            pass

    # 5. 집중도 점수 계산
    focus_score = compute_focus_score(stable_ml_focused, distraction_signals, rule_reasons)

    # 6. 최종 상태 결정
    if rule_result.get("drowsy"):
        status = "Drowsy"
    elif rule_reasons:
        status = "Distracted"
    elif stable_ml_focused:
        status = "Focused"
    else:
        status = "Distracted"

    return AnalyzeResponse(
        focus_score=round(focus_score, 1),
        status=status,
        ear=round(ear, 3),
        mar=round(mar, 3),
        yaw=round(yaw, 1),
        rule_reasons=rule_reasons,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/reset")
async def reset():
    """세션 시작 시 상태 초기화"""
    global rule_detector, distraction_detector, feature_buffer, ml_history, stable_ml_focused
    rule_detector        = RuleBasedDetector()
    distraction_detector = DistractionDetector()
    feature_buffer.clear()
    ml_history.clear()
    stable_ml_focused = True
    return {"status": "reset ok"}
