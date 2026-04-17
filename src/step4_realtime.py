"""
Step 4: 실시간 집중도 추론 + 적응형 뽀모도로 타이머

구조:
  1층 - Rule-based (확정적 판단):
    - 연속 2초 이상 눈 감김 → 확정 비집중
    - 하품 감지 → 확정 비집중
    - 얼굴 미감지 3초 이상 → 확정 비집중

  2층 - XGBoost (미세 판단):
    - Rule이 트리거되지 않을 때 ML 모델로 집중도 예측
    - 최근 N초간의 feature 통계량으로 판단

  적응형 타이머:
    - 세션 중 집중도 점수 누적
    - 세션 종료 후 다음 세션 시간 추천

Usage:
  python step4_realtime.py
  python step4_realtime.py --no-model   # Rule-based만 사용
"""

import argparse
import time
import json
import joblib
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque

# step1, step2의 함수 재사용
from step1_extract_features import (
    compute_ear, compute_mar, compute_head_pose, compute_gaze_offset,
    LEFT_EYE, RIGHT_EYE, mp_face_mesh
)
from step2_prepare_dataset import compute_clip_features


# ── 설정 ────────────────────────────────────────────────────────

class Config:
    # Rule-based thresholds (졸음)
    EAR_CLOSE_THRESHOLD = 0.20    # 눈 감김 판단
    EAR_CLOSE_DURATION = 3.0      # 연속 N초 이상이면 졸음
    MAR_YAWN_THRESHOLD = 0.65     # 하품 판단
    MAR_YAWN_FRAMES = 5           # N프레임 이상 지속이면 하품

    # 비집중 감지 thresholds (지속 시간 기반)
    HEAD_TURN_YAW = 28.0          # 고개 돌림 판단 각도 (도)
    HEAD_TURN_DURATION = 15.0     # N초 이상 지속되면 비집중
    NO_FACE_ABSENT = 30.0         # 얼굴 미감지 N초 이상 → 자리 이탈
    STARE_BLINK_WINDOW = 30.0     # 멍때리기 판단 윈도우 (초)
    STARE_BLINK_MIN = 2           # 윈도우 내 최소 눈깜빡임 횟수 (이하면 멍때리기)

    # ML 판단 주기
    ML_WINDOW_SEC = 3.0           # 최근 N초 데이터로 ML 판단
    ML_INTERVAL_SEC = 1.0         # N초마다 ML 추론

    # 적응형 타이머
    DEFAULT_FOCUS_MIN = 25        # 기본 집중 세션 (분)
    DEFAULT_BREAK_MIN = 5         # 기본 휴식 (분)
    MIN_FOCUS_MIN = 25            # 최소 집중 세션
    MAX_FOCUS_MIN = 50            # 최대 집중 세션
    FOCUS_ADJUST_STEP = 5         # 조절 단위 (분)

    # 집중도 점수 가중치 (합계 = 1.0)
    W_ML = 0.50                   # ML 판단 (졸음 여부)
    W_PRESENCE = 0.30             # 자리 이탈 없음
    W_STARE = 0.20                # 멍때리기 없음

    # 멍때리기 정상 깜빡임 기준 (30초당 평균 ~6회)
    NORMAL_BLINK_PER_WINDOW = 6


# ── Rule-based 판단기 ───────────────────────────────────────────

class RuleBasedDetector:
    """확정적 비집중 신호 감지"""

    def __init__(self, fps=30):
        self.fps = fps
        self.eye_close_frames = 0
        self.yawn_frames = 0
        self.no_face_frames = 0

    def update(self, ear_avg, mar, face_detected):
        """프레임 단위 업데이트. 졸음 신호가 있으면 reason 리스트 반환, 없으면 None"""
        reasons = []

        if not face_detected:
            self.eye_close_frames = 0
            self.yawn_frames = 0
            return None

        # 눈 감김 (졸음)
        if ear_avg < Config.EAR_CLOSE_THRESHOLD:
            self.eye_close_frames += 1
            if self.eye_close_frames >= self.fps * Config.EAR_CLOSE_DURATION:
                reasons.append("eyes_closed")
        else:
            self.eye_close_frames = 0

        # 하품
        if mar > Config.MAR_YAWN_THRESHOLD:
            self.yawn_frames += 1
            if self.yawn_frames >= Config.MAR_YAWN_FRAMES:
                reasons.append("yawning")
        else:
            self.yawn_frames = 0

        return reasons if reasons else None


# ── 비집중 감지기 (지속 시간 기반) ─────────────────────────────

class DistractionDetector:
    """
    공부 중 비집중 신호를 지속 시간 기반으로 감지.
    순간 판단이 아닌 N초 이상 지속될 때만 비집중으로 판단.
    """

    def __init__(self, fps=30):
        self.fps = fps

        # 고개 돌림 추적
        self.head_turn_frames = 0

        # 자리 이탈 추적
        self.no_face_frames = 0

        # 멍때리기 추적 (윈도우 내 눈깜빡임 횟수)
        self.blink_window = deque()   # (timestamp, is_blink) 저장
        self.prev_ear_closed = False
        self.blink_count_window = 0

    def update(self, face_detected, yaw, ear_avg, timestamp):
        """
        매 프레임 업데이트.
        반환: (signals, scores)
          signals: dict {signal: bool} — 각 비집중 신호 활성화 여부 (상태 표시용)
          scores:  dict {signal: float 0.0~1.0} — 연속값 점수 (집중도 계산용)
        """
        signals = {
            'head_turned': False,
            'absent':      False,
            'staring':     False,
        }

        # ── 자리 이탈 ──
        if not face_detected:
            self.no_face_frames += 1
            if self.no_face_frames >= self.fps * Config.NO_FACE_ABSENT:
                signals['absent'] = True
        else:
            self.no_face_frames = 0

        # presence_score: 얼굴 미감지 프레임 비율 → 1.0(완전 존재) ~ 0.0(이탈)
        absent_threshold_frames = self.fps * Config.NO_FACE_ABSENT
        presence_ratio = 1.0 - min(self.no_face_frames / absent_threshold_frames, 1.0)

        if not face_detected:
            scores = {
                'presence_ratio': presence_ratio,
                'blink_ratio':    0.0,
            }
            return signals, scores

        # ── 고개 장시간 돌림 (상태 표시용으로 유지) ──
        if abs(yaw) > Config.HEAD_TURN_YAW:
            self.head_turn_frames += 1
            if self.head_turn_frames >= self.fps * Config.HEAD_TURN_DURATION:
                signals['head_turned'] = True
        else:
            self.head_turn_frames = max(0, self.head_turn_frames - 2)

        # ── 멍때리기: 윈도우 내 눈깜빡임 부족 ──
        ear_closed = ear_avg < Config.EAR_CLOSE_THRESHOLD
        is_blink = (not self.prev_ear_closed) and ear_closed
        self.prev_ear_closed = ear_closed

        self.blink_window.append((timestamp, is_blink))
        if is_blink:
            self.blink_count_window += 1

        cutoff = timestamp - Config.STARE_BLINK_WINDOW
        while self.blink_window and self.blink_window[0][0] < cutoff:
            _, old_blink = self.blink_window.popleft()
            if old_blink:
                self.blink_count_window -= 1

        window_full = (len(self.blink_window) >= self.fps * Config.STARE_BLINK_WINDOW * 0.8)
        if window_full and self.blink_count_window <= Config.STARE_BLINK_MIN:
            signals['staring'] = True

        # blink_ratio: 깜빡임 횟수 / 정상 기준 → 1.0(정상) ~ 0.0(멍때리기)
        if window_full:
            blink_ratio = min(self.blink_count_window / Config.NORMAL_BLINK_PER_WINDOW, 1.0)
        else:
            blink_ratio = 1.0  # 윈도우 미충족 시 정상으로 간주

        scores = {
            'presence_ratio': presence_ratio,
            'blink_ratio':    blink_ratio,
        }
        return signals, scores


# ── ML 판단기 ───────────────────────────────────────────────────

class MLDetector:
    """XGBoost 모델 기반 집중도 판단"""

    def __init__(self, model_path, threshold_path):
        self.model = joblib.load(model_path)

        with open(threshold_path) as f:
            config = json.load(f)
        self.threshold = config['threshold']

        print(f"[ML] Model loaded, threshold={self.threshold:.2f}")

    def predict(self, frame_buffer):
        """
        frame_buffer: deque of feature vectors (N, 10)
        반환: (is_focused: bool, confidence: float)
        """
        if len(frame_buffer) < 10:
            return True, 0.5  # 데이터 부족 시 기본 집중

        frames = np.array(list(frame_buffer), dtype=np.float32)
        clip_feats, _ = compute_clip_features(frames)

        if clip_feats is None:
            return True, 0.5

        # feature dict → numpy array (feature_names 순서대로)
        from step2_prepare_dataset import _get_feature_names
        feature_names = _get_feature_names()
        X = np.array([[clip_feats.get(name, 0.0) for name in feature_names]],
                     dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        proba = self.model.predict_proba(X)[0]
        p_alert = proba[1]
        is_focused = p_alert >= self.threshold

        return is_focused, float(p_alert)


# ── 적응형 타이머 ───────────────────────────────────────────────

class AdaptiveTimer:
    def __init__(self):
        self.current_focus_min = Config.DEFAULT_FOCUS_MIN
        self.current_break_min = Config.DEFAULT_BREAK_MIN
        self.session_scores = [] 

    def record_session(self, avg_focus_score):
        self.session_scores.append(avg_focus_score)

    def recommend_next(self):
        if not self.session_scores:
            return self.current_focus_min, self.current_break_min

        last_score = self.session_scores[-1]

        if last_score >= 70:
            self.current_focus_min = min(
                self.current_focus_min + Config.FOCUS_ADJUST_STEP,
                Config.MAX_FOCUS_MIN
            )
            self.current_break_min = 5
            msg = f"Great focus! Next: {self.current_focus_min}min study + {self.current_break_min}min break"

        elif last_score >= 40:
            msg = f"Not bad! Keeping {self.current_focus_min}min study + {self.current_break_min}min break"

        else:
            self.current_focus_min = max(
                self.current_focus_min - Config.FOCUS_ADJUST_STEP,
                Config.MIN_FOCUS_MIN
            )
            self.current_break_min = 10
            msg = f"Tough session! Next: {self.current_focus_min}min study + {self.current_break_min}min break"

        return self.current_focus_min, self.current_break_min, msg


# ── 메인 실시간 루프 ────────────────────────────────────────────

def compute_focus_score(ml_confidence, is_drowsy, continuous_scores):
    """
    연속값 기반 집중도 점수 계산.
    반환: int (0 ~ 100)
    """
    # ML: 졸음이면 confidence를 반전 (졸음 확신 높을수록 낮은 점수)
    ml_score = Config.W_ML * (ml_confidence if not is_drowsy else (1.0 - ml_confidence))

    # 자리 이탈: 얼굴 감지 비율 (1.0 = 계속 있음, 0.0 = 완전 이탈)
    presence_score = Config.W_PRESENCE * continuous_scores.get('presence_ratio', 1.0)

    # 멍때리기: 깜빡임 빈도 비율 (1.0 = 정상, 0.0 = 전혀 안 깜빡임)
    stare_score = Config.W_STARE * continuous_scores.get('blink_ratio', 1.0)

    return int((ml_score + presence_score + stare_score) * 100)


def run_realtime(use_model=True):
    """웹캠 기반 실시간 집중도 모니터링"""
    base = Path(__file__).resolve().parent.parent

    # 감지기 초기화
    fps = 30
    rule_detector = RuleBasedDetector(fps=fps)
    distraction_detector = DistractionDetector(fps=fps)
    ml_detector = None

    if use_model:
        model_path = base / "models" / "xgb_model.joblib"
        threshold_path = base / "models" / "threshold.json"
        if model_path.exists() and threshold_path.exists():
            ml_detector = MLDetector(model_path, threshold_path)
        else:
            print("[WARN] Model files not found. Using rule-based only.")

    timer = AdaptiveTimer()

    # 프레임 버퍼 (ML 판단용)
    buffer_size = int(Config.ML_WINDOW_SEC * fps)
    frame_buffer = deque(maxlen=buffer_size)

    # 세션 상태
    focus_scores = []
    session_start = time.time()
    last_ml_time = time.time()
    ml_focused = True
    ml_confidence = 0.5
    distraction_signals = {'head_turned': False, 'absent': False, 'staring': False}
    continuous_scores = {'presence_ratio': 1.0, 'blink_ratio': 1.0}

    # 상태 안정화: 3번 연속 같은 결과여야 화면에 반영
    ml_history = deque(maxlen=3)
    stable_ml_focused = True

    # 카메라
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print("\n" + "="*50)
    print("  Adaptive Pomodoro - Real-time Monitor")
    print("  Focused: Green / Drowsy: Red / Distracted: Orange")
    print("  Press 'q' to quit, 's' to end session")
    print("="*50)

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

            now = time.time()
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # ── Feature 추출 ──
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                ear_l = compute_ear(lm, LEFT_EYE)
                ear_r = compute_ear(lm, RIGHT_EYE)
                ear_avg = (ear_l + ear_r) / 2
                mar = compute_mar(lm)
                pitch, yaw, roll = compute_head_pose(lm, img_w, img_h)
                gaze_h, gaze_v = compute_gaze_offset(lm, img_w, img_h)
                face_detected = True
                feat_vec = [ear_l, ear_r, ear_avg, mar,
                            pitch, yaw, roll, gaze_h, gaze_v, 1.0]
            else:
                ear_avg = mar = yaw = 0.0
                face_detected = False
                feat_vec = [0.0] * 9 + [0.0]

            frame_buffer.append(feat_vec)

            # ── 1층: 졸음 rule-based ──
            rule_reasons = rule_detector.update(ear_avg, mar, face_detected)

            # ── 2층: 비집중 감지 (지속 시간 기반) ──
            distraction_signals, continuous_scores = distraction_detector.update(
                face_detected, yaw, ear_avg, now
            )

            # ── 3층: ML (주기적) ──
            if ml_detector and (now - last_ml_time) >= Config.ML_INTERVAL_SEC:
                ml_focused, ml_confidence = ml_detector.predict(frame_buffer)
                ml_history.append(ml_focused)
                # 3번 연속 같은 결과일 때만 상태 변경
                if len(ml_history) == 3 and all(v == ml_history[0] for v in ml_history):
                    stable_ml_focused = ml_history[0]
                last_ml_time = now

            # ── 집중도 점수 계산 (연속값 기반) ──
            is_drowsy = bool(rule_reasons)
            score = compute_focus_score(ml_confidence, is_drowsy, continuous_scores)
            focus_scores.append(score)

            # ── 상태 텍스트 & 색상 결정 ──
            if rule_reasons:
                status_text = f"DROWSY ({', '.join(rule_reasons)})"
                color = (0, 0, 255)       # 빨강
            elif distraction_signals.get('absent'):
                status_text = "ABSENT (away from desk)"
                color = (0, 0, 200)       # 진빨강
            elif distraction_signals.get('head_turned'):
                status_text = "DISTRACTED (head turned)"
                color = (0, 140, 255)     # 주황
            elif distraction_signals.get('staring'):
                status_text = "DISTRACTED (spacing out)"
                color = (0, 165, 255)     # 연주황
            elif not stable_ml_focused:
                status_text = f"DROWSY (ML, conf:{ml_confidence:.2f})"
                color = (0, 80, 255)      # 빨강 계열
            else:
                status_text = f"FOCUSED (conf:{ml_confidence:.2f})"
                color = (0, 200, 80)      # 초록

            # ── 화면 표시 ──
            elapsed = now - session_start
            elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"

            cv2.putText(frame, status_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, f"Time: {elapsed_str}", (10, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if face_detected:
                cv2.putText(frame,
                            f"EAR:{ear_avg:.2f}  MAR:{mar:.2f}  Yaw:{yaw:.1f}",
                            (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (200, 200, 200), 1)

            # 집중도 점수 바 (최근 30초)
            recent = focus_scores[-fps * 30:]
            if recent:
                recent_score = np.mean(recent)
                bar_w = int(200 * recent_score)
                cv2.rectangle(frame, (10, img_h - 45), (210, img_h - 15),
                              (50, 50, 50), -1)
                bar_color = (0, 200, 80) if recent_score >= 0.7 else \
                            (0, 165, 255) if recent_score >= 0.4 else (0, 0, 255)
                cv2.rectangle(frame, (10, img_h - 45), (10 + bar_w, img_h - 15),
                              bar_color, -1)
                cv2.putText(frame, f"Focus: {recent_score:.0%}",
                            (220, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1)

            cv2.imshow('Adaptive Pomodoro', frame)

            # ── 키 입력 ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if focus_scores:
                    avg_score = np.mean(focus_scores)
                    timer.record_session(avg_score)
                    focus_min, break_min, msg = timer.recommend_next()
                    print(f"\n[Session End] Avg focus score: {avg_score:.2f}")
                    print(f"[Recommend] {msg}")
                focus_scores = []
                session_start = now
                frame_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()

    if focus_scores:
        avg_score = np.mean(focus_scores)
        timer.record_session(avg_score)
        focus_min, break_min, msg = timer.recommend_next()
        print(f"\n[Final Session] Avg focus score: {avg_score:.2f}")
        print(f"[Recommend] {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-model', action='store_true',
                        help='Rule-based only, no ML model')
    args = parser.parse_args()

    run_realtime(use_model=not args.no_model)


if __name__ == "__main__":
    main()
