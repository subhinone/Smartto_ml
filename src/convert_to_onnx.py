"""
XGBoost 모델 → ONNX 변환 스크립트
Flutter 온디바이스 추론을 위해 한 번만 실행하면 됩니다.

실행:
  cd Smartto_ml
  python src/convert_to_onnx.py

출력:
  models/xgb_model.onnx  ← 이 파일을 Flutter assets에 복사
"""

from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgb_model.joblib"
ONNX_PATH  = Path(__file__).parent.parent / "models" / "xgb_model.onnx"

def convert():
    print(f"[1] 모델 로드: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # feature 수 확인 (step2에서 만든 62개 통계 feature)
    n_features = model.n_features_in_
    print(f"    입력 feature 수: {n_features}")

    # ── onnxmltools 사용 (XGBoost 전용 변환기) ───────────────────────────
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)

        with open(ONNX_PATH, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"[2] onnxmltools 변환 완료: {ONNX_PATH}")
        _verify(n_features)
    except Exception as e:
        print(f"[오류] 변환 실패: {e}")

def _verify(n_features: int):
    """변환된 ONNX 모델 검증"""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ONNX_PATH))
        dummy = np.random.rand(1, n_features).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: dummy})
        print(f"[3] 검증 완료 ✓  더미 입력 → 출력: {out[0]}")
        print(f"\n완료! 다음 파일을 Flutter assets에 복사하세요:")
        print(f"  {ONNX_PATH}")
        print(f"  → smartto_new/assets/models/xgb_model.onnx")
    except ImportError:
        print("[3] onnxruntime 없어서 검증 생략 (pip install onnxruntime)")
    except Exception as e:
        print(f"[3] 검증 실패: {e}")

if __name__ == "__main__":
    convert()
