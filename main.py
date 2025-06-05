# main.py

import os
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from model import DeepVoiceDetector
from utils import (
    preprocess_mp3_bytes_to_input_tensor,
    preprocess_flac_file_to_input_tensor
)

app = FastAPI(title="DeepVoice Detection API", version="1.0")

# ──────────── 1) 모델 로드 ────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ← sample_input_shape을 (1, 400, 80)으로 수정
model = DeepVoiceDetector(input_channels=1,
                          hidden_size=128,
                          sample_input_shape=(1, 400, 80))
MODEL_PATH = "deepvoice_best.pt"

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("✅ 모델 로드 완료")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")

# ──────────── 2) 엔드포인트 정의 ────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Multipart/form-data로 .mp3 또는 .flac 파일을 업로드하면,
    내부적으로 전처리 → 모델 추론 → JSON 반환
    """

    filename = file.filename.lower()

    # 확장자 검사: .mp3 또는 .flac 만 허용
    if not (filename.endswith(".mp3") or filename.endswith(".flac")):
        raise HTTPException(status_code=400, detail="mp3 또는 flac 파일만 업로드 가능합니다.")

    # 파일 바이트 읽기
    file_bytes = await file.read()

    # 전처리 분기 (.mp3 / .flac)
    try:
        if filename.endswith(".mp3"):
            input_tensor = preprocess_mp3_bytes_to_input_tensor(
                file_bytes,
                sr=16000,
                n_mels=80,
                fixed_length=400
            )  # → (1, 1, 400, 80)
        else:
            # FLAC은 임시 파일로 저장 후 경로 기반 전처리
            with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as tmp:
                tmp_path = tmp.name
                tmp.write(file_bytes)

            input_tensor = preprocess_flac_file_to_input_tensor(
                tmp_path,
                sr=16000,
                n_mels=80,
                fixed_length=400
            )  # → (1, 1, 400, 80)

            os.remove(tmp_path)

        # GPU/CPU로 이동
        input_tensor = input_tensor.to(DEVICE)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"전처리 오류: {e}")

    # ──────────── 3) 모델 추론 ────────────
    with torch.no_grad():
        output = model(input_tensor)                     # (batch,) 출력
        prob = torch.sigmoid(output).squeeze().item()    # 확률값(0~1)
        is_spoof = True if prob > 0.5 else False

    # ──────────── 4) JSON 반환 ────────────
    return JSONResponse(content={"deepvoice": is_spoof, "prob": prob})


# ──────────── 5) 서버 실행 ────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
