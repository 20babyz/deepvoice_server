"""
FastAPI 실행 파일임
$ uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import tempfile
from typing import Dict

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException

from model import load_model
from utils import extract_mel_from_waveform

# --------------------------------------------- #
# 환경 및 모델 준비
# --------------------------------------------- #
BASE_PATH  = os.environ.get("DEEPVOICE_BASE_PATH", ".")
WEIGHT_PATH = os.path.join(BASE_PATH, "deepvoice_best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = load_model(WEIGHT_PATH, device)   # ⚡ 서버 부팅 시 1회만 로드

app = FastAPI(
    title="DeepVoice Detection API",
    description="딥보이스(voice spoofing) 탐지 서비스 v1.0",
    version="1.0.0",
)


# --------------------------------------------- #
# 라우터
# --------------------------------------------- #
@app.post("/predict", summary="오디오 파일 예측")
async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
    # ⬇️ STEP 1) 업로드 파일 검증 및 로드
    if not file.filename.lower().endswith((".wav", ".flac", ".mp3")):
        raise HTTPException(status_code=400, detail="지원하지 않는 오디오 형식임")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    waveform, sr = torchaudio.load(tmp_path)   # STEP 1
    os.remove(tmp_path)                        # 임시 파일 바로 삭제

    # ⬇️ STEP 2) Mel-spectrogram 추출
    mel_np = extract_mel_from_waveform(waveform, sr=sr)

    # ⬇️ STEP 3) Tensor 변환 + 차원 맞추기
    mel_tensor = (
        torch.tensor(mel_np)
        .unsqueeze(0)          # batch
        .unsqueeze(0)          # channel
        .float()
        .to(device)
    )                          # (1, 1, 80, 400)

    # ⬇️ STEP 4) 모델 추론
    with torch.no_grad():
        logit = model(mel_tensor)
        logit = torch.clamp(logit, min=-10, max=10)
        prob  = torch.sigmoid(logit).item()

    result = "spoof 🔴" if prob > 0.5 else "bonafide 🟢"

    return {
        "filename": file.filename,
        "probability": f"{prob:.4f}",
        "result": result,
    }


@app.get("/health", summary="헬스 체크")
def health():
    return {"status": "ok"}
