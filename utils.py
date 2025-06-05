import numpy as np
import torchaudio

def extract_mel_from_waveform(
    waveform,
    sr: int = 16000,
    n_mels: int = 80,
    fixed_length: int = 400,
):
    """
    STEP 2) Mel-spectrogram 추출 → dB → z-score 정규화 → 패딩/트리밍 함수임
    """
    # Mel + dB
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
    )(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

    mel_np = mel_db.squeeze(0).numpy()  # (n_mels, T)

    # 길이 맞추기
    if mel_np.shape[1] < fixed_length:          # 패딩
        pad_width = fixed_length - mel_np.shape[1]
        mel_np = np.pad(mel_np, ((0, 0), (0, pad_width)), mode="constant")
    else:                                       # 트리밍
        mel_np = mel_np[:, :fixed_length]

    # z-score 정규화
    mel_np = (mel_np - mel_np.mean()) / (mel_np.std() + 1e-6)
    return mel_np
