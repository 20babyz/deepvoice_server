import io
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment


def extract_mel_from_waveform(
    waveform: torch.Tensor,
    sr: int = 16000,
    n_mels: int = 80,
    fixed_length: int = 400
) -> np.ndarray:
    """
    waveform (1차원 Tensor, shape=[num_samples])을 받아서
      1) torchaudio.transforms.MelSpectrogram 적용 → Tensor shape = [1, n_mels, time_steps]
      2) torchaudio.transforms.AmplitudeToDB 적용 → dB 스케일
      3) numpy array로 변환 → shape = [n_mels, time_steps]
      4) 좌우(시간차원) padding/trimming → shape = [n_mels, fixed_length]
      5) z-score 정규화 (mean=0, std=1)
    최종적으로 (n_mels, fixed_length) 형태의 numpy.ndarray 반환
    """

    # ── 1) Mel Spectrogram 추출 ──
    # waveform이 1D Tensor([num_samples])이므로, unsqueeze(0) → (1, num_samples)
    mel_tensor = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels
    )(waveform.unsqueeze(0))  # 결과 shape: (1, n_mels, time_steps)

    # ── 2) dB 스케일 변환 ──
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_tensor)  # (1, n_mels, time_steps)

    # ── 3) numpy 배열로 변환 ──
    mel_np = mel_db.squeeze(0).numpy()  # (n_mels, time_steps)

    # ── 4) Padding or Trimming to fixed_length ──
    if mel_np.shape[1] < fixed_length:
        pad_width = fixed_length - mel_np.shape[1]
        # 시간축 차원(두 번째 axis) 우측으로 0 패딩
        mel_np = np.pad(mel_np, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # 너무 긴 경우 앞에서부터 잘라냄
        mel_np = mel_np[:, :fixed_length]

    # ── 5) z-score 정규화 ──
    mel_np = (mel_np - mel_np.mean()) / (mel_np.std() + 1e-6)

    return mel_np


def load_mp3_bytes_to_waveform(mp3_bytes: bytes, sr: int = 16000) -> torch.Tensor:
    """
    MP3 바이트를 AudioSegment로 로드하여 mono, 재샘플링한 뒤
    1D waveform Tensor([num_samples])로 반환.
    """
    # MP3 바이트를 파일처럼 읽어들여서 AudioSegment 생성
    audio_seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    audio_seg = audio_seg.set_channels(1)      # mono
    audio_seg = audio_seg.set_frame_rate(sr)   # 재샘플링

    # AudioSegment 내부 raw samples를 numpy array로 변환
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples)       # shape: (num_samples,)
    return waveform


def preprocess_mp3_bytes_to_input_tensor(
    mp3_bytes: bytes,
    sr: int = 16000,
    n_mels: int = 80,
    fixed_length: int = 400
) -> torch.Tensor:
    """
    MP3 바이트 → (1, 1, n_mels, fixed_length) 형태의 입력 텐서 반환.
    내부적으로 extract_mel_from_waveform을 사용하여 Mel-Spectrogram 생성 및 정규화.
    """

    # ── 1) MP3 바이트 → waveform Tensor ([num_samples]) ──
    waveform = load_mp3_bytes_to_waveform(mp3_bytes, sr=sr)

    # ── 2) waveform → Mel-Numpy ([n_mels, fixed_length]) ──
    mel_np = extract_mel_from_waveform(
        waveform,
        sr=sr,
        n_mels=n_mels,
        fixed_length=fixed_length
    )  # 결과 shape: (n_mels, fixed_length)

    # ── 3) numpy → torch.Tensor ([n_mels, fixed_length]) ──
    mel_tensor = torch.from_numpy(mel_np).float()

    # ── 4) 채널 차원 추가 → ([1, n_mels, fixed_length]) ──
    mel_tensor = mel_tensor.unsqueeze(0)

    # ── 5) 배치 차원 추가 → ([1, 1, n_mels, fixed_length]) ──
    mel_tensor = mel_tensor.unsqueeze(0)

    return mel_tensor  # 최종 shape: (1, 1, n_mels, fixed_length)


def load_flac_file_to_waveform(file_path: str, sr: int = 16000) -> torch.Tensor:
    """
    디스크상의 FLAC 파일 경로를 받아 waveform Tensor로 반환.
    - torchaudio.load()를 사용하여 stereo → mono, 필요한 경우 재샘플링 수행.
    """
    waveform, orig_sr = torchaudio.load(file_path)  # 반환: (Tensor [channels, num_samples], orig_sr)
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        # stereo 등의 다중 채널인 경우, 평균하여 mono로 변환
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, num_samples)
    waveform = waveform.squeeze(0)  # (num_samples,)

    # 샘플레이트가 다르면 재샘플링
    if orig_sr != sr:
        waveform = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sr)(
            waveform.unsqueeze(0)
        ).squeeze(0)  # (num_samples,)

    return waveform


def preprocess_flac_file_to_input_tensor(
    file_path: str,
    sr: int = 16000,
    n_mels: int = 80,
    fixed_length: int = 400
) -> torch.Tensor:
    """
    FLAC 파일 경로 → (1, 1, n_mels, fixed_length) 형태의 입력 텐서 반환.
    """
    waveform = load_flac_file_to_waveform(file_path, sr=sr)  # (num_samples,)
    mel_np = extract_mel_from_waveform(
        waveform,
        sr=sr,
        n_mels=n_mels,
        fixed_length=fixed_length
    )  # (n_mels, fixed_length)

    mel_tensor = torch.from_numpy(mel_np).float().unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,fixed_length)
    return mel_tensor
