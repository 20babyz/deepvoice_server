# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepVoiceModel(nn.Module):
    """
    deepvoice_best.pt 내부의 state_dict 키와 정확히 매칭되도록
    채널 수와 차원을 모두 동일하게 맞춘 모델 구현 예시입니다.

    요점:
    - 첫 번째 Conv2d out_channels=32
    - 두 번째 Conv2d out_channels=64
    - 세 번째 Conv2d out_channels=128
    - 3회 MaxPool2d(2,2) → freq 축이 80 → 40 → 20 → 10 으로 줄어든다고 가정
    - GRU input_size = 128 * 10 = 1280, hidden_size = 128, bidirectional=True
    - Attention query/key/value projection 출력 차원 = 256
    - 최종 FC in_features = 256, out_features = 1
    """

    def __init__(self):
        super(DeepVoiceModel, self).__init__()

        # ──────────── 1) CNN 블록 정의 ────────────
        # state_dict에 저장된 키들을 보면 cnn.0, cnn.1, cnn.4, cnn.5, cnn.8, cnn.9 가 사용됨
        # 따라서 아래처럼 nn.Sequential을 구성하여 인덱스가 정확히 일치하도록 합니다.
        self.cnn = nn.Sequential(
            # ── 첫 번째 Conv 블록 ──
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # cnn.0
            nn.BatchNorm2d(num_features=32),                                                # cnn.1
            nn.ReLU(),                                                                       # cnn.2 (파라미터 없음)
            nn.MaxPool2d(kernel_size=2, stride=2),                                           # cnn.3

            # ── 두 번째 Conv 블록 ──
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # cnn.4
            nn.BatchNorm2d(num_features=64),                                                # cnn.5
            nn.ReLU(),                                                                       # cnn.6
            nn.MaxPool2d(kernel_size=2, stride=2),                                           # cnn.7

            # ── 세 번째 Conv 블록 ──
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),# cnn.8
            nn.BatchNorm2d(num_features=128),                                               # cnn.9
            nn.ReLU(),                                                                       # cnn.10
            nn.MaxPool2d(kernel_size=2, stride=2)                                           # cnn.11
            # (이로써 freq 차원: 80 → 40 → 20 → 10 이 됩니다)
        )

        # ──────────── 2) GRU 블록 정의 ────────────
        # state_dict을 보면 gru.weight_ih_l0 의 shape가 (384, 1280) 이므로,
        #   – hidden_size = 128 (여기서 3*128 = 384)
        #   – input_size = 1280
        # bidirectional=True 이므로 양방향 출력을 위해 hidden_size*2=256 차원이 나옵니다.
        self.gru = nn.GRU(
            input_size=1280,      # CNN 마지막 출력(128 채널 × freq'=10) = 1280
            hidden_size=128,      # state_dict 상 3*128 = 384 행
            num_layers=1,
            batch_first=True,
            bidirectional=True    # state_dict에 weight_ih_l0_reverse 가 있으므로 True
        )

        # ──────────── 3) Attention 블록 정의 ────────────
        # GRU 이후에 hidden_size*2 = 128*2 = 256 차원이 나옵니다.
        # 따라서 query/key/value projection 모두 in_features=256, out_features=256 이 되어야 합니다.
        self.attn = nn.Module()
        self.attn.query = nn.Linear(in_features=256, out_features=256)  # attn.query.weight/bias
        self.attn.key   = nn.Linear(in_features=256, out_features=256)  # attn.key.weight/bias
        self.attn.value = nn.Linear(in_features=256, out_features=256)  # attn.value.weight/bias

        # ──────────── 4) 최종 FC 블록 정의 ────────────
        # attention 후 컨텍스트 벡터 차원이 256 이 나옵니다.
        # 따라서 FC in_features=256, out_features=1 이 되어야 합니다 (fc.weight, fc.bias).
        self.fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        """
        x: (batch_size, 1, n_mels=80, time_steps) 형태의 멜 스펙트로그램 텐서
        """
        # 1) CNN 통과
        out = self.cnn(x)
        #   → out shape: (batch, 128, freq', time') 여기서 freq' = 10 (80 → 40 → 20 → 10)

        # 2) CNN 출력에서 freq 차원(차원 2)을 time 축 앞으로 옮기고, 입출력 크기를 맞춥니다.
        #    out shape 현재 예시: (batch, 128, 10, time')
        #    우리는 GRU에 (batch, time', input_size=1280) 형태로 넣어야 하므로,
        #    먼저 (batch, 128, 10, time') → permute → (batch, time', 128, 10)
        #    그다음 flatten: (batch, time', 128*10=1280)
        out = out.permute(0, 3, 1, 2)       # (batch, time', 128, 10)
        batch_size, seq_len, ch, freq_dim = out.size()  # ch=128, freq_dim=10
        out = out.contiguous().view(batch_size, seq_len, ch * freq_dim)  # (batch, time', 1280)

        # 3) GRU 통과
        #    out shape: (batch, time', hidden_size * num_directions)
        out, _ = self.gru(out)             # (batch, time', 128*2 = 256)

        # 4) Attention 계산 (Scaled Dot-Product 예시)
        #    Q, K, V: 모두 (batch, time', 256)
        Q = self.attn.query(out)           # (batch, time', 256)
        K = self.attn.key(out)             # (batch, time', 256)
        V = self.attn.value(out)           # (batch, time', 256)

        #    Scaled dot-product: attn_scores = Q @ K^T / sqrt(256)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (256 ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, time', time')
        context = torch.matmul(attn_weights, V)            # (batch, time', 256)

        # 5) 시퀀스 마지막 타임스텝(time'[-1]) 컨텍스트 벡터 추출
        #    final shape: (batch, 256)
        final = context[:, -1, :]

        # 6) 최종 FC → Sigmoid
        out = self.fc(final)               # (batch, 1)
        out = torch.sigmoid(out)           # (batch, 1) 범위 [0,1]

        return out
