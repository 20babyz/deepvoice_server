# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    간단한 Scaled Dot-Product Attention 구현 예시
    - input_dim: GRU 양방향 출력 차원 (hidden_size * 2)
    """
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key   = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        x: (batch, time_steps, input_dim)
        """
        # 1) Q, K, V 생성
        Q = self.query(x)  # (batch, time_steps, input_dim)
        K = self.key(x)    # (batch, time_steps, input_dim)
        V = self.value(x)  # (batch, time_steps, input_dim)

        # 2) Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)         # (batch, time_steps, time_steps)
        context = torch.matmul(weights, V)              # (batch, time_steps, input_dim)

        # 3) 마지막 타임스텝 컨텍스트 벡터 리턴
        return context[:, -1, :]                        # (batch, input_dim)


class DeepVoiceDetector(nn.Module):
    """
    DeepVoiceModel 대신 이 클래스를 쓰도록 변경했습니다.
    GRU 입력 차원은 dummy input을 통해 자동으로 계산됩니다.
    입력 형태: (batch, 1, 400, 80)  ←  utils.py가 이 형태를 만들어 줍니다.
    """

    def __init__(self, input_channels=1, hidden_size=128, sample_input_shape=(1, 1, 400, 80)):
        super(DeepVoiceDetector, self).__init__()

        # ── 1) CNN 블록 ──
        # 입력 채널(input_channels=1), 출력 채널 32 → 64 → 128
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),     # (1,32,200,40)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),     # (1,64,100,20)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)     # (1,128,50,10)
        )
        self.dropout = nn.Dropout(0.3)

        # ── 2) GRU 입력 차원 자동 계산 ──
        with torch.no_grad():
            dummy = torch.zeros(sample_input_shape)  # (1,1,400,80)
            cnn_out = self.cnn(dummy)               # (1,128,50,10)
            _, c, t, f = cnn_out.shape              # c=128, t=50, f=10
            self.rnn_input_dim = c * f              # 128 * 10 = 1280
            self.rnn_time_steps = t                 # 50

        # ── 3) GRU 블록 (bidirectional=True) ──
        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # ── 4) Attention ──
        # GRU 출력 차원 = hidden_size * 2 ( = 256 )
        self.attn = Attention(input_dim=hidden_size * 2)

        # ── 5) 최종 FC ──
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        """
        x: (batch, 1, 400, 80)
        """
        batch_size = x.size(0)

        # 1) CNN 통과 → (batch, 128, 50, 10)
        out = self.cnn(x)
        out = self.dropout(out)

        # 2) RNN 입력 형태로 변환
        #    (batch, 128, 50, 10) → permute → (batch, 50, 128, 10) → reshape → (batch, 50, 128*10=1280)
        out = out.permute(0, 2, 1, 3)              # (batch, 50, 128, 10)
        out = out.contiguous().view(batch_size, self.rnn_time_steps, self.rnn_input_dim)  # (batch,50,1280)

        # 3) GRU 통과 → (batch, 50, hidden_size*2=256)
        out, _ = self.gru(out)

        # 4) Attention → (batch, 256)
        out = self.attn(out)

        # 5) FC → Sigmoid → (batch,)
        out = self.fc(out).squeeze(1)              # (batch,)
        out = torch.sigmoid(out)                   # (batch,)
        return out
