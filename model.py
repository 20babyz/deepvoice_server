# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────── 1) Attention 블록 정의 ────────────
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.query   = nn.Linear(input_dim, input_dim)
        self.key     = nn.Linear(input_dim, input_dim)
        self.value   = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: (batch, time_steps, input_dim)
        Q, K, V 모두 (batch, time_steps, input_dim)
        attn_weights: (batch, time_steps, time_steps)
        out: (batch, time_steps, input_dim)
        반환: (batch, input_dim)  ← time_steps 차원을 평균(mean)하여 리턴
        """
        Q = self.query(x)                          # (B, T, D)
        K = self.key(x)                            # (B, T, D)
        V = self.value(x)                          # (B, T, D)

        # Scaled dot-product attention
        scores       = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)  # (B, T, T)
        attn_weights = self.softmax(scores)                                 # (B, T, T)
        out          = torch.bmm(attn_weights, V)                           # (B, T, D)

        # time_steps 차원을 평균하여 (B, D)로 반환
        return out.mean(dim=1)                                             # (B, D)


# ──────────── 2) DeepVoiceDetector 정의 ────────────
class DeepVoiceDetector(nn.Module):
    """
    - 입력: (batch, 1, 400, 80)
    - CNN → (batch, 128, 50, 10)
    - reshape → (batch, 50, 1280)
    - GRU(bidirectional) → (batch, 50, 256)
    - Attention → (batch, 256)
    - FC → Sigmoid → (batch,)
    """
    def __init__(self, input_channels=1, hidden_size=128, sample_input_shape=(1, 1, 400, 80)):
        super(DeepVoiceDetector, self).__init__()

        # ── 1) CNN 블록 ──
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
            dummy    = torch.zeros(sample_input_shape)  # (1, 1, 400, 80)
            cnn_out  = self.cnn(dummy)                 # (1, 128, 50, 10)
            _, c, t, f = cnn_out.shape                  # c=128, t=50, f=10
            self.rnn_input_dim  = c * f                  # 128 * 10 = 1280
            self.rnn_time_steps = t                      # 50

        # ── 3) GRU ──
        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # ── 4) Attention ──
        # GRU 출력 차원 = hidden_size * 2
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

        # 2) (batch, 128, 50, 10) → (batch, 50, 128, 10) → (batch, 50, 1280)
        out = out.permute(0, 2, 1, 3)  # (B, 50, 128, 10)
        out = out.contiguous().view(batch_size, self.rnn_time_steps, self.rnn_input_dim)

        # 3) GRU → (batch, 50, hidden_size*2=256)
        out, _ = self.gru(out)

        # 4) Attention → (batch, 256)
        out = self.attn(out)

        # 5) FC → Sigmoid → (batch,)
        out = self.fc(out).squeeze(1)   # (batch,)
        out = torch.sigmoid(out)        # (batch,)
        return out
