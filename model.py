import torch
import torch.nn as nn
import os

# ---------------- Attention 블록 ---------------- #
class Attention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key   = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):               # x: (B, T, D)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attn = self.softmax(attn)
        out  = torch.bmm(attn, V)       # (B, T, D)
        return out.mean(dim=1)          # (B, D)


# ---------------- DeepVoiceDetector ---------------- #
class DeepVoiceDetector(nn.Module):
    """
    STEP 4) 실제 모델 정의부임
    """
    def __init__(
        self,
        input_channels: int = 1,
        hidden_size: int = 128,
        sample_input_shape=(1, 400, 80),  # (C, T, F)
    ):
        super().__init__()

        # 1️⃣ CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(0.3)

        # 2️⃣ GRU 입력 크기 자동 산출
        with torch.no_grad():
            dummy = torch.zeros((1, input_channels, *sample_input_shape[1:]))
            c_out = self.cnn(dummy)                  # (1, 128, T', F')
            _, C, T, F = c_out.shape
            self.rnn_input_dim = C * F
            self.rnn_time_steps = T

        # 3️⃣ Bi-GRU + Attention + FC
        self.gru  = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = Attention(hidden_size * 2)
        self.fc   = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):               # x: (B, 1, 400, 80)
        B = x.size(0)
        x = self.cnn(x)                 # (B, 128, T', F')
        x = self.dropout(x)
        x = x.permute(0, 2, 1, 3)       # (B, T', 128, F')
        x = x.reshape(B, self.rnn_time_steps, -1)
        x, _ = self.gru(x)              # (B, T', 256)
        x = self.attn(x)                # (B, 256)
        x = self.fc(x).squeeze(1)       # (B,)
        return x


# ---------------- 모델 로더 ---------------- #
def load_model(weight_path: str, device=None):
    """
    학습된 파라미터를 불러와 평가 모드로 반환함
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepVoiceDetector().to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
