# 학습 모델 정의
# CNN + Bi-GRU + Attention
import torch
import torch.nn as nn


# Attention 블록 정의
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x: (B, T, D)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = self.softmax(torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5))
        out = torch.bmm(attn_weights, V)  # (B, T, D)
        return out.mean(dim=1)  # (B, D)

# DeepVoiceDetector with automatic GRU input size inference
class DeepVoiceDetector(nn.Module):
    def __init__(self, input_channels=1, hidden_size=128, sample_input_shape=(1, 400, 80)):
        super(DeepVoiceDetector, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.3)

        # GRU 입력 크기 자동 추론
        with torch.no_grad():
            dummy_input = torch.zeros((1, input_channels, *sample_input_shape[1:]))  # (1, 1, 400, 80)
            cnn_out = self.cnn(dummy_input)  # (1, 128, T', F') 예상
            print("CNN output shape:", cnn_out.shape)
            _, c, t, f = cnn_out.shape
            self.rnn_input_dim = c * f
            self.rnn_time_steps = t

        self.gru = nn.GRU(input_size=self.rnn_input_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.attn = Attention(input_dim=hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):  # x: (B, 1, 400, 80)
      batch_size = x.size(0)
      x = self.cnn(x)                    # (B, 128, 50, 10)
      x = self.dropout(x)
      x = x.permute(0, 2, 1, 3)          # (B, 50, 128, 10)
      x = x.reshape(batch_size, 50, -1)  # (B, 50, 1280)
      out, _ = self.gru(x)               # (B, 50, 256)
      out = self.attn(out)               # (B, 256)
      out = self.fc(out).squeeze(1)      # (B,)
      return out


