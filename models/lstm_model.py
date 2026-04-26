import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        hidden_size=256,
        num_layers=2,
        num_classes=10,
        dropout=0.3,
        max_len=100
    ):
        super(LSTMModel, self).__init__()

        # Token embedding
        self.embedding = nn.Embedding(num_classes, embed_dim)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.long()

        # Token embedding
        x = self.embedding(x)

        # Positional encoding
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos_embed = self.pos_embedding(positions)

        x = x + pos_embed

        # LSTM
        out, _ = self.lstm(x)

        # 🔥 Use full sequence (better than last step only)
        out = torch.mean(out, dim=1)

        out = self.dropout(out)
        out = self.fc(out)

        return out