import torch
import torch.nn as nn


class MemoryModel(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        hidden_size=256,
        num_layers=2,
        num_classes=10,
        dropout=0.3,
        max_len=100
    ):
        super(MemoryModel, self).__init__()

        # Token embedding
        self.embedding = nn.Embedding(num_classes, embed_dim)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 🔥 FIXED Attention (dimension-safe)
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),        # 256 → 256
            nn.LayerNorm(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size // 2),   # 256 → 128
            nn.ReLU(),

            nn.Linear(hidden_size // 2, 1)              # 128 → 1
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
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
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention
        attn_scores = self.attn(out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Context vector
        context = torch.sum(attn_weights * out, dim=1)  # (batch, hidden_size)

        # Output
        context = self.dropout(context)
        output = self.fc(context)

        return output