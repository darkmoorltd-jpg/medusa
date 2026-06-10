import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=14, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, embed_dim) – cls token
        x = self.norm(x)
        x = self.dropout(x)
        return self.fc(x)