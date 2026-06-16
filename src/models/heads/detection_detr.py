import torch
import torch.nn as nn

class DetectionDETR(nn.Module):
    """
    Simplified DETR head for object detection on 2D medical images.
    Expects transformer output (B, N, E) where N includes patches + cls token.
    Predicts bounding boxes and class logits.
    """
    def __init__(self, embed_dim=768, num_classes=2, num_queries=10):
        super().__init__()
        self.num_queries = num_queries
        self.class_embed = nn.Linear(embed_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)

    def forward(self, x):
        # x: (B, N, E) – full sequence
        # Use the first num_queries tokens (or learnable queries)
        # Placeholder: use mean of tokens to generate fixed predictions
        global_feat = x.mean(dim=1)  # (B, E)
        B = global_feat.shape[0]
        # Simple: predict a single box per image for demonstration
        cls_logits = self.class_embed(global_feat).unsqueeze(1)  # (B,1,num_classes+1)
        boxes = self.bbox_embed(global_feat).unsqueeze(1).sigmoid()  # (B,1,4)
        return {'pred_logits': cls_logits, 'pred_boxes': boxes}

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)