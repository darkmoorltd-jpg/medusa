import torch
import torch.nn as nn
import timm
from einops import rearrange

class MedusaEncoder(nn.Module):
    """
    Shared Vision Transformer backbone for MEDUSA.
    Uses timm's standard ViT (with its own class token) and adds
    a learnable modality embedding to the classification output.
    """
    def __init__(
        self,
        img_size_2d=224,
        img_size_3d=128,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_modalities=6,
        use_3d=False
    ):
        super().__init__()
        self.use_3d = use_3d

        # Modality embedding
        self.modality_embed = nn.Embedding(num_modalities, embed_dim)

        if not use_3d:
            # Standard timm ViT (class_token=True by default)
            self.vit = timm.create_model(
                'vit_base_patch16_224', pretrained=False,
                img_size=img_size_2d, in_chans=in_chans,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            )
        else:
            # 3D patch embedding + transformer blocks
            self.patch_embed = nn.Conv3d(
                in_chans, embed_dim,
                kernel_size=patch_size, stride=patch_size
            )
            num_patches = (img_size_3d // patch_size) ** 3
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.blocks = nn.Sequential(*[
                timm.models.vision_transformer.Block(
                    dim=embed_dim, num_heads=num_heads
                ) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, modality_id):
        """
        x: (B, C, H, W) or (B, C, D, H, W)
        modality_id: (B,) tensor of ints
        Returns: (B, N+1, embed_dim) where token 0 is the class token.
        """
        B = x.shape[0]
        if not self.use_3d:
            # Let timm handle everything (patch embed + class token + pos embed + blocks + norm)
            x = self.vit.forward_features(x)   # (B, 197, embed_dim)
        else:
            # Manual 3D pathway
            x = self.patch_embed(x)                         # (B, E, D', H', W')
            x = rearrange(x, 'b e d h w -> b (d h w) e')    # (B, num_patches, E)
            cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, E)
            x = torch.cat((cls_tokens, x), dim=1)           # (B, num_patches+1, E)
            x = x + self.pos_embed
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

        # Add modality embedding to the class token (position 0)
        modality_emb = self.modality_embed(modality_id).unsqueeze(1)  # (B,1,E)
        x[:, 0] = x[:, 0] + modality_emb.squeeze(1)
        return x