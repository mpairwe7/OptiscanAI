"""
VisualLanguageGNN - Visual-Language Graph Neural Network.
Extracted from notebook cell 47. ~48M parameters.
"""

import torch
import torch.nn as nn

from src.models.vignn import MultiResolutionEncoder, SparseTopKAttention


class VisualLanguageGNN(nn.Module):
    """
    Fuses visual and text embeddings via cross-modal attention.
    Features: Multi-resolution processing, adaptive region selection, sparse attention.
    REQUIRES: clinical_knowledge_graph (ClinicalKnowledgeGraph instance)
    """

    def __init__(
        self,
        num_classes=45,
        visual_dim=384,
        text_dim=256,
        hidden_dim=384,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        clinical_knowledge_graph=None,
        backbone="vit_small_patch16_224",
        img_size=224,
    ):
        super().__init__()
        if clinical_knowledge_graph is None:
            raise ValueError("VisualLanguageGNN requires clinical_knowledge_graph parameter")
        self.clinical_knowledge_graph = clinical_knowledge_graph

        self.visual_encoder = MultiResolutionEncoder(backbone, visual_dim, img_size=img_size)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.region_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.disease_text_embed = nn.Parameter(torch.randn(num_classes, text_dim))
        nn.init.normal_(self.disease_text_embed, std=0.02)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.cross_modal_layers = nn.ModuleList(
            [
                SparseTopKAttention(hidden_dim, num_heads=num_heads, dropout=dropout, top_k=20)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        patch_features = self.visual_encoder(x)  # [B, N, D] - real patch tokens
        visual_embed = self.visual_proj(patch_features)  # [B, N, D]
        importance = self.region_importance(visual_embed)  # [B, N, 1]
        visual_embed_w = visual_embed * importance

        text_embed = self.text_proj(self.disease_text_embed).unsqueeze(0).expand(B, -1, -1)

        for cross_attn, norm in zip(self.cross_modal_layers, self.norms):
            out, _ = cross_attn(visual_embed_w, text_embed, text_embed)
            visual_embed_w = norm(visual_embed_w + out)

        fused = torch.cat([visual_embed_w.mean(dim=1), text_embed.mean(dim=1)], dim=1)
        return self.classifier(fused)
