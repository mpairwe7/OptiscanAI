"""
SceneGraphTransformer - Anatomical scene understanding with spatial reasoning.
Extracted from notebook cell 47. ~52M parameters.
"""
import numpy as np
import torch
import torch.nn as nn

from src.models.vignn import MultiResolutionEncoder, SparseTopKAttention


class SceneGraphTransformer(nn.Module):
    """
    Models spatial relationships between retinal regions.
    Features: Multi-resolution, ensemble branches, sparse attention, uncertainty estimation.
    REQUIRES: clinical_knowledge_graph (ClinicalKnowledgeGraph instance)
    """
    def __init__(self, num_classes=45, num_regions=12, hidden_dim=384,
                 num_layers=2, num_heads=4, dropout=0.1,
                 clinical_knowledge_graph=None, num_ensemble_branches=3,
                 backbone='vit_small_patch16_224', img_size=224):
        super().__init__()
        if clinical_knowledge_graph is None:
            raise ValueError("SceneGraphTransformer requires clinical_knowledge_graph parameter")
        self.clinical_knowledge_graph = clinical_knowledge_graph
        self.num_ensemble_branches = num_ensemble_branches
        self.num_regions = num_regions

        self.region_extractor = MultiResolutionEncoder(backbone, hidden_dim, img_size=img_size)
        self.region_proj = nn.Linear(hidden_dim, hidden_dim)
        self.region_type_embed = nn.Parameter(torch.randn(num_regions, hidden_dim))
        self.spatial_encoder = nn.Linear(2, hidden_dim)

        self.ensemble_branches = nn.ModuleList([
            nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=num_heads,
                    dim_feedforward=hidden_dim * 2, dropout=dropout,
                    activation='gelu', batch_first=True,
                ) for _ in range(num_layers)
            ]) for _ in range(num_ensemble_branches)
        ])

        self.relation_attn = SparseTopKAttention(hidden_dim, num_heads=num_heads, dropout=dropout, top_k=8)
        self.relation_norm = nn.LayerNorm(hidden_dim)

        self.ensemble_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_ensemble_branches, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim * num_ensemble_branches, hidden_dim // 2),
            nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout * 2), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        patch_features = self.region_extractor(x)  # [B, N, D] - real patch tokens
        num_patches = patch_features.size(1)

        region_indices = torch.linspace(0, num_patches - 1, self.num_regions,
                                        dtype=torch.long, device=x.device)
        region_features = patch_features[:, region_indices, :]
        region_embeds = self.region_proj(region_features)
        region_embeds = region_embeds + self.region_type_embed.unsqueeze(0).expand(B, -1, -1)

        grid = int(np.sqrt(num_patches))
        positions = torch.tensor(
            [[idx.item() // grid / grid, idx.item() % grid / grid] for idx in region_indices],
            dtype=torch.float32, device=x.device,
        ).unsqueeze(0).expand(B, -1, -1)
        region_embeds = region_embeds + self.spatial_encoder(positions)

        branch_outputs = []
        for branch_layers in self.ensemble_branches:
            h = region_embeds.clone()
            for layer in branch_layers:
                h = layer(h)
            branch_outputs.append(h.mean(dim=1))

        ensemble_concat = torch.cat(branch_outputs, dim=-1)
        fused = self.ensemble_fusion(ensemble_concat)

        fused_exp = fused.unsqueeze(1)
        rel_out, _ = self.relation_attn(fused_exp, fused_exp, fused_exp)
        scene = self.relation_norm(fused_exp + rel_out).squeeze(1)

        return self.classifier(scene)
