"""
GraphCLIP - Graph-Enhanced CLIP for retinal disease classification.
Extracted from notebook cell 47. ~45M parameters, mobile-friendly.
"""

import torch
import torch.nn as nn

from src.models.vignn import MultiResolutionEncoder, SparseTopKAttention


class GraphCLIP(nn.Module):
    """
    GraphCLIP combines visual features with disease knowledge graphs.
    Uses sparse attention and dynamic graph learning for efficiency.
    REQUIRES: clinical_knowledge_graph (ClinicalKnowledgeGraph instance)
    """

    def __init__(
        self,
        num_classes=45,
        hidden_dim=384,
        num_graph_layers=2,
        num_heads=4,
        dropout=0.1,
        clinical_knowledge_graph=None,
        backbone="vit_small_patch16_224",
        img_size=224,
    ):
        super().__init__()
        if clinical_knowledge_graph is None:
            raise ValueError("GraphCLIP requires clinical_knowledge_graph parameter")
        self.clinical_knowledge_graph = clinical_knowledge_graph

        self.visual_encoder = MultiResolutionEncoder(backbone, hidden_dim, img_size=img_size)
        self.visual_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.disease_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))
        nn.init.normal_(self.disease_embeddings, std=0.02)

        self.graph_weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.graph_layers = nn.ModuleList(
            [
                SparseTopKAttention(hidden_dim, num_heads=num_heads, dropout=dropout, top_k=16)
                for _ in range(num_graph_layers)
            ]
        )
        self.graph_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_graph_layers)]
        )

        self.cross_attn = SparseTopKAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, top_k=24
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

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

        disease_nodes = self.disease_embeddings.unsqueeze(0).expand(B, -1, -1)
        graph_weights = self.graph_weight_generator(disease_nodes)
        graph_adj = torch.softmax(graph_weights, dim=-1)
        disease_nodes_w = torch.bmm(graph_adj, disease_nodes)

        for graph_attn, norm in zip(self.graph_layers, self.graph_norms):
            out, _ = graph_attn(disease_nodes_w, disease_nodes_w, disease_nodes_w)
            disease_nodes_w = norm(disease_nodes_w + out)

        cross_out, _ = self.cross_attn(visual_embed, disease_nodes_w, disease_nodes_w)
        visual_enhanced = self.cross_norm(visual_embed + cross_out)

        fused = torch.cat([visual_enhanced.mean(dim=1), disease_nodes_w.mean(dim=1)], dim=1)
        return self.classifier(fused)
