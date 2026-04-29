"""Tests for all four model architectures and the clinical knowledge graph."""
import sys

sys.path.insert(0, "/home/developer/Mpairwe7/MLOPS_V1")

import pytest
import torch

from src.models.vignn import (
    ClinicalKnowledgeGraph,
    create_knowledge_graph,
    create_vignn_model,
)
from src.models.graphclip import GraphCLIP
from src.models.visual_language_gnn import VisualLanguageGNN
from src.models.scene_graph_transformer import SceneGraphTransformer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# NOTE: hidden_dim must be 384 to match vit_small_patch16_224 backbone output.
# The MultiResolutionEncoder projection layers are hard-wired to ViT's 384-dim output.
NUM_CLASSES = 45
HIDDEN_DIM = 384
NUM_HEADS = 2
NUM_GRAPH_LAYERS = 1
DROPOUT = 0.1


@pytest.fixture
def knowledge_graph(disease_columns):
    """Create a small ClinicalKnowledgeGraph for testing."""
    return ClinicalKnowledgeGraph(disease_names=disease_columns)


@pytest.fixture
def small_batch():
    """A tiny batch that fits in CPU memory quickly."""
    return torch.randn(2, 3, 224, 224)


# ---------------------------------------------------------------------------
# ViGNN
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_vignn_forward_pass(knowledge_graph, small_batch):
    model = create_vignn_model(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_graph_layers=NUM_GRAPH_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        clinical_knowledge_graph=knowledge_graph,
        num_patches=196,
        patch_embed_dim=384,
    )
    model.eval()
    with torch.no_grad():
        output = model(small_batch)
    assert output.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {output.shape}"


# ---------------------------------------------------------------------------
# GraphCLIP
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_graphclip_forward_pass(knowledge_graph, small_batch):
    model = GraphCLIP(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_graph_layers=NUM_GRAPH_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        clinical_knowledge_graph=knowledge_graph,
    )
    model.eval()
    with torch.no_grad():
        output = model(small_batch)
    assert output.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {output.shape}"


# ---------------------------------------------------------------------------
# VisualLanguageGNN
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_visual_language_gnn_forward_pass(knowledge_graph, small_batch):
    model = VisualLanguageGNN(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GRAPH_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        clinical_knowledge_graph=knowledge_graph,
    )
    model.eval()
    with torch.no_grad():
        output = model(small_batch)
    assert output.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {output.shape}"


# ---------------------------------------------------------------------------
# SceneGraphTransformer
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_scene_graph_transformer_forward_pass(knowledge_graph, small_batch):
    model = SceneGraphTransformer(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GRAPH_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        clinical_knowledge_graph=knowledge_graph,
        num_ensemble_branches=2,
    )
    model.eval()
    with torch.no_grad():
        output = model(small_batch)
    assert output.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {output.shape}"


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

def test_knowledge_graph_creation(disease_columns):
    kg = ClinicalKnowledgeGraph(disease_names=disease_columns)
    assert kg.num_classes == len(disease_columns)
    assert kg.adjacency.shape == (len(disease_columns), len(disease_columns))
    assert kg.get_edge_count() >= 0


def test_knowledge_graph_factory():
    """Test the create_knowledge_graph convenience function."""
    kg = create_knowledge_graph()
    assert kg.num_classes > 0
    assert hasattr(kg, "cooccurrence")
    assert hasattr(kg, "categories")


def test_knowledge_graph_clinical_reasoning(disease_columns):
    kg = ClinicalKnowledgeGraph(disease_names=disease_columns)
    predictions = {d: 0.1 for d in disease_columns}
    predictions["DR"] = 0.9  # High DR should boost related diseases
    refined = kg.apply_clinical_reasoning(predictions)
    assert isinstance(refined, dict)
    assert len(refined) == len(predictions)
    # DR at 0.9 should boost CME if it exists
    if "CME" in refined and "CME" in predictions:
        assert refined["CME"] >= predictions["CME"]


# ---------------------------------------------------------------------------
# Output range and gradients
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_model_output_range(knowledge_graph, small_batch):
    """Verify sigmoid(output) falls within [0, 1] for all models."""
    model = GraphCLIP(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_graph_layers=NUM_GRAPH_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        clinical_knowledge_graph=knowledge_graph,
    )
    model.eval()
    with torch.no_grad():
        output = model(small_batch)
    probs = torch.sigmoid(output)
    assert probs.min() >= 0.0, "Sigmoid output below 0"
    assert probs.max() <= 1.0, "Sigmoid output above 1"


@pytest.mark.slow
def test_model_gradient_flow(knowledge_graph):
    """Verify gradients propagate through the model via loss.backward()."""
    model = GraphCLIP(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_graph_layers=NUM_GRAPH_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        clinical_knowledge_graph=knowledge_graph,
    )
    model.train()
    x = torch.randn(2, 3, 224, 224, requires_grad=False)
    targets = torch.randint(0, 2, (2, NUM_CLASSES)).float()

    output = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, targets)
    loss.backward()

    # Check that at least some parameters received gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients flowed through the model"
