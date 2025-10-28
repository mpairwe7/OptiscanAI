"""
Multi-Label Retinal Disease Classification Models
==================================================
This script implements 3 state-of-the-art models with mathematical foundations:
1. Vision Transformer (ViT) with Multi-Label Head
2. EfficientNet-B4 with Attention Mechanism
3. Graph Neural Network for Disease Relationship Modeling

Mathematical Foundations:
-------------------------
1. FOCAL LOSS for Class Imbalance:
   FL(p_t) = -α_t(1-p_t)^γ * log(p_t)
   where p_t is the model's estimated probability for the class with label y=1
   
2. MULTI-HEAD ATTENTION:
   Attention(Q,K,V) = softmax(QK^T/√d_k)V
   
3. GRAPH CONVOLUTIONAL NETWORKS:
   H^(l+1) = σ(D^(-1/2)ÃD^(-1/2)H^(l)W^(l))
   
4. BINARY CROSS-ENTROPY with Logits:
   BCE = -∑[y*log(σ(x)) + (1-y)*log(1-σ(x))]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import timm
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             hamming_loss, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# MATHEMATICAL FOUNDATION 1: FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in multi-label classification.
    
    Mathematical Formula:
    FL(p_t) = -α_t(1-p_t)^γ * log(p_t)
    
    where:
    - p_t: predicted probability for the true class
    - α_t: weighting factor for class imbalance (0-1)
    - γ: focusing parameter (typically 2)
    
    Intuition: Down-weights easy examples and focuses on hard examples.
    When γ=0, FL reduces to standard cross-entropy loss.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) - Raw logits from model
            targets: (N, C) - Binary labels
        """
        # Convert logits to probabilities using sigmoid
        # σ(x) = 1 / (1 + e^(-x))
        probs = torch.sigmoid(inputs)
        
        # Compute binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal loss: FL = -α(1-p_t)^γ * log(p_t)
        # where log(p_t) is contained in bce_loss
        focal_weight = self.alpha * ((1 - p_t) ** self.gamma)
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# MATHEMATICAL FOUNDATION 2: ATTENTION MECHANISM
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention for capturing disease-specific features.
    
    Mathematical Formula:
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    Intuition: Different attention heads learn different aspects of the input.
    Scaling by √d_k prevents softmax saturation for large d_k.
    """
    
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear projections and split into heads
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # scores = QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply softmax to get attention weights
        # α = softmax(scores)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        # output = αV
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        output = self.W_o(attention_output)
        
        return output, attention_weights


# ============================================================================
# MODEL 1: VISION TRANSFORMER WITH MULTI-LABEL HEAD
# ============================================================================
class ViTMultiLabel(nn.Module):
    """
    Vision Transformer for Multi-Label Classification
    
    Architecture:
    1. Image Patches: Split image into patches
    2. Linear Embedding: Project patches to d_model dimensions
    3. Positional Encoding: Add position information
    4. Transformer Encoder: Apply self-attention layers
    5. Classification Head: Multi-label prediction
    
    Mathematical Foundation:
    - Patch Embedding: Linear projection of flattened patches
    - Positional Encoding: PE(pos,2i) = sin(pos/10000^(2i/d_model))
    - Layer Normalization: LN(x) = γ * (x-μ)/σ + β
    """
    
    def __init__(self, num_classes=45, img_size=224, pretrained=True):
        super(ViTMultiLabel, self).__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        
        # Get feature dimension
        num_features = self.vit.head.in_features
        
        # Replace classification head with multi-label head
        self.vit.head = nn.Identity()
        
        # Multi-label classification head with attention
        self.attention = MultiHeadAttention(num_features, num_heads=8)
        self.layer_norm = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(0.3)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_features // 2, num_classes)
        )
        
    def forward(self, x):
        # Extract features from ViT
        features = self.vit(x)  # (batch, num_features)
        
        # Reshape for attention: (batch, 1, num_features)
        features = features.unsqueeze(1)
        
        # Apply multi-head attention
        attended_features, attention_weights = self.attention(features)
        
        # Layer normalization and dropout
        attended_features = self.layer_norm(attended_features.squeeze(1))
        attended_features = self.dropout(attended_features)
        
        # Classification
        logits = self.classifier(attended_features)
        
        return logits, attention_weights


# ============================================================================
# MODEL 2: EFFICIENTNET WITH CHANNEL ATTENTION
# ============================================================================
class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation)
    
    Mathematical Formula:
    s = σ(W_2 * ReLU(W_1 * GAP(F)))
    F' = s ⊙ F
    
    where:
    - GAP: Global Average Pooling
    - σ: Sigmoid activation
    - ⊙: Element-wise multiplication
    
    Intuition: Recalibrates channel-wise feature responses by modeling
    interdependencies between channels.
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC layers with ReLU and Sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Element-wise multiplication
        return x * y.expand_as(x)


class EfficientNetMultiLabel(nn.Module):
    """
    EfficientNet-B4 with Channel Attention for Multi-Label Classification
    
    Architecture:
    1. EfficientNet Backbone: Efficient feature extraction
    2. Channel Attention: Recalibrate feature channels
    3. Global Average Pooling: Spatial dimension reduction
    4. Multi-Label Head: Disease predictions
    
    Mathematical Foundation:
    - Compound Scaling: depth = α^φ, width = β^φ, resolution = γ^φ
    - MBConv Block: Mobile Inverted Bottleneck Convolution
    - Swish Activation: f(x) = x * σ(βx)
    """
    
    def __init__(self, num_classes=45, pretrained=True):
        super(EfficientNetMultiLabel, self).__init__()
        
        # Load pretrained EfficientNet-B4
        self.efficientnet = timm.create_model('efficientnet_b4', pretrained=pretrained)
        num_features = self.efficientnet.classifier.in_features
        
        # Remove original classifier
        self.efficientnet.classifier = nn.Identity()
        
        # Add channel attention
        self.channel_attention = ChannelAttention(num_features)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-label classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_features // 2, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.efficientnet.forward_features(x)
        
        # Apply channel attention
        features = self.channel_attention(features)
        
        # Global pooling
        features = self.global_pool(features)
        features = features.flatten(1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


# ============================================================================
# MODEL 3: GRAPH NEURAL NETWORK FOR DISEASE RELATIONSHIPS
# ============================================================================
class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer
    
    Mathematical Formula:
    H^(l+1) = σ(D^(-1/2) * Ã * D^(-1/2) * H^(l) * W^(l))
    
    where:
    - Ã = A + I (adjacency matrix with self-loops)
    - D: Degree matrix
    - H^(l): Layer l features
    - W^(l): Learnable weight matrix
    - σ: Activation function
    
    Intuition: Aggregates information from neighboring nodes in the graph.
    """
    
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_features)
            adj: (num_nodes, num_nodes) - normalized adjacency matrix
        """
        # Linear transformation: H^(l) * W^(l)
        support = torch.matmul(x, self.weight)
        
        # Graph convolution: Ã * H^(l) * W^(l)
        output = torch.matmul(adj, support)
        
        return output + self.bias


class GCNMultiLabel(nn.Module):
    """
    Graph Convolutional Network for Multi-Label Disease Classification
    
    Architecture:
    1. CNN Backbone: Extract visual features
    2. Graph Construction: Build disease co-occurrence graph
    3. GCN Layers: Propagate information through disease relationships
    4. Feature Fusion: Combine visual and graph features
    5. Multi-Label Prediction: Final disease predictions
    
    Mathematical Foundation:
    - Disease Co-occurrence: P(D_i, D_j) = count(D_i ∩ D_j) / N
    - Graph Laplacian: L = D - A
    - Spectral Convolution: g_θ * x ≈ ∑ θ_k T_k(L̃)x
    """
    
    def __init__(self, num_classes=45, pretrained=True):
        super(GCNMultiLabel, self).__init__()
        
        # CNN backbone (ResNet50)
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Feature dimensions
        self.cnn_features = 2048
        self.gcn_features = 512
        
        # Spatial pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        
        # Project CNN features to GCN input
        self.feature_projection = nn.Linear(self.cnn_features, self.gcn_features)
        
        # GCN layers
        self.gc1 = GraphConvolution(self.gcn_features, self.gcn_features)
        self.gc2 = GraphConvolution(self.gcn_features, self.gcn_features)
        
        # Learnable class embeddings
        self.class_embeddings = nn.Parameter(torch.randn(num_classes, self.gcn_features))
        
        # Adjacency matrix (will be set during training based on co-occurrence)
        self.register_buffer('adjacency_matrix', torch.eye(num_classes))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.gcn_features * 2, self.gcn_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.gcn_features, 1)
        )
    
    def set_adjacency_matrix(self, adj_matrix):
        """Set the disease co-occurrence adjacency matrix"""
        self.adjacency_matrix = adj_matrix
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract CNN features
        cnn_features = self.backbone(x)
        cnn_features = self.spatial_pool(cnn_features)
        cnn_features = cnn_features.flatten(1)
        
        # Project to GCN feature space
        visual_features = self.feature_projection(cnn_features)  # (batch, gcn_features)
        
        # Prepare graph features
        class_features = self.class_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply GCN layers
        graph_features = F.relu(self.gc1(class_features, self.adjacency_matrix))
        graph_features = F.dropout(graph_features, p=0.3, training=self.training)
        graph_features = self.gc2(graph_features, self.adjacency_matrix)
        
        # Combine visual and graph features for each class
        visual_features_expanded = visual_features.unsqueeze(1).repeat(1, graph_features.size(1), 1)
        combined_features = torch.cat([visual_features_expanded, graph_features], dim=-1)
        
        # Classify each disease
        logits = self.classifier(combined_features).squeeze(-1)
        
        return logits


# ============================================================================
# DATASET CLASS
# ============================================================================
class RetinalDiseaseDataset(Dataset):
    """Dataset class for retinal disease images"""
    
    def __init__(self, labels_df, img_dir, transform=None):
        self.labels_df = labels_df
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Get disease columns
        self.disease_columns = [col for col in labels_df.columns 
                               if col not in ['ID', 'Disease_Risk', 'split']]
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Get image ID
        img_id = self.labels_df.iloc[idx]['ID']
        img_path = self.img_dir / f"{img_id}.png"
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = self.labels_df.iloc[idx][self.disease_columns].values.astype(np.float32)
        labels = torch.tensor(labels)
        
        return image, labels, img_id


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================
def compute_disease_adjacency_matrix(train_labels, disease_columns):
    """
    Compute disease co-occurrence adjacency matrix
    
    Mathematical Formula:
    A[i,j] = P(D_i ∩ D_j) / sqrt(P(D_i) * P(D_j))
    
    This is the normalized pointwise mutual information.
    """
    # Extract disease labels
    disease_matrix = train_labels[disease_columns].values
    
    # Compute co-occurrence matrix
    co_occurrence = np.dot(disease_matrix.T, disease_matrix)
    
    # Get disease frequencies
    disease_freq = disease_matrix.sum(axis=0)
    
    # Normalize by geometric mean of frequencies
    adj_matrix = np.zeros_like(co_occurrence, dtype=np.float32)
    for i in range(len(disease_columns)):
        for j in range(len(disease_columns)):
            if disease_freq[i] > 0 and disease_freq[j] > 0:
                adj_matrix[i, j] = co_occurrence[i, j] / np.sqrt(disease_freq[i] * disease_freq[j])
    
    # Add self-loops and normalize
    adj_matrix = adj_matrix + np.eye(len(disease_columns))
    
    # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    row_sum = adj_matrix.sum(axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj_matrix @ d_mat_inv_sqrt
    
    return torch.FloatTensor(adj_normalized)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for images, labels, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if isinstance(model, ViTMultiLabel):
            logits, _ = model(images)
        else:
            logits = model(images)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            if isinstance(model, ViTMultiLabel):
                logits, _ = model(images)
            else:
                logits = model(images)
            
            # Get probabilities
            probs = torch.sigmoid(logits)
            
            # Get predictions (threshold = 0.5)
            preds = (probs > 0.5).float()
            
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_probabilities.append(probs.cpu().numpy())
    
    # Concatenate all batches
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    
    # Compute metrics
    metrics = {
        'hamming_loss': hamming_loss(all_labels, all_predictions),
        'micro_f1': f1_score(all_labels, all_predictions, average='micro'),
        'macro_f1': f1_score(all_labels, all_predictions, average='macro'),
        'samples_f1': f1_score(all_labels, all_predictions, average='samples'),
    }
    
    # Compute per-class metrics
    try:
        metrics['auc_roc'] = roc_auc_score(all_labels, all_probabilities, average='macro')
    except:
        metrics['auc_roc'] = 0.0
    
    return metrics


def save_model_comparison(results, save_path='model_comparison.png'):
    """Visualize model performance comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    models = list(results.keys())
    metrics = ['micro_f1', 'macro_f1', 'samples_f1', 'auc_roc']
    
    # Bar plot for F1 scores
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics[:3]):
        values = [results[model][metric] for model in models]
        axes[0].bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    axes[0].set_xlabel('Models', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Model Performance Comparison - F1 Scores', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # AUC-ROC comparison
    auc_values = [results[model]['auc_roc'] for model in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = axes[1].bar(models, auc_values, color=colors, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('AUC-ROC Score', fontsize=12)
    axes[1].set_title('Model Performance - AUC-ROC', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RETINAL DISEASE CLASSIFICATION - MODEL TRAINING")
    print("="*80)
    
    print("\nNote: This is a template. Full training requires GPU and takes hours.")
    print("For demonstration, we'll show the model architectures and mathematical foundations.")
    
    # Model instantiation examples
    print("\n1. Vision Transformer with Multi-Label Head")
    vit_model = ViTMultiLabel(num_classes=45)
    print(f"   Parameters: {sum(p.numel() for p in vit_model.parameters()):,}")
    
    print("\n2. EfficientNet-B4 with Channel Attention")
    efficient_model = EfficientNetMultiLabel(num_classes=45)
    print(f"   Parameters: {sum(p.numel() for p in efficient_model.parameters()):,}")
    
    print("\n3. Graph Convolutional Network")
    gcn_model = GCNMultiLabel(num_classes=45)
    print(f"   Parameters: {sum(p.numel() for p in gcn_model.parameters()):,}")
    
    print("\n" + "="*80)
    print("✓ MODEL ARCHITECTURES INITIALIZED")
    print("="*80)
