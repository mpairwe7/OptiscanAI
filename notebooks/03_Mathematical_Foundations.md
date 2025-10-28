# Mathematical Foundations for Retinal Disease Classification
## A Comprehensive Technical Documentation

---

## Table of Contents
1. [Problem Formulation](#problem-formulation)
2. [Multi-Label Classification Framework](#multi-label-classification)
3. [Deep Learning Architectures](#architectures)
4. [Loss Functions](#loss-functions)
5. [Attention Mechanisms](#attention)
6. [Graph Neural Networks](#gnn)
7. [Optimization](#optimization)
8. [Evaluation Metrics](#metrics)

---

## 1. Problem Formulation {#problem-formulation}

### 1.1 Dataset Definition

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be our dataset where:
- $x_i \in \mathbb{R}^{H \times W \times 3}$ represents a retinal fundus image
- $y_i \in \{0,1\}^C$ is a binary label vector for $C=45$ disease classes
- $N$ is the total number of samples

### 1.2 Multi-Label Classification Task

Given an image $x$, we aim to learn a function:

$$f: \mathbb{R}^{H \times W \times 3} \rightarrow [0,1]^C$$

that predicts the probability of each disease class:

$$\hat{y} = \sigma(f_\theta(x))$$

where $\sigma$ is the sigmoid function and $\theta$ represents learnable parameters.

---

## 2. Multi-Label Classification Framework {#multi-label-classification}

### 2.1 Sigmoid Activation

Unlike softmax (used in single-label classification), we use sigmoid for each class independently:

$$\sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$

This allows multiple diseases to be predicted simultaneously, as diseases often co-occur.

### 2.2 Decision Threshold

For prediction, we apply a threshold $\tau$ (typically 0.5):

$$\hat{y}_i = \begin{cases} 
1 & \text{if } \sigma(z_i) \geq \tau \\
0 & \text{otherwise}
\end{cases}$$

### 2.3 Multi-Label vs Multi-Class

**Key Difference:**

| Aspect | Multi-Class | Multi-Label |
|--------|-------------|-------------|
| Output Activation | Softmax: $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | Sigmoid: $\frac{1}{1+e^{-z_i}}$ |
| Constraint | $\sum_i p_i = 1$ | Independent probabilities |
| Example | "Cat" OR "Dog" | "Diabetes" AND "Hypertension" |

---

## 3. Deep Learning Architectures {#architectures}

### 3.1 Convolutional Neural Networks (CNNs)

#### 3.1.1 Convolution Operation

The convolution operation for layer $l$ is defined as:

$$h_{ij}^{(l+1)} = \sigma\left(\sum_{a=0}^{k-1}\sum_{b=0}^{k-1} w_{ab}^{(l)} \cdot h_{(i+a)(j+b)}^{(l)} + b^{(l)}\right)$$

where:
- $w_{ab}^{(l)}$ are learnable filter weights
- $k \times k$ is the kernel size
- $b^{(l)}$ is the bias term
- $\sigma$ is a non-linear activation (ReLU, Swish, etc.)

#### 3.1.2 Batch Normalization

Normalizes activations to stabilize training:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

where $\gamma$ and $\beta$ are learnable parameters.

### 3.2 Vision Transformer (ViT)

#### 3.2.1 Patch Embedding

Divide image into patches and project to embedding space:

$$z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \ldots; x_p^N E] + E_{\text{pos}}$$

where:
- $x_p^i \in \mathbb{R}^{P^2 \cdot 3}$ is the $i$-th flattened patch
- $E \in \mathbb{R}^{(P^2 \cdot 3) \times D}$ is the embedding matrix
- $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are positional embeddings

#### 3.2.2 Transformer Encoder

Each encoder layer applies:

1. **Multi-Head Self-Attention (MSA):**

$$\text{MSA}(z) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(zW_i^Q, zW_i^K, zW_i^V)$$

2. **Feed-Forward Network (FFN):**

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

Complete encoder block:

$$z'_l = \text{MSA}(\text{LN}(z_l)) + z_l$$

$$z_{l+1} = \text{FFN}(\text{LN}(z'_l)) + z'_l$$

### 3.3 EfficientNet

#### 3.3.1 Compound Scaling

EfficientNet scales depth, width, and resolution jointly:

$$\text{depth: } d = \alpha^\phi$$
$$\text{width: } w = \beta^\phi$$
$$\text{resolution: } r = \gamma^\phi$$

subject to: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ and $\alpha \geq 1, \beta \geq 1, \gamma \geq 1$

#### 3.3.2 MBConv Block

Mobile Inverted Bottleneck Convolution:

1. **Expansion:** $1 \times 1$ convolution to increase channels
2. **Depthwise:** $3 \times 3$ or $5 \times 5$ depthwise convolution
3. **Squeeze-Excitation:** Channel attention
4. **Projection:** $1 \times 1$ convolution to reduce channels

#### 3.3.3 Swish Activation

$$\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$$

---

## 4. Loss Functions {#loss-functions}

### 4.1 Binary Cross-Entropy Loss

For multi-label classification:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{C}\sum_{i=1}^C \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

where:
- $y_i \in \{0,1\}$ is the true label
- $\hat{y}_i = \sigma(z_i)$ is the predicted probability

### 4.2 Focal Loss

Addresses class imbalance by down-weighting easy examples:

$$\mathcal{L}_{\text{FL}} = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

where:

$$p_t = \begin{cases} 
\hat{y}_i & \text{if } y_i = 1 \\
1-\hat{y}_i & \text{otherwise}
\end{cases}$$

**Parameters:**
- $\alpha_t \in [0,1]$: class weighting factor
- $\gamma \geq 0$: focusing parameter (typically 2)

**Intuition:**
- When $\gamma = 0$, Focal Loss reduces to standard BCE
- As $\gamma$ increases, easy examples (high $p_t$) are down-weighted
- Hard examples (low $p_t$) receive more attention

**Gradient Analysis:**

$$\frac{\partial \mathcal{L}_{\text{FL}}}{\partial z_i} = \alpha_t [(1-p_t)^\gamma p_t - \gamma (1-p_t)^{\gamma-1} p_t \log(p_t)](1-p_t)$$

### 4.3 Asymmetric Loss

Designed specifically for positive-negative imbalance:

$$\mathcal{L}_{\text{ASL}} = \mathcal{L}_+ + \mathcal{L}_-$$

where:

$$\mathcal{L}_+ = -(1-p)^{\gamma_+} \log(p)$$

$$\mathcal{L}_- = -p^{\gamma_-} \log(1-p) \cdot m$$

with $m = \max(p - m_{\text{threshold}}, 0)$ to suppress easy negatives.

---

## 5. Attention Mechanisms {#attention}

### 5.1 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix
- $K \in \mathbb{R}^{m \times d_k}$: Key matrix
- $V \in \mathbb{R}^{m \times d_v}$: Value matrix
- $d_k$: dimension of keys/queries

**Scaling Factor $\sqrt{d_k}$:**

Prevents softmax saturation. Without scaling, dot products can grow large:

$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d_k$$

Scaling normalizes variance to 1.

### 5.2 Multi-Head Attention

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Projection Matrices:**
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

### 5.3 Channel Attention (Squeeze-and-Excitation)

**Squeeze:** Global Average Pooling

$$s_c = \frac{1}{H \times W}\sum_{i=1}^H\sum_{j=1}^W x_c(i,j)$$

**Excitation:** Two FC layers

$$\mathbf{s} = \sigma(W_2 \delta(W_1 \mathbf{s}))$$

where $\delta$ is ReLU and $\sigma$ is sigmoid.

**Scale:**

$$\tilde{x}_c = s_c \cdot x_c$$

---

## 6. Graph Neural Networks {#gnn}

### 6.1 Graph Representation

Model disease relationships as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, A)$:
- $\mathcal{V}$: set of disease nodes (vertices)
- $\mathcal{E}$: set of co-occurrence relationships (edges)
- $A \in \mathbb{R}^{C \times C}$: adjacency matrix

### 6.2 Disease Co-occurrence Matrix

Compute from training data:

$$A_{ij} = \frac{\text{count}(D_i \cap D_j)}{\sqrt{\text{count}(D_i) \times \text{count}(D_j)}}$$

This is the **normalized pointwise mutual information**.

### 6.3 Graph Convolutional Layer

#### 6.3.1 Spectral Convolution

Based on spectral graph theory:

$$g_\theta \star x = U g_\theta U^T x$$

where $U$ is the eigenvector matrix of the graph Laplacian $L = D - A$.

#### 6.3.2 Simplified GCN (Kipf & Welling)

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$

where:
- $\tilde{A} = A + I_N$ (adjacency with self-loops)
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ (degree matrix)
- $H^{(l)} \in \mathbb{R}^{N \times d^{(l)}}$ (node features at layer $l$)
- $W^{(l)} \in \mathbb{R}^{d^{(l)} \times d^{(l+1)}}$ (learnable weights)

#### 6.3.3 Message Passing Framework

Can be viewed as:

1. **Message:** $m_{ij}^{(l)} = W^{(l)}h_j^{(l)}$
2. **Aggregate:** $\tilde{h}_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} m_{ij}^{(l)}$
3. **Update:** $h_i^{(l+1)} = \sigma(\tilde{h}_i^{(l+1)})$

where $\alpha_{ij} = \frac{1}{\sqrt{d_i d_j}}$ is the normalization coefficient.

### 6.4 Graph Attention Networks (GAT)

Learn attention weights for neighbors:

$$h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)} W^{(l)}h_j^{(l)}\right)$$

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}(i)}\exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_k]))}$$

---

## 7. Optimization {#optimization}

### 7.1 Stochastic Gradient Descent (SGD)

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

### 7.2 Adam Optimizer

Combines momentum and adaptive learning rates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Hyperparameters:**
- $\beta_1 = 0.9$ (momentum decay)
- $\beta_2 = 0.999$ (variance decay)
- $\eta = 10^{-4}$ (learning rate)
- $\epsilon = 10^{-8}$ (numerical stability)

### 7.3 Learning Rate Scheduling

#### Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right)$$

---

## 8. Evaluation Metrics {#metrics}

### 8.1 Hamming Loss

Fraction of incorrect labels:

$$\text{Hamming Loss} = \frac{1}{N \times C}\sum_{i=1}^N\sum_{j=1}^C \mathbb{1}[y_{ij} \neq \hat{y}_{ij}]$$

**Lower is better.** Range: $[0, 1]$

### 8.2 F1 Score

Harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Multi-label variants:**

1. **Micro-F1:** Aggregate over all instance-label pairs

$$F_1^{\text{micro}} = \frac{2\sum_{j=1}^C TP_j}{2\sum_{j=1}^C TP_j + \sum_{j=1}^C FP_j + \sum_{j=1}^C FN_j}$$

2. **Macro-F1:** Average of per-class F1 scores

$$F_1^{\text{macro}} = \frac{1}{C}\sum_{j=1}^C F_{1,j}$$

3. **Sample-F1:** Average F1 per sample

$$F_1^{\text{sample}} = \frac{1}{N}\sum_{i=1}^N \frac{2|y_i \cap \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$

### 8.3 Area Under ROC Curve (AUC-ROC)

For each class, compute:

$$\text{AUC}_j = \int_0^1 \text{TPR}_j(\text{FPR}) \, d(\text{FPR})$$

where:
- $\text{TPR} = \frac{TP}{TP + FN}$ (True Positive Rate)
- $\text{FPR} = \frac{FP}{FP + TN}$ (False Positive Rate)

**Multi-label aggregation:**

$$\text{AUC}^{\text{macro}} = \frac{1}{C}\sum_{j=1}^C \text{AUC}_j$$

### 8.4 Average Precision (AP)

$$\text{AP} = \sum_k (R_k - R_{k-1})P_k$$

where $P_k$ and $R_k$ are precision and recall at threshold $k$.

---

## 9. Explainable AI Techniques

### 9.1 Grad-CAM (Gradient-weighted Class Activation Mapping)

$$\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

where:
- $y^c$ is the score for class $c$
- $A^k$ is the activation map of channel $k$
- $\alpha_k^c$ is the importance weight

### 9.2 Attention Visualization

Visualize attention weights $\alpha_{ij}$ to understand which image regions influence predictions.

---

## 10. Summary of Key Equations

| Concept | Equation |
|---------|----------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| **Focal Loss** | $\mathcal{L}_{\text{FL}} = -\alpha_t(1-p_t)^\gamma\log(p_t)$ |
| **Self-Attention** | $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ |
| **GCN Layer** | $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$ |
| **Batch Norm** | $\hat{x} = \gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$ |
| **Adam Update** | $\theta_{t+1} = \theta_t - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ |
| **Micro-F1** | $F_1^{\text{micro}} = \frac{2\sum TP}{2\sum TP + \sum FP + \sum FN}$ |

---

## References

1. Lin et al. (2017). "Focal Loss for Dense Object Detection"
2. Vaswani et al. (2017). "Attention Is All You Need"
3. Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks"
4. Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition"
5. Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

---

**Document Version:** 1.0  
**Last Updated:** October 2025  
**Author:** AI Research Team
