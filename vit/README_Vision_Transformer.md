# Vision Transformer (ViT) From Scratch in PyTorch

An educational implementation of a **Vision Transformer (ViT)** built from scratch in PyTorch and trained on the **MNIST handwritten-digit dataset**.

The notebook connects the mathematical formulation in the original ViT paper to the corresponding PyTorch implementation, including patch embedding, positional information, multi-head self-attention, the GELU MLP, residual connections, and classification using the final class token.

## Notebook

```text
vision_transformer_from_scratch_mnist.ipynb
```

## Main Components

- Conv2D-based patch extraction and projection
- Learnable class token
- Positional encoding
- Four-head multi-head self-attention
- Pre-normalization Transformer encoder blocks
- Two-layer MLP with GELU
- Residual connections and dropout
- Final class-token classification head

---

## Architecture

```text
Input image: [B, 1, 32, 32]
          │
          ▼
Conv2D patch embedding
kernel = 8, stride = 8
          │
          ▼
Patch tokens: [B, 16, 96]
          │
          ▼
Prepend learnable [CLS] token
          │
          ▼
Token sequence: [B, 17, 96]
          │
          ▼
Add positional encoding
          │
          ▼
Transformer encoder × 4
  ├─ LayerNorm
  ├─ 4-head self-attention
  ├─ Residual connection
  ├─ LayerNorm
  ├─ MLP: 96 → 384 → 96
  └─ Residual connection
          │
          ▼
Final LayerNorm
          │
          ▼
Select [CLS] token: [B, 96]
          │
          ▼
Linear classifier: [B, 10]
```

---

## Dataset

The notebook uses MNIST, which contains ten classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The original `1 × 28 × 28` grayscale images are resized to `1 × 32 × 32` so that the spatial dimensions are divisible by the patch size.

---

## Configuration

| Parameter | Value |
|---|---:|
| Image size | 32 × 32 |
| Input channels | 1 |
| Patch size | 8 × 8 |
| Number of patches | 16 |
| Sequence length | 17 |
| Embedding dimension | 96 |
| Attention heads | 4 |
| Dimension per head | 24 |
| Transformer layers | 4 |
| MLP hidden dimension | 384 |
| Dropout | 0.1 |
| Number of classes | 10 |
| Batch size | 128 |
| Epochs | 5 |
| Learning rate | 3e-4 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

---

# ViT Paper Equations Mapped to Code

GitHub renders the equations below using fenced `math` blocks. This is more reliable in README files than `\[ ... \]` or numbered `\tag{}` expressions.

## Equation 1: Transformer Input

```math
\mathbf{z}_0 =
\left[
\mathbf{x}_{\mathrm{class}};
\mathbf{x}_p^1\mathbf{E};
\mathbf{x}_p^2\mathbf{E};
\cdots;
\mathbf{x}_p^N\mathbf{E}
\right]
+
\mathbf{E}_{\mathrm{pos}}
```

```math
\mathbf{E}\in\mathbb{R}^{(P^2C)\times D},
\qquad
\mathbf{E}_{\mathrm{pos}}\in\mathbb{R}^{(N+1)\times D}
```

This equation combines three operations:

1. project every image patch into a `D`-dimensional token;
2. prepend the learnable class token;
3. add positional information.

### Number of patches

For an image of height `H`, width `W`, and square patch size `P`:

```math
N = \frac{H}{P}\times\frac{W}{P}
```

For this notebook:

```math
N = \frac{32}{8}\times\frac{32}{8}=16
```

Each grayscale patch contains:

```math
P^2C = 8^2\times1=64
```

It is projected into a 96-dimensional embedding:

```math
\mathbb{R}^{64}\rightarrow\mathbb{R}^{96}
```

### Code mapping: patch embedding

```python
self.projection = nn.Conv2d(
    in_channels=1,
    out_channels=96,
    kernel_size=8,
    stride=8,
)
```

Using a convolution whose kernel size and stride equal the patch size is equivalent to:

```text
extract each patch → flatten it → apply a shared Linear layer
```

The forward pass is:

```python
x = self.projection(x)  # [B, 96, 4, 4]
x = x.flatten(2)        # [B, 96, 16]
x = x.transpose(1, 2)   # [B, 16, 96]
```

This implements the patch projection term:

```math
\mathbf{x}_p^i\mathbf{E}
```

### Code mapping: class token

```python
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
x = torch.cat([cls_tokens, x], dim=1)
```

The tensor shape changes from:

```text
[B, 16, 96] → [B, 17, 96]
```

### Code mapping: positional information

```python
x = x + self.positional_encoding
```

The notebook uses fixed sinusoidal positional encoding:

```math
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/D}}\right)
```

```math
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/D}}\right)
```

> **Implementation note:** The original ViT paper uses learnable positional embeddings. This notebook uses fixed sinusoidal positional encoding for educational clarity. Both methods provide token-position information.

---

## Equation 2: Multi-Head Self-Attention Sub-Layer

```math
\mathbf{z}'_{\ell}
=
\mathrm{MSA}\!\left(
\mathrm{LN}(\mathbf{z}_{\ell-1})
\right)
+
\mathbf{z}_{\ell-1},
\qquad \ell=1,\ldots,L
```

This is a **pre-normalization** attention block:

```text
input → LayerNorm → Multi-Head Self-Attention → residual addition
```

### Code mapping

```python
x = x + self.attention(self.norm1(x))
```

| Paper notation | Code |
|---|---|
| `z_(l-1)` | `x` before attention |
| `LN(z_(l-1))` | `self.norm1(x)` |
| `MSA(...)` | `self.attention(...)` |
| residual addition | `x + ...` |
| `z'_l` | updated `x` |

<details>
<summary><strong>Query, key, and value equations</strong></summary>

Each normalized token is projected into query, key, and value representations:

```math
\mathbf{Q}=\mathbf{X}\mathbf{W}_Q,
\qquad
\mathbf{K}=\mathbf{X}\mathbf{W}_K,
\qquad
\mathbf{V}=\mathbf{X}\mathbf{W}_V
```

The learnable matrices `W_Q`, `W_K`, and `W_V` allow the model to create different representations for matching tokens and aggregating information.

</details>

<details>
<summary><strong>Scaled dot-product attention</strong></summary>

```math
\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
=
\mathrm{Softmax}\!\left(
\frac{\mathbf{Q}\mathbf{K}^{\mathsf{T}}}{\sqrt{d_h}}
\right)\mathbf{V}
```

where `d_h` is the feature dimension of one attention head.

The notebook implements the attention scores as:

```python
attention_scores = (
    queries @ keys.transpose(-2, -1)
) * self.scale

self.scale = self.head_dim ** -0.5
```

Multiplication by `head_dim ** -0.5` is equivalent to division by:

```math
\sqrt{d_h}
```

The attention weights and output are then calculated with:

```python
attention_weights = attention_scores.softmax(dim=-1)
output = attention_weights @ values
```

</details>

<details>
<summary><strong>Four-head tensor shapes</strong></summary>

The notebook uses:

```math
D=96,
\qquad h=4,
\qquad d_h=\frac{D}{h}=24
```

The input attention tensor has shape:

```text
[B, 17, 96]
```

After splitting into four heads, each of query, key, and value has shape:

```text
[B, 4, 17, 24]
```

The four head outputs are concatenated:

```math
4\times24=96
```

so the attention output returns to:

```text
[B, 17, 96]
```

</details>

### Multi-head attention equation

```math
\mathrm{MSA}(\mathbf{X})
=
\mathrm{Concat}
\left(
\mathrm{head}_1,
\ldots,
\mathrm{head}_h
\right)
\mathbf{W}_O
```

Each head can learn a different relationship among the class token and image-patch tokens.

---

## Equation 3: MLP Sub-Layer

```math
\mathbf{z}_{\ell}
=
\mathrm{MLP}\!\left(
\mathrm{LN}(\mathbf{z}'_{\ell})
\right)
+
\mathbf{z}'_{\ell},
\qquad \ell=1,\ldots,L
```

The second sub-layer follows this sequence:

```text
attention output → LayerNorm → two-layer MLP → residual addition
```

### Code mapping

```python
x = x + self.mlp(self.norm2(x))
```

| Paper notation | Code |
|---|---|
| `z'_l` | `x` after attention |
| `LN(z'_l)` | `self.norm2(x)` |
| `MLP(...)` | `self.mlp(...)` |
| residual addition | `x + ...` |
| `z_l` | updated `x` |

### Two-layer MLP with GELU

```math
\mathrm{MLP}(\mathbf{x})
=
\mathbf{W}_2\,
\mathrm{GELU}\!\left(
\mathbf{W}_1\mathbf{x}+\mathbf{b}_1
\right)
+
\mathbf{b}_2
```

The notebook implements it as:

```python
self.mlp = nn.Sequential(
    nn.Linear(embed_dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, embed_dim),
    nn.Dropout(dropout),
)
```

The dimensional flow is:

```text
96 → 384 → GELU → 96
```

<details>
<summary><strong>GELU definition</strong></summary>

```math
\mathrm{GELU}(x)=x\Phi(x)
```

A commonly used approximation is:

```math
\mathrm{GELU}(x)
\approx
\frac{x}{2}
\left[
1+
\tanh\!\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)
\right]
```

GELU smoothly weights the input rather than setting every negative value exactly to zero.

</details>

---

## Equation 4: Final Image Representation

```math
\mathbf{y}=\mathrm{LN}(\mathbf{z}_L^0)
```

After the final encoder layer, the sequence is:

```math
\mathbf{z}_L=
\left[
\mathbf{z}_L^0,
\mathbf{z}_L^1,
\ldots,
\mathbf{z}_L^N
\right]
```

The token at index `0`, written as `z_L^0`, is the class token. It is used as the global image representation.

### Code mapping

```python
x = self.encoder(x)
x = self.final_norm(x)
cls_features = x[:, 0]
```

Because `LayerNorm` is applied independently to each token, normalizing the complete sequence and then selecting token zero is equivalent to normalizing the class token itself.

The classifier produces one logit per MNIST class:

```python
logits = self.classifier(cls_features)  # [B, 10]
```

```math
\widehat{\mathbf{y}}
=
\mathbf{W}_{\mathrm{head}}\mathbf{y}
+
\mathbf{b}_{\mathrm{head}}
```

---

## Complete Equation-to-Code Mapping

| ViT equation | Purpose | Main code |
|---|---|---|
| Equation 1 | Patch projection, class token, positional information | `PatchEmbedding`, `torch.cat`, positional addition |
| Equation 2 | LayerNorm, four-head self-attention, residual connection | `x = x + self.attention(self.norm1(x))` |
| Equation 3 | LayerNorm, two-layer GELU MLP, residual connection | `x = x + self.mlp(self.norm2(x))` |
| Equation 4 | Final normalized class-token representation | `self.final_norm(x)` and `x[:, 0]` |

---

## Complete Tensor-Shape Flow

| Stage | Shape |
|---|---|
| Input images | `[B, 1, 32, 32]` |
| Conv2D patch projection | `[B, 96, 4, 4]` |
| Flattened patch tokens | `[B, 96, 16]` |
| Transposed patch sequence | `[B, 16, 96]` |
| Class token added | `[B, 17, 96]` |
| Positional encoding added | `[B, 17, 96]` |
| Transformer output | `[B, 17, 96]` |
| Final class-token representation | `[B, 96]` |
| Classification logits | `[B, 10]` |

---

## Transformer Encoder Block

The complete pre-normalization encoder block is:

```python
def forward(self, x):
    x = x + self.attention(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
```

```text
Input z_(l-1)
      │
      ▼
LayerNorm → Multi-Head Self-Attention
      │
      ▼
Residual addition → z'_l
      │
      ▼
LayerNorm → MLP with GELU
      │
      ▼
Residual addition → z_l
```

---

## Loss Function

The notebook uses cross-entropy loss:

```math
\mathcal{L}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log\left(
\frac{\exp(s_{i,y_i})}
{\sum_{c=1}^{K}\exp(s_{i,c})}
\right)
```

```python
criterion = nn.CrossEntropyLoss()
```

The model returns raw logits rather than probabilities because `CrossEntropyLoss` applies the required log-softmax operation internally.

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install torch torchvision numpy matplotlib jupyter
```

---

## Run the Notebook

```bash
jupyter notebook vision_transformer_from_scratch_mnist.ipynb
```

Before publishing changes, verify the notebook by using **Restart Kernel and Run All Cells**.

---

## Model Checkpoint

A checkpoint can be saved with:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": vars(config),
    "test_accuracy": test_accuracy,
}

torch.save(checkpoint, "vit_mnist_from_scratch.pt")
```

Large model checkpoints should normally be excluded from Git using `.gitignore` or stored with Git LFS.

---

## Key Learning Outcomes

This notebook demonstrates:

- how images are converted into sequences of patch tokens;
- why Conv2D can implement patch extraction and linear projection;
- how class and positional tokens are incorporated;
- how queries, keys, and values are used in attention;
- how four attention heads divide a 96-dimensional embedding;
- how residual connections and pre-normalization are implemented;
- how the two-layer GELU MLP processes every token;
- how the final class token is used for image classification;
- how the principal ViT equations map directly to PyTorch code.

---

## References

1. Alexey Dosovitskiy et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR, 2021.
2. Ashish Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017.
3. Dan Hendrycks and Kevin Gimpel. *Gaussian Error Linear Units (GELUs).* 2016.
4. PyTorch documentation.
5. TorchVision MNIST documentation.

---

## License

MIT