# Vision Transformer (ViT) From Scratch in PyTorch

## Overview

This project implements a **Vision Transformer (ViT)** from scratch using **PyTorch** and trains it on the **MNIST handwritten digit dataset**.

Vision Transformers process images as sequences of image patches. Each patch is converted into a learnable embedding vector, positional information is added, and the resulting token sequence is passed through multiple Transformer encoder blocks.

The notebook demonstrates:

- image patch extraction;
- learnable patch projection;
- learnable class token;
- positional encoding;
- multi-head self-attention;
- Transformer encoder blocks;
- MLP with GELU activation;
- residual connections;
- layer normalization;
- image classification using the final class token.

The implementation follows the architecture introduced in:

> Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.

---

## Project Files

```text
vision_transformer_from_scratch_mnist.ipynb
README.md
data/
```

The `data/` directory is created automatically when MNIST is downloaded.

---

## Architecture Overview

```text
Input Image
    │
    ▼
Resize to 32 × 32
    │
    ▼
Patch Embedding using Conv2D
    │
    ▼
Sequence of Patch Tokens
    │
    ▼
Prepend Learnable Class Token
    │
    ▼
Add Positional Encoding
    │
    ▼
Transformer Encoder Block × L
    │
    ├── Layer Normalization
    ├── Multi-Head Self-Attention
    ├── Residual Connection
    ├── Layer Normalization
    ├── Two-Layer MLP with GELU
    └── Residual Connection
    │
    ▼
Final Layer Normalization
    │
    ▼
Select Class Token
    │
    ▼
Linear Classification Head
    │
    ▼
Predicted Digit Class
```

---

## Dataset

The notebook uses the **MNIST handwritten digit dataset**, which contains ten classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The original MNIST images have shape:

```text
1 × 28 × 28
```

The notebook resizes each image to:

```text
1 × 32 × 32
```

This allows the image dimensions to be divided evenly by the selected patch size.

---

## Configuration

| Parameter | Value |
|---|---:|
| Input image size | 32 × 32 |
| Input channels | 1 |
| Patch size | 8 × 8 |
| Number of patches | 16 |
| Sequence length | 17 |
| Embedding dimension | 96 |
| Attention heads | 4 |
| Transformer layers | 4 |
| MLP expansion ratio | 4 |
| Dropout | 0.1 |
| Number of classes | 10 |
| Batch size | 128 |
| Epochs | 5 |
| Learning rate | 3e-4 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |

---

# Mathematical Formulation and Code Mapping

The following equations are from the original Vision Transformer paper and are mapped directly to the corresponding notebook sections.

## 1. Patch Embedding, Class Token, and Positional Encoding

The Transformer input is defined as:

$$
\mathbf{z}_0 =
\left[
\mathbf{x}_{\text{class}};
\mathbf{x}_p^1\mathbf{E};
\mathbf{x}_p^2\mathbf{E};
\cdots;
\mathbf{x}_p^N\mathbf{E}
\right]
+
\mathbf{E}_{\text{pos}}
\tag{1}
$$

where:

$$
\mathbf{E}\in\mathbb{R}^{(P^2C)\times D}
$$

and:

$$
\mathbf{E}_{\text{pos}}\in\mathbb{R}^{(N+1)\times D}.
$$

### Symbols

- $P$: patch height and width;
- $C$: number of image channels;
- $P^2C$: number of values in one flattened patch;
- $N$: total number of patches;
- $D$: embedding dimension;
- $\mathbf{x}_p^i$: flattened image patch $i$;
- $\mathbf{E}$: learnable patch-projection matrix;
- $\mathbf{x}_{\text{class}}$: learnable class token;
- $\mathbf{E}_{\text{pos}}$: positional embedding;
- $\mathbf{z}_0$: initial token sequence passed to the Transformer.

Equation (1) corresponds to:

1. patch embedding;
2. adding the class token;
3. adding positional information.

### Number of Patches

For image height $H$, width $W$, and square patch size $P$:

$$
N = \frac{H}{P}\times\frac{W}{P}.
$$

For this notebook:

$$
H=W=32,\qquad P=8,
$$

so:

$$
N=\frac{32}{8}\times\frac{32}{8}=4\times4=16.
$$

### Patch Projection

Each flattened patch contains:

$$
P^2C
$$

values. For an $8\times8$ grayscale patch:

$$
P^2C=8^2\times1=64.
$$

The patch projection maps:

$$
\mathbb{R}^{64}\rightarrow\mathbb{R}^{96}.
$$

Therefore:

$$
\mathbf{E}\in\mathbb{R}^{64\times96}.
$$

### Mapping to the Notebook

The notebook implements patch extraction and projection with:

```python
self.projection = nn.Conv2d(
    in_channels,
    embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

Using `kernel_size = patch_size` and `stride = patch_size` is equivalent to extracting non-overlapping patches, flattening them, and applying the same linear projection to every patch.

The forward pass is:

```python
x = self.projection(x)  # [B, D, H/P, W/P]
x = x.flatten(2)        # [B, D, N]
x = x.transpose(1, 2)   # [B, N, D]
```

This corresponds to:

$$
\mathbf{x}_p^i\mathbf{E}.
$$

### Tensor Shapes

```text
Input image:                  [B, 1, 32, 32]
After Conv2D projection:      [B, 96, 4, 4]
After flattening:             [B, 96, 16]
After transposing:            [B, 16, 96]
```

### Class Token

The class token is prepended with:

```python
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
```

The sequence changes from:

```text
[B, 16, 96]
```

to:

```text
[B, 17, 96]
```

This implements:

$$
\left[
\mathbf{x}_{\text{class}};
\mathbf{x}_p^1\mathbf{E};
\cdots;
\mathbf{x}_p^N\mathbf{E}
\right].
$$

### Positional Encoding

The notebook adds positional information using:

```python
tokens = tokens + self.positional_encoding
```

This corresponds to:

$$
+\mathbf{E}_{\text{pos}}.
$$

The notebook uses fixed sinusoidal positional encoding:

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/D}}\right),
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/D}}\right).
$$

The original ViT paper uses learnable positional embeddings. The notebook uses fixed sinusoidal encoding for educational clarity. Both methods provide information about token position.

---

## 2. Multi-Head Self-Attention Sub-Layer

The attention sub-layer is defined as:

$$
\mathbf{z}'_{\ell}=
\operatorname{MSA}\left(
\operatorname{LN}(\mathbf{z}_{\ell-1})
\right)
+
\mathbf{z}_{\ell-1},
\qquad \ell=1,\ldots,L.
\tag{2}
$$

where:

- $\mathbf{z}_{\ell-1}$ is the input to encoder layer $\ell$;
- $\operatorname{LN}$ is layer normalization;
- $\operatorname{MSA}$ is multi-head self-attention;
- the addition is a residual connection;
- $\mathbf{z}'_{\ell}$ is the attention sub-layer output.

The notebook implements Equation (2) with:

```python
x = x + self.attention(self.norm1(x))
```

| Paper notation | Notebook code |
|---|---|
| $\mathbf{z}_{\ell-1}$ | `x` |
| $\operatorname{LN}(\mathbf{z}_{\ell-1})$ | `self.norm1(x)` |
| $\operatorname{MSA}(\cdot)$ | `self.attention(...)` |
| Residual connection | `x + ...` |
| $\mathbf{z}'_{\ell}$ | updated `x` |

### Query, Key, and Value

Each token is projected into query, key, and value vectors:

$$
\mathbf{Q}=\mathbf{X}\mathbf{W}_Q,
$$

$$
\mathbf{K}=\mathbf{X}\mathbf{W}_K,
$$

$$
\mathbf{V}=\mathbf{X}\mathbf{W}_V.
$$

### Scaled Dot-Product Attention

$$
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=
\operatorname{Softmax}\left(
\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{d_h}}
\right)\mathbf{V}.
$$

The notebook computes attention scores with:

```python
attention_scores = (
    queries @ keys.transpose(-2, -1)
) * self.scale
```

where:

```python
self.scale = self.head_dim ** -0.5
```

This is equivalent to dividing by $\sqrt{d_h}$.

The attention weights are:

```python
attention_weights = attention_scores.softmax(dim=-1)
```

and the weighted value vectors are:

```python
output = attention_weights @ values
```

### Multi-Head Attention

For embedding dimension $D$ and $h$ heads:

$$
d_h=\frac{D}{h}.
$$

In the notebook:

$$
D=96,\qquad h=4,
$$

therefore:

$$
d_h=\frac{96}{4}=24.
$$

The heads are concatenated and projected:

$$
\operatorname{MSA}(\mathbf{X})=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)\mathbf{W}_O.
$$

---

## 3. MLP Sub-Layer

The second sub-layer is:

$$
\mathbf{z}_{\ell}=
\operatorname{MLP}\left(
\operatorname{LN}(\mathbf{z}'_{\ell})
\right)
+
\mathbf{z}'_{\ell},
\qquad \ell=1,\ldots,L.
\tag{3}
$$

The notebook implements Equation (3) with:

```python
x = x + self.mlp(self.norm2(x))
```

| Paper notation | Notebook code |
|---|---|
| $\mathbf{z}'_{\ell}$ | `x` after attention |
| $\operatorname{LN}(\mathbf{z}'_{\ell})$ | `self.norm2(x)` |
| $\operatorname{MLP}(\cdot)$ | `self.mlp(...)` |
| Residual connection | `x + ...` |
| $\mathbf{z}_{\ell}$ | updated `x` |

### Two-Layer MLP with GELU

The MLP is:

$$
\operatorname{MLP}(\mathbf{x})=
\mathbf{W}_2
\operatorname{GELU}\left(
\mathbf{W}_1\mathbf{x}+\mathbf{b}_1
\right)
+
\mathbf{b}_2.
$$

The first layer expands the representation:

$$
D\rightarrow D_{\text{MLP}},
$$

and the second projects it back:

$$
D_{\text{MLP}}\rightarrow D.
$$

With an MLP ratio of 4:

$$
D_{\text{MLP}}=D\times4=96\times4=384.
$$

The notebook code is:

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

### GELU

The Gaussian Error Linear Unit is:

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

A common approximation is:

$$
\operatorname{GELU}(x)\approx
\frac{x}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)
\right].
$$

---

## 4. Final Image Representation

The final image representation is:

$$
\mathbf{y}=\operatorname{LN}(\mathbf{z}_L^0).
\tag{4}
$$

where:

- $L$ is the number of Transformer layers;
- $\mathbf{z}_L^0$ is the token at index 0 after the final layer;
- index 0 is the class token;
- $\mathbf{y}$ is the final representation of the complete image.

The final sequence is:

$$
\mathbf{z}_L=
[\mathbf{z}_L^0,\mathbf{z}_L^1,\ldots,\mathbf{z}_L^N].
$$

The notebook performs:

```python
x = self.encoder(x)
x = self.final_norm(x)
return x[:, 0]
```

Because layer normalization acts independently on each token, this is equivalent to:

```python
x = self.encoder(x)
class_token = x[:, 0]
class_features = self.final_norm(class_token)
return class_features
```

### Classification Head

The class-token representation is mapped to class logits:

$$
\hat{\mathbf{y}}=
\mathbf{W}_{\text{head}}\mathbf{y}+
\mathbf{b}_{\text{head}}.
$$

The notebook uses:

```python
return self.classifier(cls_features)
```

This maps:

$$
\mathbb{R}^{96}\rightarrow\mathbb{R}^{10}.
$$

---

## Complete Equation-to-Code Mapping

| ViT equation | Purpose | Notebook component |
|---|---|---|
| Equation (1) | Patch projection, class token, and positional encoding | `PatchEmbedding` and `TokenAndPositionEmbedding` |
| Equation (2) | LayerNorm, multi-head self-attention, and residual connection | `x = x + self.attention(self.norm1(x))` |
| Equation (3) | LayerNorm, MLP with GELU, and residual connection | `x = x + self.mlp(self.norm2(x))` |
| Equation (4) | Final normalized class-token representation | `self.final_norm(x)` and `x[:, 0]` |

---

## Complete Tensor-Shape Flow

```text
Input images:                         [B, 1, 32, 32]
After Conv2D patch projection:        [B, 96, 4, 4]
After flattening:                     [B, 96, 16]
After transposing:                    [B, 16, 96]
After adding class token:             [B, 17, 96]
After positional encoding:            [B, 17, 96]
After every Transformer block:        [B, 17, 96]
After selecting class token:          [B, 96]
After classification head:            [B, 10]
```

---

## Transformer Encoder Block

The complete block combines Equations (2) and (3):

```python
def forward(self, x):
    x = x + self.attention(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
```

```text
Input tokens z_(l-1)
        │
        ▼
Layer Normalization
        │
        ▼
Multi-Head Self-Attention
        │
        ▼
Residual Addition
        │
        ▼
Intermediate tokens z'_l
        │
        ▼
Layer Normalization
        │
        ▼
Two-Layer MLP with GELU
        │
        ▼
Residual Addition
        │
        ▼
Output tokens z_l
```

The notebook uses the **pre-normalization** Transformer configuration because layer normalization is applied before attention and before the MLP.

---

## Loss Function

The notebook uses cross-entropy loss:

$$
\mathcal{L}=
-\frac{1}{B}
\sum_{i=1}^{B}
\log\left(
\frac{\exp(s_{i,y_i})}
{\sum_{c=1}^{K}\exp(s_{i,c})}
\right).
$$

where:

- $B$ is the batch size;
- $K$ is the number of classes;
- $s_{i,c}$ is the logit for sample $i$ and class $c$;
- $y_i$ is the correct class.

The PyTorch implementation is:

```python
criterion = nn.CrossEntropyLoss()
```

The model returns raw logits with shape:

```text
[B, 10]
```

No Softmax layer is included during training because `CrossEntropyLoss` internally applies LogSoftmax.

---

## Training Procedure

For each batch:

1. Load images and labels.
2. Move them to the selected device.
3. Clear previous gradients.
4. Run the forward pass.
5. Calculate cross-entropy loss.
6. Perform backpropagation.
7. Update model parameters.
8. Calculate accuracy.

```python
optimizer.zero_grad(set_to_none=True)

logits = model(batch_images)
loss = criterion(logits, batch_labels)

loss.backward()
optimizer.step()
```

---

## Evaluation

During evaluation, gradient calculation is disabled and predictions are selected using:

```python
predictions = logits.argmax(dim=1)
```

Accuracy is:

$$
\text{Accuracy}=
\frac{\text{Number of correct predictions}}
{\text{Total number of predictions}}.
$$

---

## Model Checkpoint

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": vars(config),
    "test_accuracy": test_accuracy,
}

torch.save(
    checkpoint,
    "vit_mnist_from_scratch.pt",
)
```

The checkpoint stores:

- trained model parameters;
- model configuration;
- final test accuracy.

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy jupyter
```

---

## Running the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
vision_transformer_from_scratch_mnist.ipynb
```

Alternatively:

```bash
jupyter lab
```

The notebook can also be opened in Visual Studio Code using a Python environment that contains PyTorch, TorchVision, NumPy, and Matplotlib.

---

## Hardware

The notebook automatically uses a CUDA GPU when available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Otherwise, it runs on the CPU.

---

## Key Learning Outcomes

After completing this notebook, you should understand:

- how an image is divided into patches;
- how Conv2D performs patch extraction and projection;
- how patch embeddings become tokens;
- why a class token is added;
- why positional information is required;
- how queries, keys, and values are created;
- how scaled dot-product attention works;
- how multi-head attention captures different patch relationships;
- how residual connections improve optimization;
- how pre-normalization is used;
- how the MLP uses two linear layers and GELU;
- how Equations (1)–(4) map to the code;
- how the final class token represents the whole image;
- how ViT is trained for image classification.

---

## Implementation Notes

### Conv2D Versus Explicit Patch Extraction

These operations are mathematically equivalent when the Conv2D kernel and stride equal the patch size:

```text
Extract patches → Flatten patches → Shared Linear projection
```

and:

```text
Conv2D(kernel_size=patch_size, stride=patch_size)
```

### Positional-Encoding Difference

The original ViT architecture uses learnable positional embeddings. The notebook uses fixed sinusoidal positional encoding.

A learnable alternative is:

```python
self.positional_embedding = nn.Parameter(
    torch.zeros(
        1,
        num_patches + 1,
        embed_dim,
    )
)
```

and:

```python
tokens = tokens + self.positional_embedding
```

### Educational Scale

The original ViT models were trained on much larger datasets and images. This notebook uses a compact architecture so it can be understood and trained on modest hardware.

---

## References

1. Alexey Dosovitskiy et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR, 2021.
2. Ashish Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017.
3. Dan Hendrycks and Kevin Gimpel. *Gaussian Error Linear Units (GELUs).* 2016.
4. PyTorch Documentation.
5. TorchVision MNIST Documentation.
6. Matt Nguyen. *Building a Vision Transformer Model From Scratch.* Medium.

---

## License

This implementation is intended for educational and research purposes. Add the license appropriate for your repository, such as MIT or Apache 2.0.
