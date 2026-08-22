# Maths to Code

**Turning machine learning equations into understandable, executable PyTorch code.**

`maths_to_code` is a learning and research repository focused on translating mathematical equations from machine learning and computer vision into small, readable implementations.

The goal is not simply to reproduce a paper or call a high-level library function. Instead, each example aims to answer:

> **What does this equation actually do, and how do we write it in code?**

The notebooks move step by step from mathematical notation to tensors, matrix operations, losses, and model components.


## Repository Topics

| Topic | Main idea | Current material |
|---|---|---|
| [Vision Transformer](./vit/) | Patches, embeddings, self-attention, Transformer blocks | ViT from scratch on MNIST + detailed notes |
| [CLIP](./clip/) | Image-text representations and contrastive learning | CLIP-style implementation from scratch using FashionMNIST |
| [VOS](./vos/) | Virtual Outlier Synthesis for OOD detection | Notebook + mathematical explanation |
| [NPOS](./npos/) | Non-Parametric Outlier Synthesis for OOD detection | Mathematical and conceptual notes |



# Current Projects

## 1. Vision Transformer

Folder:

```text
vit/
├── README_Vision_Transformer.md
└── vision_transformer_from_scratch_mnist.ipynb
```

The Vision Transformer material explores how an image becomes a sequence of tokens and how those tokens are processed with self-attention.

Core ideas include:

- image patching
- patch/token embeddings
- positional embeddings
- Query, Key, and Value projections
- scaled dot-product attention
- multi-head self-attention
- Transformer encoder blocks
- classification from learned representations




## 2. CLIP

Folder:

```text
clip/
└── clip_from_scratch_fashionmnist.ipynb
```

This section explores CLIP-style contrastive representation learning.


## 3. Virtual Outlier Synthesis (VOS)

Folder:

```text
vos/
├── README_vos.md
└── vos.ipynb
```




## 4. Non-Parametric Outlier Synthesis (NPOS)

Folder:

```text
npos/
└── README_npos.md
```


The goal of this section is to understand the complete path from **feature geometry → boundary selection → synthetic outliers → OOD training objective**.






# Philosophy

The notebooks intentionally favour **clarity over abstraction**.

A one-line library implementation may be shorter, but an explicit implementation often makes the mathematics easier to understand.

The guiding question throughout the repository is:

> **Can I look at the equation, look at the code, and understand how one became the other?**



# References

Each topic folder contains its own notes and references to the relevant papers or methods.

The implementations in this repository are intended for educational and research exploration. When a notebook is based on a published method, please refer to and cite the original paper for the authoritative formulation.

---

## Author

**Yona Falinie Binti Abd Gaus**

- Portfolio: https://yonafalinie.github.io/
- GitHub: https://github.com/yonafalinie
- LinkedIn: https://www.linkedin.com/in/yona-falinie-abd-gaus-46906792/
