# Section 1 — Introduction
Great. We’ll start with **Section 1: Introduction**, and we’ll read it the way you would when preparing to really understand or reproduce the method—not just summarize it.

## Section 1 — Introduction
The introduction is doing three things:

1. defining the OOD problem,

2. explaining the weakness of previous outlier-synthesis methods,

3. motivating the specific design of NPOS.

### Paragraph 1: Why OOD detection matters
The paper starts with the standard open-world problem: a classifier is trained on a known distribution $P_{\text{in}}$, but once deployed it can encounter samples from unknown classes. A conventional classifier has no explicit mechanism saying “I have never seen anything like this,” so it may still assign one of the known labels with high confidence.

For example, suppose training contains

$$
Y_{\text{in}} = \{\text{cat},\text{dog},\text{car}\}.
$$

At test time you provide an airplane.

A normal classifier is still forced to output something like:

$$
P(\text{car}\mid x)=0.91.
$$

OOD detection adds another question:

$$
\boxed{\text{Does this sample belong to the known distribution at all?}}
$$

So there are really two tasks:

$$
\text{classification}
$$

and

$$
\text{OOD detection}.
$$

The authors explicitly argue that a useful model should do both.

---
## Paragraph 2: Their training philosophy
This is an important paragraph because it introduces the basic mathematical structure used later.

The authors propose jointly optimizing:

$$
\text{ID classification}
$$

and

$$
\text{ID/OOD separation}.
$$

In other words, rather than training an ordinary classifier and adding an OOD score afterward, they want **OOD awareness to affect training itself**.

Conceptually:

$$
\boxed{\begin{aligned}\text{Training objective} &= \text{classification objective} \\\\ &\quad + \text{uncertainty objective}\end{aligned}}
$$

Later these become:

$$
R_{\text{closed}}

+

\alpha R_{\text{open}}.
$$

The introduction already describes $R_{\text{open}}$ conceptually as a form of **level-set estimation** that separates ID from OOD.

We'll discuss level sets properly in Section 2/3, but intuitively think of it as drawing a contour around the dense region occupied by ID data:

```text

             OOD

          ┌─────────┐

          │ ● ● ● ● │

     OOD  │ ● ● ● ● │  OOD

          │ ● ● ●   │

          └─────────┘

              ID

```

You want the model to learn something resembling that boundary.

---
# The fundamental problem
The obvious difficulty is:

> How do we train an ID-vs-OOD separator if we don't actually have OOD examples?

This is really the central problem of the whole paper.

The training set contains only

$$
D_{\text{in}} = \{(x_i, y_i)\}_{i=1}^{n}
$$

There is no $D_{\text{out}}$.


One option would be to obtain some external dataset and call it OOD.

But then your detector may become dependent on whichever OOD dataset you selected.

Instead, NPOS creates **synthetic OOD examples** from the ID feature space itself.

---
# Paragraph 3: Previous solution — VOS
The paper then points to VOS, which also synthesizes artificial outliers.

VOS works in **feature space**, rather than trying to synthesize realistic OOD images.

That distinction is important.

Instead of:

$$
\text{generate a strange image}
$$

they generate:

$$
\text{a strange feature vector}.
$$

The paper says this is more tractable than synthesizing directly in the input space.

So if the network produces

$$
h(x)\in\mathbb{R}^d,
$$

VOS models the distribution of these features.

Very roughly, VOS assumes:

$$
h(x)\mid y=c

\sim

\mathcal N(\mu_c,\Sigma).
$$

Then it generates samples from low-density regions of that Gaussian model.

---
# Why do the NPOS authors dislike this?
Because real neural-network features may not actually follow Gaussian distributions.

Imagine a particular class has embeddings shaped like this:

```text

        ● ●

      ●     ●

     ●       ●

     ●

      ●       ●

        ● ● ●

```

This distribution could be curved, multimodal, asymmetric, irregular, etc.

VOS might approximate it using something like:

```text

        Gaussian ellipse

       ┌───────────┐

      /             \\

     \|      ●●       |

     \|   ●      ●    |

      \             /

       └───────────┘

```

The approximation may not capture the true geometry.

The authors call the class-conditional Gaussian assumption **strong and restrictive**, particularly for complex embedding distributions.

This is the exact gap NPOS is designed to address.

---
# Paragraph 4: NPOS's central idea
Now we reach the main contribution.

The authors propose **Non-Parametric Outlier Synthesis**.

The key claim is:

$$
\boxed{

\text{Don't assume a global probability distribution for ID embeddings.}

}
$$

Instead, use the actual geometry of observed embeddings.

They identify **low-likelihood ID embeddings close to the boundary** and then “spray” synthetic points around them.

This wording—**spray around the boundary**—is probably the easiest way to remember the method.

Suppose ID features look approximately like:

```text

           ●

       ● ● ● ●

     ● ● ● ● ● ●

      ● ● ● ● ●

        ● ● ●

```

Some points are deep inside:

```text

       ● ●

      ● X ●

       ● ●

```

while some are near the edge:

```text

                    B

                    ●

             ● ● ●

           ● ● ● ●

```

NPOS wants the \\(B\\)-type points.

Then it samples around them:

```text

                    ×

               ×    ●    ×

                    ×

```

where

- \\(●\\) = boundary ID feature,

- \\(×\\) = candidate synthetic outlier.

---
# How does NPOS know which ID points are near the boundary?
This is where **k-nearest-neighbour distance** enters.

The authors use nearest-neighbour distance as a non-parametric estimate of local density.

Consider two points.

### Interior point
```text

        ● ●

       ● A ●

        ● ●

```

The $k$-th nearest neighbour is fairly close.

Therefore:

$$
d_k(A)\text{ is small}.
$$

Interpretation:

$$
\text{high local density}.
$$

### Boundary point
```text

 B

          ● ●

         ● ● ●

```

Its \\(k\\)-th nearest neighbour is further away.

Therefore:

$$
d_k(B)\text{ is large}.
$$

Interpretation:

$$
\text{lower local density}.
$$

So NPOS essentially uses

$$
d_k(z,Z)
$$

as an inverse density indicator:

$$
d_k \uparrow

\Rightarrow

P_{\text{in}}(z)\downarrow.
$$

It isn't literally computing the probability density $P_{\text{in}}$ ; it is using neighbour geometry as a surrogate.

That is the **non-parametric** part.

---
# Then comes Gaussian sampling
Once they find a boundary feature $h(x_i)$, they generate candidates:

$$
v

\sim

\mathcal N(h(x_i),\sigma^2I).
$$

This can seem contradictory at first:

> “If the method is non-parametric, why is there a Gaussian?”

Because the Gaussian is not being used as a model for

$$
P_{\text{in}}.
$$

Instead it is merely a **local perturbation mechanism**.

There's a major difference between:

### VOS
$$
P_{\text{in}}(z)

\approx

\mathcal N(\mu,\Sigma)
$$

Global distribution assumption.

and

### NPOS
$$
v = z_{\text{boundary}} + \epsilon,

\qquad

\epsilon \sim \mathcal{N}(0,\sigma^2 I)
$$


Local random perturbation.

So NPOS can still truthfully be described as non-parametric.

---
# But some sampled points will be bad
Suppose this is the ID distribution:

```text

          ● ● ●

        ● ● ● ● ●

       ● ● ● ● ● ●

         ● ● ●

```

Take a boundary point on the right:

```text

          ● ● ●

        ● ● ● ● B

       ● ● ● ● ●

```

Gaussian sampling around \\(B\\) might generate:

```text

               x1

          ● ● ●

        ● ● ● x2 B

       ● ● ● ● ●

               x3

```

Here x1 and x3 are plausibly OOD.

But x2 moved back into the ID region.

So NPOS cannot simply call every perturbation OOD.

It performs another density check and keeps candidates with **large kNN distance**.

The paper describes this as rejection sampling: low-likelihood synthetic samples are retained while unsuitable candidates are rejected.

---
# The two-loss idea introduced in Section 1
The introduction also makes clear that NPOS doesn't only synthesize OOD points.

It simultaneously improves the structure of the ID representation.

The authors want samples belonging to each class to cluster around their class prototype.

So one loss effectively says:

$$
\boxed{

\text{make ID classes compact and distinguishable}

}
$$

while another says:

$$
\boxed{

\text{separate ID features from synthetic OOD features}

}
$$

These become:

$$
R_{\text{closed}}
$$

and

$$
R_{\text{open}}.
$$

This interaction is important:

```text

R_closed

   │

   ▼

better ID features

   │

   ▼

better boundary identification

   │

   ▼

better synthetic OOD points

   │

   ▼

R_open

   │

   ▼

better ID/OOD boundary

```

So the two parts reinforce one another.

---
# What Figure 1 is really telling you
Figure 1 on page 2 is almost the entire paper compressed into one diagram.

It has three stages:

### (a) ID embeddings
First create useful, compact ID representations.

```text

       ● ●

     ● ● ● ●

      ● ● ●

```

### (b) Boundary samples
Use kNN distance to identify low-density ID points:

```text

       B

     ● ● ●

   B ● ● ● B

      ● ●

```

### (c) Non-parametric outlier synthesis
Perturb around those points and retain low-density candidates:

```text

    ×               ×

       ● ● ●

     ● ● ● ●

      ● ● ●

               ×

```

The resulting synthetic outliers provide supervision for learning an ID/OOD boundary.

---
# Headline experimental claim in the Introduction
The authors emphasize one particularly dramatic comparison using CLIP-B/16.

With the same pretrained model and similar fine-tuning setting:

$$
\text{MSP FPR95}=41.87\%
$$

versus

$$
\text{NPOS FPR95}=5.76\%.
$$

They also report that NPOS improves over VOS by **13.40 percentage points in FPR95**, which they use as evidence that non-parametric synthesis itself matters.

We'll inspect whether that comparison is fully convincing when we reach the experiments.

---
# The three contributions
The introduction finishes with essentially three claims.

**First:** NPOS synthesizes artificial OOD examples without assuming a parametric distribution for ID embeddings.

**Second:** the authors give a theoretical interpretation of the procedure in terms of **rejection sampling and level-set estimation**.

**Third:** they test the method extensively, including ImageNet-scale experiments and ablations.

---
## What you should retain from Section 1
Before moving on, the mental model I want you to have is:

$$
\boxed{

\text{Real ID embeddings}

\xrightarrow{\text{kNN}}

\text{boundary ID points}

\xrightarrow{\text{Gaussian perturb}}

\text{candidate outliers}

\xrightarrow{\text{kNN filter}}

\text{virtual OOD}

}
$$

and then

$$
\boxed{

R_{\text{closed}}

+

\alpha R_{\text{open}}

}
$$

trains the network.

The **new idea isn't simply “generate Gaussian noise.”** The novelty is where they generate it from and how they decide which generated features count as useful outliers:

$$
\boxed{\text{non-parametric boundary estimation using nearest neighbours}}.
$$

The next section, **Section 2: Preliminaries**, is short but important because it introduces the **lower level set**

$$
L=\{x\:P_{\text{in}}(x)\le\beta\},
$$

which initially looks abstract but is actually the mathematical foundation for everything they do in Section 3.

---

# Reproduction Results — CIFAR-100 OOD Evaluation

This section records the OOD evaluation results from our NPOS reproduction using **CIFAR-100 as the in-distribution (ID) dataset** and a **ResNet-34-style encoder**.

## Evaluation setup

- **ID feature bank:** CIFAR-100 training set, 50,000 images.
- **ID evaluation set:** CIFAR-100 test set, 10,000 images.
- **Feature used for OOD detection:** 512-dimensional penultimate encoder representation.
- **Feature normalization:** L2 normalization with `F.normalize(..., dim=1)`.
- **OOD score:** negative distance to the \(K\)-th nearest neighbour in the CIFAR-100 training feature bank.
- **FAISS index:** `faiss.IndexFlatL2`.
- **KNN setting:** \(K=300\), matching `test_npos_cifar100.sh` in the original NPOS repository.
- **Metrics:** FPR95, AUROC, and AUPR. Lower FPR95 is better; higher AUROC/AUPR is better.

The score is

$$
s(x) = -d_K(x),
$$

where \(d_K(x)\) is the squared Euclidean distance to the \(K\)-th nearest CIFAR-100 training feature. Therefore, a **higher score means more ID-like**, while a more negative score indicates a sample is farther from the ID feature manifold.

## OOD datasets used

To make the comparison as close as possible to the original NPOS setup, the OOD datasets were downloaded from the same or equivalent official sources used by the original repository.

| Dataset | Evaluation source/setup | Number of OOD samples |
|---|---|---:|
| SVHN | Official `test_32x32.mat`; randomly sampled to 10,000 | 10,000 |
| Textures | Official DTD archive, full `dtd/images` folder | 5,640 |
| Places365 | Official `test_256.tar`; random subset with seed 42 | 10,000 |
| LSUN-C | NPOS README-linked LSUN archive; resized from 36×36 to 32×32 for our CIFAR model | 10,000 |
| iSUN | NPOS README-linked iSUN archive | 8,925 |

For CIFAR-100 normalization we used

$$
\mu=(0.5071, 0.4867, 0.4408),
\qquad
\sigma=(0.2675, 0.2565, 0.2761).
$$

## Our OOD results

| OOD dataset | Mean OOD KNN score | FPR95 ↓ | AUROC ↑ | AUPR ↑ |
|---|---:|---:|---:|---:|
| SVHN | -0.6808 | **12.93** | **97.29** | **97.14** |
| Textures | -0.4175 | 50.99 | 89.47 | 94.02 |
| Places365 | -0.3284 | 76.98 | 74.35 | 73.76 |
| LSUN-C | -0.4767 | 33.37 | 92.52 | 92.52 |
| iSUN | -0.4160 | 42.94 | 90.73 | 91.92 |
| **Average** | — | **43.44** | **88.87** | **89.87** |

The mean CIFAR-100 ID score with \(K=300\) was

$$
\boxed{-0.2504}.
$$

## Comparison with the original NPOS results

The table below compares our reproduction with the reported NPOS CIFAR-100 / ResNet-34 benchmark values.

| OOD dataset | Our FPR95 ↓ | Original NPOS FPR95 ↓ | Difference | Our AUROC ↑ | Original NPOS AUROC ↑ | Difference |
|---|---:|---:|---:|---:|---:|---:|
| SVHN | **12.93** | 17.98 | **-5.05** | **97.29** | 96.43 | **+0.86** |
| Places365 | **76.98** | 80.41 | **-3.43** | **74.35** | 73.74 | **+0.61** |
| LSUN-C | 33.37 | **28.90** | +4.47 | 92.52 | **92.99** | -0.47 |
| iSUN | **42.94** | 43.50 | **-0.56** | **90.73** | 89.56 | **+1.17** |
| Textures | 50.99 | **33.07** | +17.92 | 89.47 | **92.86** | -3.39 |
| **Average** | 43.44 | **40.77** | +2.67 | 88.87 | **89.12** | -0.25 |

Overall, the reproduced average AUROC is very close to the original NPOS result:

$$
88.87 \quad \text{vs.} \quad 89.12.
$$

The average FPR95 is also reasonably close:

$$
43.44 \quad \text{vs.} \quad 40.77.
$$

The largest remaining discrepancy is on **Textures**, where our FPR95 is 50.99 compared with 33.07 in the original NPOS result.

## Dataset checks and reproduction notes

### SVHN

Using the official Stanford `test_32x32.mat` file produced the same result as the earlier torchvision-based evaluation. This confirms that the SVHN dataset source was not responsible for any discrepancy.

### Textures

Using the official DTD archive and evaluating all 5,640 images produced the same result as the previous DTD evaluation:

$$
\text{FPR95}=50.99,\qquad
\text{AUROC}=89.47.
$$

Therefore, the remaining Textures gap is more likely related to the learned feature representation or training reproduction than to the dataset loader.

### Places365

Using the official Places365 `test_256.tar` test set and randomly sampling 10,000 images produced:

$$
\text{FPR95}=76.98,\qquad
\text{AUROC}=74.35.
$$

This is close to the original NPOS result and also close to our earlier torchvision-based Places365 evaluation.

### LSUN-C

The LSUN-C archive linked by the NPOS README contains **36×36** images. Feeding the 36×36 images directly into our encoder gave very poor OOD separation:

| LSUN-C preprocessing | FPR95 ↓ | AUROC ↑ |
|---|---:|---:|
| Original 36×36 images | 94.87 | 62.58 |
| CenterCrop to 32×32 | 34.06 | 92.78 |
| Resize to 32×32 | **33.37** | 92.52 |

Because CIFAR-100 is 32×32, the final reproduction uses

```python
transforms.Resize((32, 32))
```

before normalization. This was essential for obtaining an LSUN-C result close to the original NPOS benchmark.

### iSUN

The downloaded iSUN benchmark contains 8,925 images at 32×32 resolution. The resulting performance was:

$$
\text{FPR95}=42.94,\qquad
\text{AUROC}=90.73,
$$

which is very close to the original NPOS result.

## Current conclusion

The reproduction is broadly consistent with the original NPOS CIFAR-100 benchmark. The average AUROC differs by only **0.25 percentage points**, while the average FPR95 differs by **2.67 percentage points**.

The main unresolved difference is **Textures**. Future investigation should therefore focus on differences between our training implementation and the original NPOS representation learning setup rather than on the OOD dataset source itself.

