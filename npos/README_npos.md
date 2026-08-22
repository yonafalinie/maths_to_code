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
Y_{\text{in}} =
\{\text{cat},\text{dog},\text{car}\}.
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
\boxed{
\begin{aligned}
\text{Training objective} &= \text{classification objective} \\
&\quad + \text{uncertainty objective}
\end{aligned}
}
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
      /             \
     |      ●●       |
     |   ●      ●    |
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

NPOS wants the \(B\)-type points.

Then it samples around them:

```text
                    ×

               ×    ●    ×

                    ×
```

where

- \(●\) = boundary ID feature,
- \(×\) = candidate synthetic outlier.

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

Its \(k\)-th nearest neighbour is further away.

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

Gaussian sampling around \(B\) might generate:

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
\boxed{\text{NPOS FPR95}=5.76\%}.
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
L=\{x:P_{\text{in}}(x)\le\beta\},
$$

which initially looks abstract but is actually the mathematical foundation for everything they do in Section 3.
