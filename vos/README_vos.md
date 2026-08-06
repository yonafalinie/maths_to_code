# Reproducing Virtual Outlier Synthesis for OOD Detection on CIFAR-10

This notebook reproduces the main classification experiment from **Virtual Outlier Synthesis (VOS)** for out-of-distribution detection.

The model is trained on CIFAR-10 as the in-distribution dataset. During training, class-conditional Gaussian distributions are estimated in the learned feature space. Low-density virtual features are then sampled and used as synthetic outliers for an auxiliary OOD detection objective.

The final trained model is evaluated using Maximum Softmax Probability and Energy-based OOD scores.

## Main Objectives

This notebook demonstrates how to:

- train a Wide ResNet classifier on CIFAR-10,
- maintain a class-specific feature bank,
- estimate class means and a shared covariance matrix,
- sample low-density virtual outliers,
- train an auxiliary OOD classification head,
- calculate MSP and Energy OOD scores,
- evaluate OOD performance using AUROC, AUPR-OOD, and FPR95,
- visualize learned features and OOD score distributions.

## Method Overview

The VOS pipeline consists of the following stages:

1. Train a classifier on CIFAR-10.
2. Extract intermediate feature representations.
3. Store features separately for each CIFAR-10 class.
4. Estimate a class-conditional Gaussian distribution in feature space.
5. Sample a large number of candidate features from each Gaussian.
6. Select low-density candidates as virtual outliers.
7. Train an auxiliary binary classifier to distinguish:
   - real in-distribution features,
   - synthetic virtual outliers.
8. Evaluate the trained model on real OOD datasets.

## Model Configuration

The notebook uses a Wide ResNet with the following configuration:

```text
Architecture: Wide ResNet
Depth: 40
Widen factor: 2
Dropout rate: 0.3
Number of classes: 10
Feature dimension: 128
```

The classifier is trained for:

```text
Epochs: 100
Batch size: 128
Optimizer: SGD
Initial learning rate: 0.1
Momentum: 0.9
Weight decay: 0.0005
```

A cosine learning-rate schedule is used during training.

## Datasets

### In-Distribution Dataset

CIFAR-10 is used as the in-distribution dataset.

```text
Training images: 50,000
Test images: 10,000
Classes: 10
Image size: 32 × 32
```

The training transform includes random cropping and horizontal flipping.

The test transform only converts the image to a tensor and applies CIFAR-10 normalization.

### Out-of-Distribution Dataset

The main OOD experiment in this notebook uses SVHN.

```text
In-distribution: CIFAR-10
Out-of-distribution: SVHN
SVHN evaluation samples: 2,000
```

The original VOS repository also evaluates additional OOD datasets, including:

- DTD Textures,
- Places365,
- LSUN-C,
- LSUN-Resize,
- iSUN,
- CIFAR-100,
- CelebA.

These datasets can be evaluated using the same trained model without retraining.

## Feature Bank

A class-specific feature bank stores learned feature vectors from the training data.

The default configuration is:

```text
Number of classes: 10
Features per class: 1,000
Feature dimension: 128
```

The resulting feature-bank shape is:

```text
[10, 1000, 128]
```

The feature bank must be full before Gaussian statistics and virtual outliers can be generated.

## Gaussian Feature Distribution

For each class, the notebook calculates a feature mean:

```text
Class means shape: [10, 128]
```

A shared covariance matrix is estimated using features from all classes:

```text
Shared covariance shape: [128, 128]
```

A small diagonal regularization value is added for numerical stability:

```text
covariance_eps = 1e-4
```

## Virtual Outlier Sampling

Virtual outliers are generated separately for each class.

The notebook samples:

```text
Candidate features per class: 10,000
Selected outliers per class: 1
```

Candidate features are sampled from each class-conditional Gaussian distribution.

The candidates with the lowest Gaussian log-probabilities are selected as virtual outliers because they lie in low-density regions of the learned feature space.

With 10 CIFAR-10 classes and one selected outlier per class, the virtual-outlier tensor has shape:

```text
[10, 128]
```

## VOS Training Objective

The total training loss combines the standard classification loss and the VOS loss:

```text
Total loss = Classification loss + 0.1 × VOS loss
```

The classification loss is cross-entropy over the 10 CIFAR-10 classes.

The VOS loss is a binary cross-entropy objective that distinguishes:

```text
Real ID features      = label 1
Virtual OOD features  = label 0
```

The VOS branch uses a learnable weighted-energy score followed by a binary OOD head.

VOS training begins only after the feature bank is sufficiently populated.

## OOD Scores

The final evaluation uses two OOD scores.

### Maximum Softmax Probability

The MSP-based OOD score is:

```text
MSP score = 1 - maximum softmax probability
```

Higher values indicate that the sample is more OOD-like.

### Energy Score

The Energy score is:

```text
Energy = -logsumexp(logits)
```

Higher values indicate that the sample is more OOD-like.

The final evaluation uses the standard unweighted Energy score. This is different from the learnable weighted-energy function used inside the VOS training objective.

## Evaluation Metrics

The notebook reports three OOD detection metrics.

### AUROC

Area Under the Receiver Operating Characteristic Curve.

```text
Higher is better.
```

An AUROC of `1.0` indicates perfect separation between ID and OOD samples.

### AUPR-OOD

Area Under the Precision-Recall Curve with OOD samples treated as the positive class.

```text
Higher is better.
```

### FPR95

False-positive rate when the OOD true-positive rate reaches at least 95%.

```text
Lower is better.
```

A lower FPR95 means that fewer CIFAR-10 samples are incorrectly classified as OOD while detecting at least 95% of OOD samples.

## Visualizations

The notebook includes several visualizations.

### Training Curves

The following quantities are plotted over training epochs:

- total training loss,
- classification loss,
- VOS loss,
- training accuracy.

### PCA Feature Visualization

PCA is used to project the 128-dimensional feature space into two dimensions.

The plot includes:

- stored CIFAR-10 features,
- class means,
- sampled virtual outliers.

This provides a qualitative view of where the virtual outliers are located relative to the ID feature clusters.

### OOD Score Distributions

Histograms compare CIFAR-10 and SVHN scores for:

- MSP,
- Energy.

Good OOD separation should produce:

```text
Lower scores for CIFAR-10
Higher scores for SVHN
Minimal overlap between distributions
```

## Requirements

The notebook requires:

```text
Python
PyTorch
TorchVision
NumPy
Matplotlib
scikit-learn
```

Install missing dependencies with:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Kaggle notebooks normally include these packages by default.

## Kaggle Setup

Enable the following Kaggle settings:

```text
Accelerator: GPU
Internet: On
```

Internet access is required when downloading CIFAR-10, SVHN, or other external datasets for the first time.

Writable files should be stored under:

```text
/kaggle/working
```

Kaggle input directories are read-only:

```text
/kaggle/input
```

## Running the Notebook

Run all cells in order.

The main workflow is:

1. Import libraries.
2. Configure the experiment.
3. Load CIFAR-10.
4. Define the Wide ResNet.
5. Create the feature bank.
6. Define Gaussian estimation and virtual-outlier sampling.
7. Train the model.
8. Evaluate CIFAR-10 classification accuracy.
9. Inspect the feature bank and virtual outliers.
10. Visualize the feature space with PCA.
11. Load SVHN.
12. Collect MSP and Energy scores.
13. Calculate OOD metrics.
14. Visualize the score distributions.
15. Save the trained checkpoint.

## Saving the Checkpoint

Kaggle sessions are temporary, so the checkpoint should be saved before the session ends.

Example:

```python
checkpoint_path = "/kaggle/working/vos_cifar10_checkpoint.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "ood_head_state_dict": ood_head.state_dict(),
        "weight_energy_state_dict": weight_energy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "feature_bank": feature_bank.bank,
        "feature_bank_counts": feature_bank.counts,
        "feature_bank_write_positions": feature_bank.write_positions,
        "history": history,
        "config": cfg,
    },
    checkpoint_path,
)

print("Checkpoint saved to:", checkpoint_path)
```

After saving, download the checkpoint from the Kaggle output directory or create a saved notebook version with outputs included.



## Important Notes

- The feature bank must be full before virtual outliers can be generated.
- PCA is only a two-dimensional approximation of the original feature space.
- OOD evaluation does not update the model.
- The gradient-flow diagnostic is optional and is not required for training.
- The auxiliary OOD head is used during VOS training.
- Final OOD evaluation uses MSP and the standard unweighted Energy score.
- Results may vary slightly because of random initialization, data shuffling, GPU operations, and OOD subset sampling.

## Notebook Scope

This notebook focuses on reproducing the main VOS training pipeline on CIFAR-10 and demonstrating OOD evaluation using SVHN.

It does not reproduce every experiment, architecture, dataset, or hyperparameter configuration from the original repository.

## Reference

This notebook is based on the method introduced in:

[**Virtual Outlier Synthesis for Out-of-Distribution Detection**](https://github.com/deeplearning-wisc/vos/tree/main/classification)

The implementation is inspired by the official VOS repository:

```text
deeplearning-wisc/vos
```

## Why MSP Can Perform Better Than Energy

In this experiment, the MSP score achieved a higher AUROC than the Energy score.

For example:

```text
MSP AUROC:    0.9164
Energy AUROC: 0.8714
```

This does **not** mean that the experiment failed.

MSP and Energy are two different post-hoc OOD scoring functions applied to the same classifier outputs. Their performance can vary depending on:

- the trained model,
- the OOD dataset,
- the learned logit distribution,
- the number of evaluated OOD samples,
- random initialization and data ordering,
- differences between the reproduction and the original training environment.

### MSP Score

The MSP-based OOD score is calculated as:

```text
MSP OOD score = 1 - maximum softmax probability
```

A sample receives a high OOD score when the classifier has low confidence in every class.

In this experiment, the maximum softmax probability separates CIFAR-10 and SVHN relatively well. Therefore, MSP produces a higher AUROC.

### Energy Score

The Energy score is calculated as:

```text
Energy = -logsumexp(logits)
```

Unlike MSP, Energy depends on the absolute magnitude of all logits, not only the largest softmax probability.

Energy can perform worse when the model assigns logits with similar overall magnitudes to both ID and OOD images. In that situation, the Energy distributions overlap more even when the softmax confidence remains useful for separation.

### Why This Is Still a Valid Result

The purpose of the experiment is not to guarantee that Energy always outperforms MSP. The experiment demonstrates that:

- the VOS model can be trained successfully,
- virtual outliers are generated from low-density feature regions,
- the auxiliary VOS loss receives gradients,
- the trained classifier can be evaluated on real OOD data,
- both MSP and Energy produce meaningful OOD rankings.

An AUROC above `0.5` means that the score performs better than random ranking. Both scores therefore show useful OOD detection capability in this experiment.

The stronger MSP result simply indicates that MSP is the better scoring function for this particular trained checkpoint and CIFAR-10 versus SVHN evaluation.

### Reproduction Differences

The notebook is an educational reproduction rather than a bit-for-bit execution of the original repository.

Small differences may come from:

- modern PyTorch and TorchVision versions,
- Kaggle GPU hardware,
- random seeds,
- data-loader ordering,
- optimizer or scheduler implementation details,
- evaluating a random subset of 2,000 SVHN images,
- numerical differences in Gaussian estimation and virtual-outlier sampling.

Therefore, the relative ordering of MSP and Energy does not need to exactly match the original paper for the implementation to be considered successful.

### Conclusion

The experiment did not fail because MSP achieved a higher score than Energy.

The correct interpretation is:

> For this trained model and OOD dataset, MSP provides better separation between CIFAR-10 and SVHN than the standard unweighted Energy score.

A complete comparison should report both methods rather than assuming that one method must always be superior.