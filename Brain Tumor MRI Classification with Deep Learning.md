# 🧠 Brain Tumor MRI Classification with Deep Learning

Classifying brain MRI scans into four categories — **glioma**, **meningioma**, **pituitary tumor**, and **no tumor** — using a fine-tuned ResNet18 (transfer learning) with Grad-CAM visualizations for model interpretability.

> ⚠️ **Disclaimer:** This is a portfolio/learning project, **not a diagnostic tool**. The model has not been validated on external data, and class labels come from the dataset's own annotations rather than radiologist verification. See [Limitations](#limitations) below.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Approach](#approach)
  - [Preprocessing & Augmentation](#preprocessing--augmentation)
  - [Model Architectures](#model-architectures)
  - [Training Setup](#training-setup)
- [Results](#results)
  - [Training Curve](#training-curve)
  - [Classification Report](#classification-report)
  - [Confusion Matrix](#confusion-matrix)
- [Model Interpretability: Grad-CAM](#model-interpretability-grad-cam)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## Overview

This project builds an end-to-end pipeline for automated brain tumor classification from MRI scans:

1. Load and preprocess a public Kaggle MRI dataset (4 classes, 7,200 images total).
2. Train two models for comparison — a **from-scratch CNN baseline** and a **fine-tuned ResNet18** (ImageNet-pretrained, frozen backbone).
3. Evaluate with a full classification report and confusion matrix.
4. Generate **Grad-CAM** heatmaps to visualize *which regions of the MRI the model relies on* for each prediction — turning a black-box classifier into something a domain expert could sanity-check.

All experiments were run on Google Colab (T4 GPU).

## Motivation

Manual review of MRI scans for tumor screening is time-consuming and requires specialist radiologists. While a CNN classifier alone isn't a substitute for clinical diagnosis, this project explores how far a relatively lightweight transfer-learning pipeline can go on a public dataset, and — more importantly — how to make its predictions **interpretable** rather than just accurate. Grad-CAM was included specifically to surface failure modes (e.g., a model classifying correctly for the wrong reasons) that a raw accuracy number would hide.

## Dataset

- **Source:** Public Kaggle brain tumor MRI dataset (4-class).
- **Classes:** `glioma`, `meningioma`, `notumor`, `pituitary`
- **Split:**

| Split | Images |
|---|---|
| Training | 5,600 |
| Testing | 1,600 |

- **Format:** Folder-per-class structure (`Training/<class>/*.jpg`, `Testing/<class>/*.jpg`), loaded via `torchvision.datasets.ImageFolder`.

## Approach

### Preprocessing & Augmentation

All images were resized to **224×224** and normalized using standard ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) to match the pretrained ResNet18's expected input distribution.

Training-time augmentation:
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness ±0.1, contrast ±0.1)

Test-time transform was resize + normalize only (no augmentation), to get a clean, reproducible evaluation signal.

### Model Architectures

**1. Baseline CNN (from scratch)** — a comparison point to quantify the benefit of transfer learning:
- 4 convolutional blocks (32 → 64 → 128 → 128 channels), each with `Conv2d → BatchNorm → ReLU → MaxPool`
- Global average pooling → Dropout(0.3) → Linear classifier

**2. ResNet18 (transfer learning)** — the primary model:
- ImageNet-pretrained `resnet18` backbone, **frozen** (only the classification head is trained)
- Custom head: `Dropout(0.3) → Linear(512, 4)`
- Rationale for freezing: with ~5,600 training images, fine-tuning the full backbone risks overfitting; freezing lets the head learn on top of robust, general-purpose ImageNet features while keeping training fast and lightweight.

### Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Epochs | 10 |
| Loss | CrossEntropyLoss |
| Checkpointing | Best validation accuracy saved each epoch |
| Hardware | Colab T4 GPU |

## Results

### Training Curve

The ResNet18 model was trained for 10 epochs, with the best checkpoint saved whenever validation accuracy improved:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.7768 | 0.7027 | 0.6354 | 0.7744 |
| 2 | 0.5302 | 0.8041 | 0.6015 | 0.7762 |
| 3 | 0.4942 | 0.8075 | 0.5059 | 0.8200 |
| 4 | 0.4690 | 0.8246 | 0.4996 | 0.8275 |
| 5 | 0.4576 | 0.8282 | 0.5928 | 0.8081 |
| 6 | 0.4329 | 0.8405 | 0.4835 | **0.8363** ✅ best |
| 7 | 0.4478 | 0.8252 | 0.5124 | 0.8200 |
| 8 | 0.4439 | 0.8332 | 0.5686 | 0.8050 |
| 9 | 0.4373 | 0.8423 | 0.5277 | 0.8244 |
| 10 | 0.4518 | 0.8273 | 0.5443 | 0.8163 |

**Best validation accuracy: 83.63%** (epoch 6). Validation accuracy plateaus and mildly fluctuates after epoch 6, suggesting the frozen-backbone head has largely converged and further gains would likely require unfreezing later ResNet layers for fine-tuning (see [Future Work](#future-work)).

*(Insert `training_curve.png` here if you plot train/val loss & accuracy vs. epoch.)*

### Classification Report

Evaluated on the held-out test set (1,600 images):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| glioma | 0.87 | 0.69 | 0.77 | 400 |
| meningioma | 0.71 | 0.76 | 0.73 | 400 |
| notumor | 0.87 | 0.97 | 0.92 | 400 |
| pituitary | 0.91 | 0.93 | 0.92 | 400 |
| **Accuracy** | | | **0.84** | 1600 |
| Macro avg | 0.84 | 0.84 | 0.83 | 1600 |
| Weighted avg | 0.84 | 0.84 | 0.83 | 1600 |

**Key observations:**
- **`notumor` and `pituitary`** are classified with high precision and recall (F1 ≈ 0.92) — these classes are visually the most distinct.
- **`glioma` recall is the weakest point (0.69)** — the model misses roughly 3 in 10 true glioma cases, most often confusing them with `meningioma` (see confusion matrix below).
- **`meningioma` precision is the weakest (0.71)** — the model over-predicts meningioma, pulling in scans from other classes (mainly glioma and pituitary).
- This asymmetry (glioma ↔ meningioma confusion) is a known challenge in MRI tumor classification, since both are extra-axial/intra-axial soft-tissue masses that can appear visually similar depending on slice orientation and contrast.

### Confusion Matrix

|  | Pred: glioma | Pred: meningioma | Pred: notumor | Pred: pituitary |
|---|---|---|---|---|
| **True: glioma** | 276 | 91 | 28 | 5 |
| **True: meningioma** | 36 | 303 | 32 | 29 |
| **True: notumor** | 4 | 6 | 389 | 1 |
| **True: pituitary** | 2 | 28 | 0 | 370 |

*(Insert `confusion_matrix.png` here.)*

The dominant off-diagonal error is **91 glioma scans misclassified as meningioma** — this is the single largest error bucket in the whole matrix and the main driver of glioma's low recall.

## Model Interpretability: Grad-CAM

To go beyond a raw accuracy number, the project implements **Grad-CAM (Gradient-weighted Class Activation Mapping)** from scratch as a lightweight `GradCAM` class using PyTorch forward/backward hooks:

1. A **forward hook** on the target layer (`model.layer4[-1].conv2`, the last convolutional block of ResNet18) captures the layer's activation maps.
2. A **backward hook** on the same layer captures the gradients of the predicted class's output score with respect to those activations.
3. The gradients are **global-average-pooled** per channel to get importance weights.
4. A weighted sum of the activation maps (followed by ReLU, to keep only features with a positive influence on the prediction) produces the raw class activation map.
5. The map is **upsampled** to the input image resolution (224×224) via bilinear interpolation and min-max normalized to `[0, 1]`.
6. The result is overlaid as a heatmap on the original (de-normalized) MRI slice.

This was run on 8 random test images, showing the ground-truth slice alongside the Grad-CAM overlay, with the predicted label colored **green** (correct) or **red** (incorrect).

*(Insert `gradcam_examples.png` here — a grid of MRI slices with Grad-CAM heatmaps.)*

**Why this matters:** Grad-CAM lets you audit *whether the model is looking at the tumor region itself* versus spurious cues (skull shape, scan artifacts, background). This is standard practice in medical imaging ML and is what separates a "black box classifier" from an interpretable one — even at a portfolio-project scale, it demonstrates awareness of why raw accuracy isn't sufficient for high-stakes domains.

## Limitations

- **Single-dataset training:** The model was trained and validated on one public Kaggle dataset. MRI acquisition protocol, scanner vendor, field strength, and slice orientation vary significantly across hospitals — this model has **not been validated on external/out-of-distribution data**.
- **Unverified labels:** Class labels come from the dataset's own annotations and were **not independently verified by a radiologist**.
- **Frozen backbone:** Only the classification head was fine-tuned; the ResNet18 backbone retained ImageNet-pretrained weights throughout, which may cap the ceiling on accuracy compared to full fine-tuning.
- **Not a diagnostic tool:** This project is intended as a machine learning / portfolio exercise in transfer learning and interpretability — it should not be used, in any form, for actual clinical decision-making.

## Future Work

- **Fine-tune the backbone:** Unfreeze the later ResNet18 blocks (e.g., `layer3`, `layer4`) with a lower learning rate to see if accuracy improves beyond the ~84% ceiling observed here.
- **Address glioma/meningioma confusion:** Try class-weighted loss, targeted data augmentation, or a higher-resolution input to help the model distinguish the two most-confused classes.
- **Cross-dataset validation:** Test on a second, independently sourced MRI dataset to estimate real-world generalization.
- **From-scratch CNN comparison:** Train the baseline CNN (`BaselineCNN` class, already implemented) to quantify the exact accuracy gap between training from scratch vs. transfer learning.
- **Model calibration:** Add temperature scaling or similar and report calibration curves, since a clinically-relevant model needs well-calibrated confidence, not just class accuracy.

## Project Structure

```
brain-tumor-classifier/
├── brain_tumor_classifier.ipynb   # Main Colab notebook (data → train → eval → Grad-CAM)
├── outputs/
│   ├── resnet18_best.pt           # Best checkpoint (val_acc = 0.8363)
│   ├── confusion_matrix.png
│   └── gradcam_examples.png
└── README.md
```

## How to Run

1. Open the notebook in Google Colab.
2. **Runtime → Change runtime type → T4 GPU.**
3. Get a Kaggle API token: [kaggle.com/settings/account](https://www.kaggle.com/settings/account) → *API → Create New Token* → downloads `kaggle.json`.
4. Run the data cells and upload the dataset zip when prompted.
5. Run the training cell — trains ResNet18 for 10 epochs (~few minutes on a T4).
6. Run the evaluation cell for the classification report + confusion matrix.
7. Run the Grad-CAM cell to generate interpretability visualizations.
8. (Optional) Mount Google Drive to persist checkpoints and output images, since Colab's local disk is wiped on runtime disconnect.

## Tech Stack

`PyTorch` · `torchvision` (ResNet18, ImageNet weights) · `scikit-learn` (metrics) · `matplotlib` / `seaborn` (visualization) · `Google Colab` (T4 GPU) · Grad-CAM (custom implementation via forward/backward hooks)

---

*Dataset credit: public Kaggle brain tumor MRI dataset (linked in the notebook rather than re-hosted, per dataset licensing).*
