# Deepfake Detection via Transfer Learning and Embedding-Space Analysis

Deepfake detection pipeline trained and evaluated on the FaceForensics++ C23 dataset. Fine-tunes three pretrained architectures — ResNet18, EfficientNet-B4, and ViT-Base — and compares their performance on binary real/fake classification across six manipulation methods.

## Results

| Model | Accuracy | F1 (Fake) | ROC-AUC |
|---|---|---|---|
| ResNet18 | **97.43%** | **0.9849** | **0.9977** |
| EfficientNet-B4 | 95.90% | 0.9758 | 0.9936 |
| ViT-Base | 96.38% | 0.9786 | 0.9974 |

## Dataset

[FaceForensics++ C23](https://github.com/ondyari/FaceForensics) — 1,000 original videos paired with six manipulation methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, and DeepFakeDetection. 10 frames are sampled per video for a total of 7,000 frames split 70/15/15 into train/val/test.

## What's in the Notebook

- **Preprocessing** — frame extraction with OpenCV, stratified train/val/test split, inverse-frequency class weighting to handle the 6:1 fake/real imbalance
- **Training** — AdamW with StepLR scheduling, random erasing augmentation, best-checkpoint saving
- **Evaluation** — accuracy, F1, ROC-AUC, confusion matrices, per-manipulation-type accuracy breakdown
- **Embedding analysis** — UMAP projection and K-Means clustering (k=2, k=7) on ViT-Base CLS token embeddings
- **GradCAM** — attention heatmaps across all three architectures on real and fake inputs

## Requirements

```
torch
torchvision
timm
grad-cam
umap-learn
scikit-learn
opencv-python
matplotlib
seaborn
pandas
numpy
```

## Usage

The notebook is designed to run on Kaggle with the FF++ C23 dataset mounted at `/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23`. Extracted frames are cached to `/kaggle/working/frames` so re-runs skip re-extraction.
