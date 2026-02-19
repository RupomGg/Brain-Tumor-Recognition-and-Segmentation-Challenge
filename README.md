# Brain Tumor Diagnosis: Integrated Classification and U-Net Segmentation for BRISC 2025

[![Challenge: BRISC 2025](https://img.shields.io/badge/Challenge-BRISC%202025-blue.svg)](https://example.com)
[![Framework: TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://tensorflow.org)
[![Tasks: Classification + Segmentation](https://img.shields.io/badge/Tasks-Classification%20%2B%20Segmentation-green.svg)](#)

## 📌 Project Overview
This repository contains a comprehensive deep learning pipeline for the automated diagnosis and mapping of brain tumors. Developed for the **Brain Tumor Recognition and Segmentation Challenge (BRISC 2025)**, the project integrates two critical medical imaging tasks:
1.  **Classification**: Categorizing MRI scans into four types: *Glioma*, *Meningioma*, *Pituitary*, or *No Tumor*.
2.  **Segmentation**: Generating precise pixel-level masks to localize tumor boundaries using U-Net.

## 🚀 Key Features
- **Multi-View Analysis**: Processes MRI slices across **Axial**, **Coronal**, and **Sagittal** planes.
- **Advanced Architectures**:
  - **U-Net**: Optimized for medical image segmentation with binary cross-entropy loss and MeanIoU monitoring.
  - **Transfer Learning (EfficientNetB0 & ResNet50)**: Leverages pre-trained weights with multi-stage fine-tuning for high-accuracy classification.
- **Robust Pipeline**: Includes advanced data augmentation (flips, rotations, contrast adjustments) and class weight balancing to handle data distribution challenges.

## 🛠️ Technical Stack
- **Languages**: Python (Jupyter Notebook)
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, Matplotlib
- **Data Handling**: Scikit-Learn, NumPy

## 📊 Methodology

### 1. Segmentation (U-Net)
- **Model**: Custom U-Net architecture.
- **Training**: Early stopping and model checkpointing for the best validation MeanIoU.
- **Evaluation**: Visual comparison of True Masks vs. Predicted Masks.

### 2. Classification (EfficientNet/ResNet)
- **Stage 1**: Train frozen backbone (Feature Extraction).
- **Stage 2**: Fine-tune deep layers with a reduced learning rate.
- **Metrics**: Accuracy, Precision, Recall, and F1-score via Classification Reports and Confusion Matrices.

## 📂 Dataset Structure
The system is designed to work with the BRISC 2025 folder structure:
```text
/kaggle/input/brisc2025/
└── brisc2025/
    ├── classification_task/
    │   ├── train/ (4 classes)
    │   └── test/ (4 classes)
    └── segmentation_task/
        ├── train/ (images & masks)
        └── test/ (images & masks)
```

## 🏁 Getting Started
1. Clone the repository.
2. Ensure you have access to the **BRISC 2025** dataset.
3. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python scikit-learn seaborn
   ```
4. Open and run `cse428.ipynb`.

---
*Created as part of the CSE428 project requirements.*
