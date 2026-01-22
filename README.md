# 🚢 Maritime Vessel Classification from Satellite Imagery

This project focuses on **classifying maritime vessels from satellite/aerial imagery** using **deep learning and transfer learning**.  
The goal is not only high classification accuracy, but also **robustness, interpretability, and reliability** in real-world scenarios such as unknown vessel detection and confidence-aware predictions.

---

## 📌 Problem Statement

Satellite imagery is widely used for maritime surveillance, security, and monitoring.  
However, challenges such as varying lighting conditions, scale, noise, and unseen vessel types make classification difficult.

This project aims to:
- Accurately classify known vessel types
- Detect **unknown ships** (open-set recognition)
- Improve model reliability through **confidence calibration**
- Explain model decisions using **Grad-CAM visualizations**

---

## 🧠 Approach Overview

We use **transfer learning with CNN architectures**, combined with **multi-modal image representations** and advanced evaluation techniques.

### Base Models
- EfficientNet-B0 / EfficientNet-B3  
- ResNet-50  
- DenseNet-121  

All models are pretrained on ImageNet and fine-tuned on maritime vessel imagery.

---

---

## ✨ Key Contributions / Novelty

- **Multi-Modal Inputs**
  - RGB
  - Grayscale
  - Contrast-enhanced images  
  → Improves robustness and feature diversity

- **Multi-Model Ensemble**
  - Train multiple architectures independently
  - Fuse predictions using averaging / weighted ensemble

- **Confidence-Aware Predictions**
  - Outputs:
    - Class label
    - Confidence score
    - *Unknown ship* if confidence is below threshold

- **Open-Set Recognition**
  - Detect vessels not seen during training

- **Model Confidence Calibration**
  - Improves reliability of probability estimates

- **Explainability**
  - Grad-CAM–based visualizations
  - Prototype-style feature interpretation

- **Attention Mechanisms**
  - Improve spatial feature extraction

---

## 📂 Dataset

Merged datasets from Kaggle:

- Ships in Aerial Images  
  https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images  

- Ships in Aerial Images (Alternate Source)  
  https://www.kaggle.com/datasets/coderkr/ships-in-aerial-images  

### Dataset Statistics
- Train: **3,142 images**
- Validation: **2,412 images**
- Test: **1,862 images**
- Unique images: **7,416**
- Total images (RGB + Grayscale + Contrast): **22,248**

---

## 🧹 Data Preprocessing

- Cropping ships from raw aerial images
- Image enhancement:
  - RGB
  - Grayscale
  - Contrast-enhanced
- Blur removal
- Resizing to **224 × 224**
- Normalization
- Organized folder structure for multi-input training

---

## 🔁 Data Augmentation

Applied during training:
- Random horizontal / vertical flips
- Rotation
- Zoom
- Brightness & contrast adjustment

---

## 🧠 Model Training Strategy

### Option 1: Separate Training
- Train individual models on:
  - RGB images
  - Grayscale images
  - Contrast-enhanced images

### Option 2: Multi-Input Network
- Fuse RGB, grayscale, and contrast inputs
- Combine feature maps for richer representations

### Transfer Learning
- Freeze early layers
- Fine-tune higher layers
- Optimized with Adam / AdamW

---

## 📊 Model Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Calibration curves (confidence reliability)

---

## ❓ Unknown Ship Detection (Open-Set)

- Confidence-based thresholding
- Samples below confidence threshold labeled as **Unknown**
- Prevents overconfident misclassification

---

## 🔍 Explainability

- **Grad-CAM visualizations**
  - Highlight important image regions
  - Focus on ship contours and structures
- RGB EfficientNet branch visualizations included

---

## 🧪 Model Calibration

- Improves probability reliability
- Ensures confidence scores reflect true correctness likelihood

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Deep Learning**: TensorFlow / PyTorch  
- **Models**: EfficientNet, ResNet, DenseNet  
- **Visualization**: Matplotlib, Grad-CAM  
- **Data Handling**: NumPy, Pandas  
- **Training**: Transfer Learning, CNNs  

---

## 🚀 Future Improvements

- Vision Transformers (ViTs)
- Self-supervised pretraining
- Temporal satellite image analysis
- Real-time deployment pipeline

---

## 📌 Outcome

The final system outputs:
- Predicted vessel
- Confidence score
- Unknown ship detection
- Interpretable visual explanations

This makes the model suitable for **real-world maritime surveillance and monitoring applications**.

---

## 📬 Contact

For collaboration, research discussions, or improvements:
- GitHub: https://github.com/Nitya-Choudhary

