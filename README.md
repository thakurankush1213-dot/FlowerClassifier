# Flower Image Classification using Transfer Learning (MobileNetV2)

## 📌 Problem Statement
The objective of this project is to develop a deep learning model capable of classifying images based on the type of flower present. The model learns visual patterns from the provided dataset and assigns each image to its corresponding flower category with high accuracy.

---

## 🎯 Project Aim
To build, train, and evaluate a transfer learning image classification pipeline using **TensorFlow / Keras** and **MobileNetV2** to classify flower images across **5 distinct classes**.

---

## 📂 Dataset Overview

- **Total Images:** 3,670
- **Number of Classes:** 5
- **Dataset Archive:** `flower-classes-5.zip` (~222.75 MB)

### Class Distribution
| Flower Class | Image Count |
| :--- | :---: |
| **Daisy** | 633 |
| **Dandelion** | 898 |
| **Roses** | 641 |
| **Sunflowers** | 699 |
| **Tulips** | 799 |
| **Total** | **3,670** |

---

## 📊 Dataset Split Strategy

The raw dataset was split using a stratified ratio (**70% Train / 15% Validation / 15% Test**) with `random_state=42` to ensure class distribution consistency across all splits:

```text
dataset/
└── flower_split/
    ├── train/       (2,567 images - 70%)
    ├── validation/  (551 images - 15%)
    └── test/        (552 images - 15%)
```

---

## ⚙️ Data Preprocessing & Augmentation

**Image Dimensions**: Resized to 224 x 224 pixels.

**Batch Size**: 32

### Training Augmentations:

Rescaling (1./255)

Rotation Range (30°)

Width & Height Shift (0.15)

Zoom Range (0.20)

Horizontal Flip (True)

Validation/Testing: Rescaling (1./255) only.


---

## 🏗️ Model Architecture

The architecture utilizes MobileNetV2 pre-trained on ImageNet as the feature extractor, coupled with a custom classification head:

Base Model: Pre-trained **MobileNetV2** (Weights frozen: base_model.trainable = False)

Global Average Pooling: **GlobalAveragePooling2D()**

Dense Layer: 128 units with **ReLU activation**

Regularization: **Dropout(0.5)**

Output Layer: Dense layer with 5 units and **Softmax activation**

---

## Model Summary

**Total Parameters**: 2,422,597 (9.24 MB)

**Trainable Parameters**: 164,613 (643.02 KB)

**Non-trainable Parameters**: 2,257,984 (8.61 MB)


---

## 🚀 Training & Optimization

**Optimizer**: Adam

**Loss Function**: Categorical Crossentropy

**Metrics**: Accuracy

**Early Stopping**: Monitored val_loss with a patience=3 and restore_best_weights=True.

Best Model State: Achieved at **Epoch 7 (val_loss: 0.3546, val_accuracy: 87.66%)**.


---

## 📈 Evaluation & Results

Test Performance
Evaluated on 552 unseen test set images:

Test Loss: **0.3616**

Test Accuracy: **86.59%**


## Classification Report

```
precision    recall  f1-score   support

       daisy       0.87      0.84      0.86        95
   dandelion       0.96      0.87      0.91       135
       roses       0.78      0.94      0.85        97
  sunflowers       0.83      0.94      0.88       105
      tulips       0.88      0.75      0.81       120

    accuracy                           0.87       552
   macro avg       0.87      0.87      0.86       552
weighted avg       0.87      0.87      0.87       552
```


## Confusion Matrix

```
[[ 80   3   4   3   5]   (Daisy)
 [  5 118   1   9   2]   (Dandelion)
 [  0   0  91   1   5]   (Roses)
 [  4   1   1  99   0]   (Sunflowers)
 [  3   1  19   7  90]]  (Tulips)
```

---

### 💡 Key Insights

Transfer Learning Efficiency: Leveraging pre-trained ImageNet weights allowed the model to achieve ~87% test accuracy within just 7-10 epochs of training.

***Class-Wise Performance***:

Dandelion achieved the highest accuracy and overall performance with an F1-score of 0.91.

Sunflowers showed strong generalization with a recall of 0.94.

Tulips vs. Roses experienced the highest inter-class confusion (19 tulips were misclassified as roses).


---

### 🛠️ Requirements & Environment

Python: 3.8+

TensorFlow: 2.x

Libraries: numpy, scikit-learn, matplotlib, pathlib


---

### 🏃 How to Run

Clone the Repository:

git clone <your-repository-link>
cd <repository-folder>

Prepare Dataset:
Place flower-classes-5.zip in the project root directory.

Execute Notebook:
Open and run the Jupyter Notebook sequentially to extract data, build models, train, and view evaluation metrics.

