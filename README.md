# Quantum SVM for Water Potability Classification

This project investigates the application of **quantum-enhanced machine learning** to a real-world binary classification task: predicting water potability from physicochemical measurements. Classical machine learning models are used as strong baselines and compared against a **Quantum Support Vector Classifier (QSVC)** built using fidelity-based quantum kernels.

The focus of this work is methodological clarity, fair comparison, and understanding the practical limitations of current quantum machine learning techniques.

---

## Problem Description

- **Task**: Binary classification  
- **Target variable**: `Potability` ∈ {0, 1}  
- **Dataset**: Water Potability Dataset  
- **Features**: Physicochemical water quality indicators (pH, hardness, sulfate, chloramines, conductivity, etc.)

---

## Classical Machine Learning Models

The following classical models are implemented and tuned using cross-validation:

- Logistic Regression (GridSearchCV)
- Random Forest Classifier
- K-Nearest Neighbours
- Multi-Layer Perceptron (Neural Network)
- Support Vector Classifier (linear, polynomial, RBF, sigmoid kernels)

Each model is evaluated using identical train/test splits to ensure fairness.

---

## Quantum Machine Learning Approach

### Quantum Support Vector Classifier (QSVC)

- **Quantum kernel**: Fidelity-based kernel
- **Feature map**: `ZZFeatureMap`
  - Linear entanglement
  - Two repetitions
- **Fidelity estimation**: Compute–Uncompute method
- **Backend**: Qiskit Aer (statevector simulation)

In addition to QSVC, a classical SVC is trained using the **precomputed quantum kernel** to enable direct benchmarking against classical kernels.

---

## Data Preprocessing Pipeline

### Missing Value Handling
Median imputation is applied to the following features:
- `ph`
- `Sulfate`
- `Trihalomethanes`

### Classical ML Scaling
- MinMax normalization to the range [0, 1]

### Quantum ML Scaling
1. PCA for dimensionality reduction
2. Standard scaling
3. MinMax scaling to the range [-1, 1] (required for quantum feature maps)

### Train/Test Splits
- Classical models: 80/20 split
- Quantum models: Reduced subset due to kernel evaluation cost

---

## Evaluation Metrics

All models are evaluated using:

- Accuracy
- F1 Score
- Sensitivity (Recall)
- Specificity (Precision)
- Confusion Matrix

Visual diagnostics include:
- Feature boxplots
- Correlation heatmaps
- Pair plots
- Confusion matrices
- Quantum kernel matrix visualisations

---

## Key Observations

- Classical models such as Random Forests and SVCs provide strong baseline performance.
- QSVC performs competitively on small, carefully preprocessed datasets.
- Quantum kernel evaluation scales poorly and dominates runtime.
- Precomputed quantum kernels allow fair comparison with classical SVCs.
- No general quantum advantage is claimed; performance is dataset- and feature-dependent.

---

## Dependencies

```bash
numpy
pandas
scikit-learn
matplotlib
seaborn
qiskit
qiskit-machine-learning
qiskit-aer
