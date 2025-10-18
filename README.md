# CSC173: NEURAL NETWORK

**Group Members:**
1. Kaycee T. Nalzaro
2. 

## INTRODUCTION
Artificial Neural Network (ANN) is a computational model inspired by the human brain that uses interconnected nodes to process information and make decisions. This group activity aims to build a simple ANN from scratch using only **Python**,**Numpy** and **Matplotlib** for plotting, without any machine-learning libraries. The purpose of this activity is to understand how neural networks work, including forward propagation, loss computation, and backpropagation with gradient descent. The model will be used to perform **binary classification** on the **Breast Cancer Diagnostic Dataset** by using only two features.

## DATA PREPARATION
During the data preparation, we ontained our dataset using the built-in breast cancer diagnostic dataset from scikit-learn:
[https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

```python
data = load_breast_cancer()

# Select first two features
X = data.data[:, :2]

y = data.target.reshape(-1, 1)
```
The dataset contains measurements of cell nuclei from breast cancer biopsies, with targets labeled as 0 = malignant or 1 = benign. We only selected the first two features which will contain the inout layer with two neurons.
