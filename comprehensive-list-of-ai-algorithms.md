# 🧠📐 Comprehensive List of AI Algorithms, Metrics, Optimizers, Loss Functions & Regularizers (with Math Formulas)

This document provides a comprehensive list of essential AI and ML components expressed in **Mathematical LaTeX format** using **MathJax**. You can use this file in any GitHub-based documentation or Jupyter Book that supports MathJax rendering.

---

## 🤖 AI Algorithms

### 1. Linear Regression

**Hypothesis Function**:
$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b
$$

**Loss Function** (MSE):
$$
\mathcal{L}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2
$$

---

### 2. Logistic Regression

**Sigmoid Activation**:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^T \mathbf{x} + b
$$

**Loss Function** (Binary Cross-Entropy):
$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

---

### 3. Support Vector Machine (SVM)

**Objective**:
$$
\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 \quad \text{subject to} \quad y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1
$$

**Hinge Loss**:
$$
\mathcal{L} = \sum_{i=1}^n \max(0, 1 - y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b))
$$

---

### 4. Neural Networks (Feedforward)

**Neuron Output**:
$$
a^{[l]} = \sigma(\mathbf{W}^{[l]} a^{[l-1]} + \mathbf{b}^{[l]})
$$

**Backpropagation**:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \delta^{[l]} (a^{[l-1]})^T
$$

---

### 5. K-Nearest Neighbors (KNN)

**Distance Metric** (Euclidean):
$$
d(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{\sum_{j=1}^{n} (x_{1j} - x_{2j})^2}
$$

---

### 6. Decision Trees

**Information Gain**:
$$
IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

**Entropy**:
$$
H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i
$$

---

### 7. Naive Bayes Classifier

**Bayes Rule**:
$$
P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y)P(y)}{P(\mathbf{x})}
$$

Assumes feature independence:
$$
P(\mathbf{x}|y) = \prod_{i=1}^n P(x_i | y)
$$

---

## 📈 Evaluation Metrics

### 1. Accuracy
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

---

### 2. Precision
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

---

### 3. Recall
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

---

### 4. F1 Score
$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

### 5. ROC AUC

Area under the ROC curve:
$$
\text{AUC} = \int_0^1 TPR(FPR^{-1}(x)) dx
$$

---

### 6. Mean Absolute Error (MAE)
$$
MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y^{(i)} - \hat{y}^{(i)} \right|
$$

---

### 7. Mean Squared Error (MSE)
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} \left( y^{(i)} - \hat{y}^{(i)} \right)^2
$$

---

## 🔧 Optimizers

### 1. Gradient Descent
$$
\theta := \theta - \eta \cdot \nabla_\theta \mathcal{L}(\theta)
$$

---

### 2. Stochastic Gradient Descent (SGD)

Update per sample:
$$
\theta := \theta - \eta \cdot \nabla_\theta \mathcal{L}^{(i)}(\theta)
$$

---

### 3. Momentum
$$
v_t = \gamma v_{t-1} + \eta \nabla_\theta \mathcal{L}(\theta) \\
\theta := \theta - v_t
$$

---

### 4. RMSProp
$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2 \\
\theta := \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

---

### 5. Adam Optimizer
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta := \theta - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

---

## 📉 Loss Functions

### 1. Mean Squared Error (MSE)
$$
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \left( y^{(i)} - \hat{y}^{(i)} \right)^2
$$

---

### 2. Binary Cross-Entropy
$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \left[y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

---

### 3. Categorical Cross-Entropy
$$
\mathcal{L} = - \sum_{i=1}^{n} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})
$$

---

### 4. Hinge Loss (SVM)
$$
\mathcal{L} = \sum_{i=1}^{n} \max(0, 1 - y^{(i)} \cdot f(x^{(i)}))
$$

---

## 🛡️ Regularization Techniques

### 1. L1 Regularization (Lasso)
$$
\mathcal{L}_{\text{reg}} = \lambda \sum_{j=1}^{n} |\theta_j|
$$

---

### 2. L2 Regularization (Ridge)
$$
\mathcal{L}_{\text{reg}} = \lambda \sum_{j=1}^{n} \theta_j^2
$$

---

### 3. Elastic Net
$$
\mathcal{L}_{\text{reg}} = \lambda_1 \sum_{j=1}^{n} |\theta_j| + \lambda_2 \sum_{j=1}^{n} \theta_j^2
$$

---

## 📚 References

- Goodfellow et al., *Deep Learning*
- Bishop, *Pattern Recognition and Machine Learning*
- Wikipedia Math Definitions (MathJax-based)
