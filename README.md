# Machine Learning Fundamentals

A comprehensive collection of machine learning implementations from scratch using NumPy. This repository demonstrates core ML algorithms with clear mathematical foundations and practical visualizations.

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-latest-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📚 Table of Contents

- [Machine Learning Fundamentals](#machine-learning-fundamentals)
  - [📚 Table of Contents](#-table-of-contents)
  - [🎯 Overview](#-overview)
  - [📁 Project Structure](#-project-structure)
    - [Status Legend](#status-legend)
  - [🚀 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [📊 Practical Sessions](#-practical-sessions)
    - [TP1: Linear Regression](#tp1-linear-regression)
      - [Key Concepts](#key-concepts)
      - [Running](#running)
      - [Expected Output](#expected-output)
      - [Configuration](#configuration)
    - [TP2: Binary Logistic Regression (1D)](#tp2-binary-logistic-regression-1d)
      - [Key Concepts](#key-concepts-1)
      - [Running](#running-1)
      - [Expected Output](#expected-output-1)
      - [Configuration](#configuration-1)
    - [TP3: Multi-dimensional Logistic Regression](#tp3-multi-dimensional-logistic-regression)
      - [Key Concepts](#key-concepts-2)
      - [Features](#features)
      - [Running](#running-2)
      - [Expected Output](#expected-output-2)
      - [Configuration](#configuration-2)
      - [Architecture Highlights](#architecture-highlights)
    - [TP4: Neural Networks \& Backpropagation](#tp4-neural-networks--backpropagation)
    - [TP5: Generalization \& Regularization](#tp5-generalization--regularization)
    - [TP6: Mini Deep Learning Framework](#tp6-mini-deep-learning-framework)
    - [TP7: Deep Networks \& Gradient Instabilities](#tp7-deep-networks--gradient-instabilities)
    - [TP8: NumPy vs Framework Comparison](#tp8-numpy-vs-framework-comparison)
  - [📐 Mathematical Background](#-mathematical-background)
    - [Gradient Descent](#gradient-descent)
    - [Sigmoid Function](#sigmoid-function)
    - [Binary Cross-Entropy Loss](#binary-cross-entropy-loss)
  - [✨ Features](#-features)
    - [Code Quality](#code-quality)
    - [Numerical Stability](#numerical-stability)
    - [Visualization](#visualization)
  - [📝 Usage Examples](#-usage-examples)
    - [Quick Start - Linear Regression](#quick-start---linear-regression)
    - [Quick Start - Binary Classification](#quick-start---binary-classification)
    - [Quick Start - Multi-dimensional Classification](#quick-start---multi-dimensional-classification)
  - [🔧 Development](#-development)
    - [Running Tests](#running-tests)
    - [Performance Benchmarking](#performance-benchmarking)
  - [📄 License](#-license)
  - [🎓 Educational Purpose](#-educational-purpose)

## 🎯 Overview

This repository implements fundamental machine learning algorithms from scratch, focusing on:

- **Gradient Descent Optimization**: Full batch gradient descent with configurable learning rates
- **Loss Functions**: Mean Squared Error (MSE) for regression, Binary Cross-Entropy (BCE) for classification
- **Visualization**: Interactive plots showing data distribution, decision boundaries, and model convergence
- **Numerical Stability**: Proper handling of edge cases, gradient clipping, and epsilon smoothing

Each practical session (TP) builds upon previous concepts, progressing from simple linear regression to multi-dimensional classification.

## 📁 Project Structure

```
ML/
├── README.md                    # This file
├── LICENSE                      # Project license
├── TP1/                         # ✅ Linear Regression
│   ├── main.py                  # Entry point
│   └── utils.py                 # Core implementation
├── TP2/                         # ✅ Binary Classification (1D)
│   ├── main.py                  # Entry point
│   └── utils.py                 # Logistic regression implementation
└── TP3/                         # ✅ Multi-dimensional Classification
│   ├── main.py                  # Entry point
│   └── utils.py                 # Advanced logistic regression
├── TP4/                         # ⏳ Neural Networks & Backpropagation
├── TP5/                         # ⏳ Generalization & Regularization
├── TP6/                         # ⏳ Mini Deep Learning Framework
├── TP7/                         # ⏳ Deep Networks & Gradient Instabilities
└── TP8/                         # ⏳ NumPy vs Framework Comparison
```

### Status Legend

- ✅ Completed and tested
- 🚧 Work in progress
- ⏳ Not yet started

## 🚀 Installation

### Prerequisites

- Python 3.13+ (compatible with 3.8+)
- Virtual environment (recommended)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/MikaelPecyna/MachineLearningFromScratch
   cd ML
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install numpy matplotlib tqdm
   ```

## 📊 Practical Sessions

### TP1: Linear Regression

**Status**: ✅ Complete

Implementation of linear regression using gradient descent to fit a line $y = wx + b$ to noisy data.

#### Key Concepts

- **Model**: Linear function $\hat{y} = wx + b$
- **Loss**: Mean Squared Error (MSE) = $\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$
- **Gradients**:
  - $\frac{\partial L}{\partial w} = \frac{2}{N}\sum((\hat{y}_i - y_i) \cdot x_i)$
  - $\frac{\partial L}{\partial b} = \frac{2}{N}\sum(\hat{y}_i - y_i)$

#### Running

```bash
cd TP1
python main.py
```

#### Expected Output

- Console: Training progress with MSE values
- Plot: Data points with fitted regression line
- Final equation: `Found Equation = w * x + b`

#### Configuration

```python
SIZE_DATA = 1000         # Number of training samples
learning_rate = 0.0001   # Step size for gradient descent
epochs = 100000          # Training iterations
```

---

### TP2: Binary Logistic Regression (1D)

**Status**: ✅ Complete

Binary classification on 1D data using logistic regression with sigmoid activation.

#### Key Concepts

- **Model**: $\hat{y} = \sigma(wx + b)$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Loss**: Binary Cross-Entropy (BCE) = $-\frac{1}{N}\sum[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$
- **Decision Boundary**: Point where $wx + b = 0 \Rightarrow x = -\frac{b}{w}$
- **Gradients**:
  - $\frac{\partial L}{\partial w} = \frac{1}{N}\sum((\hat{y}_i - y_i) \cdot x_i)$
  - $\frac{\partial L}{\partial b} = \frac{1}{N}\sum(\hat{y}_i - y_i)$

#### Running

```bash
cd TP2
python main.py
```

#### Expected Output

- Plot 1: Binary data distribution (red vs blue classes)
- Plot 2: Sigmoid curve with decision boundary and classification regions
- Console: Training progress with BCE and accuracy metrics

#### Configuration

```python
learning_rate = 0.01     # Step size
epochs = 500000          # Training iterations
n_samples = 100          # Dataset size
```

---

### TP3: Multi-dimensional Logistic Regression

**Status**: ✅ Complete

Advanced logistic regression supporting multi-dimensional feature spaces (1D, 2D, 3D+) with proper vectorization and visualization.

#### Key Concepts

- **Model**: $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x} + b)$ where $\mathbf{x} \in \mathbb{R}^d$
- **Vectorization**: Efficient matrix operations $(N \times d)$ for batch processing
- **Normalization**: Z-score standardization $x' = \frac{x - \mu}{\sigma}$ for stable convergence
- **Decision Boundary**: Hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$
- **Early Stopping**: Monitors loss convergence to prevent overfitting

#### Features

✨ **Advanced Visualizations**:

- 1D: Scatter plot with vertical decision boundary
- 2D: Contour plots with color-coded decision regions
- 3D: Interactive 3D plots with decision plane
- Higher dimensions: Automatic projection to 2D

✨ **Performance Optimizations**:

- Vectorized operations (100x faster than naive loops)
- Numerical stability with gradient clipping
- Progress tracking with `tqdm`
- Configurable early stopping

#### Running

```bash
cd TP3
python main.py
```

#### Expected Output

```
Training: 100%|████████| 30000/30000 [00:01<00:00, 16564.22it/s, loss=0.0662, acc=0.9920]
```

- Plot 1: Initial data distribution with true hyperplane
- Plot 2: Final data with learned decision boundary

#### Configuration

```python
learning_rate = 0.01         # Step size
epochs = 30000               # Max training iterations
n_samples = 500              # Dataset size
n_features = 3               # Dimensionality (1, 2, 3+)
```

#### Architecture Highlights

**Gradient Computation** (Vectorized):

```python
error = y_pred - y          # (N, 1)
dw = (x.T @ error) / N      # (d, 1)
db = error.mean()           # scalar
```

**Prediction Pipeline**:

```python
logits = x @ w + b          # Linear combination
probs = sigmoid(logits)     # Activation
classes = (probs > 0.5)     # Threshold
```

---

### TP4: Neural Networks & Backpropagation

**Status**: ⏳ Not yet started

Manual implementation of backpropagation in a multi-layer neural network (Input → Linear → ReLU → Linear → Sigmoid). Will focus on deriving gradients analytically and validating them numerically using finite differences.

_This practical session has not been implemented yet._

---

### TP5: Generalization & Regularization

**Status**: ⏳ Not yet started

Study of overfitting through train/validation splits and L2 regularization. Will analyze the bias-variance tradeoff and implement early stopping based on validation loss.

_This practical session has not been implemented yet._

---

### TP6: Mini Deep Learning Framework

**Status**: ⏳ Not yet started

Construction of a modular deep learning framework inspired by PyTorch. Will implement a `Module` base class, automatic parameter tracking, and an SGD optimizer with proper separation of concerns.

_This practical session has not been implemented yet._

---

### TP7: Deep Networks & Gradient Instabilities

**Status**: ⏳ Not yet started

Investigation of vanishing and exploding gradient problems in networks with ≥5 layers. Will compare different weight initialization strategies (Xavier, He) and measure gradient norms across layers.

_This practical session has not been implemented yet._

---

### TP8: NumPy vs Framework Comparison

**Status**: ⏳ Not yet started

Comparative analysis between pure NumPy and PyTorch implementations. Will evaluate performance, stability, development velocity, and provide a critical analysis of trade-offs.

_This practical session has not been implemented yet._

---

## 📐 Mathematical Background

### Gradient Descent

All implementations use batch gradient descent:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t)$$

Where:

- $\theta$: Parameters (weights and bias)
- $\alpha$: Learning rate
- $\nabla_\theta L$: Gradient of loss function

### Sigmoid Function

Used in logistic regression for probabilistic output:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties:

- Range: $(0, 1)$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- Interpretation: $P(y=1|x)$

### Binary Cross-Entropy Loss

Measures the difference between predicted probabilities and true labels:

$$L = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

Gradient (via chain rule):

$$\frac{\partial L}{\partial w_j} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)x_{ij}$$

## ✨ Features

### Code Quality

- ✅ Type hints for all functions
- ✅ Comprehensive docstrings
- ✅ Modular architecture (separation of concerns)
- ✅ Consistent naming conventions
- ✅ Error handling and validation

### Numerical Stability

- ✅ Epsilon smoothing in logarithms (`1e-15`)
- ✅ Gradient clipping for sigmoid stability
- ✅ Safe division checks
- ✅ Shape validation for broadcasting prevention

### Visualization

- ✅ Clean matplotlib plots with legends and grids
- ✅ Automatic dimension detection (1D/2D/3D)
- ✅ Decision boundary overlays
- ✅ Probability heatmaps for classification

## 📝 Usage Examples

### Quick Start - Linear Regression

```python
from TP1.utils import createData, train, plotDataWithCurve

x, y = createData(size=1000)
w, b = train(x, y, learning_rate=0.0001, epochs=10000)
plotDataWithCurve(x, y, w, b)
```

### Quick Start - Binary Classification

```python
from TP2.utils import createData, train, plotDecisionBoundary

x, y = createData(size=100)
w, b = train(x, y, learning_rate=0.01, epochs=50000)
plotDecisionBoundary(x, y, w, b)
```

### Quick Start - Multi-dimensional Classification

```python
from TP3.utils import create_data, normalize, train, plot_data

x, y, w_true, b_true = create_data(n_samples=500, n_features=2)
x = normalize(x)
w, b, history = train(x, y, learning_rate=0.01, epochs=30000, early_stopping=True)
plot_data(x, y, w, b)
```

## 🔧 Development

### Running Tests

```bash
# Test all TPs sequentially
for tp in TP1 TP2 TP3; do
    echo "Testing $tp..."
    python $tp/main.py
done
```

### Performance Benchmarking

```python
import time
start = time.time()
w, b = train(x, y, learning_rate=0.01, epochs=10000)
print(f"Training time: {time.time() - start:.2f}s")
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Educational Purpose

This repository is designed for educational purposes to understand:

- Gradient-based optimization
- Vectorization in NumPy
- Numerical stability in ML algorithms
- Proper evaluation of classification models

**Note**: For production use cases, consider frameworks like scikit-learn, TensorFlow, or PyTorch that offer optimized implementations with GPU support.

---
