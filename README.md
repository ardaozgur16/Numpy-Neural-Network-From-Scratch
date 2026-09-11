# Single-Neuron Neural Network from Scratch (Pure NumPy)

A minimalist implementation of an Artificial Neuron (Perceptron / Logistic Regression unit) built entirely from scratch using pure NumPy—without high-level frameworks like PyTorch or TensorFlow.

## Key Concepts Implemented
* **Activation Function:** Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$ for binary classification.
* **Forward Propagation:** Linear combination of input vectors and weight parameters: $z = Xw + b$.
* **Backpropagation & Optimization:** Analytical gradient computation with Gradient Descent:
  * $\frac{\partial L}{\partial w} = \frac{1}{m} X^T (A - y)$
  * $\frac{\partial L}{\partial b} = \frac{1}{m} \sum (A - y)$
* **Binary Decision Threshold:** Decision boundary at probability threshold $p \ge 0.5$.

## Requirements
* Python 3.7+
* NumPy

## Quick Start
```bash
git clone https://github.com/ardaozgur16/Numpy-Neural-Network-From-Scratch.git
cd Numpy-Sinir-Agi-Sifirdan-Olusturma
python neural_network.py
