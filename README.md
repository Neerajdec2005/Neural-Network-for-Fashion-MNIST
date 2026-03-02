# Neural Network for Fashion MNIST

A neural network classifier built **from scratch in NumPy** (no PyTorch training loop) that recognizes 10 categories of clothing from 28×28 grayscale images using 5-fold cross-validation and a hand-coded Adam optimizer with learning rate decay.

---

## Project Structure

```
NN for FASHION/
├── src/
│   ├── data/
│   │   └── fashion_numpy.py    # Download, flatten & normalise Fashion MNIST
│   ├── models/
│   │   └── numpy_net.py        # Layer_Dense, Dropout, ReLU, Softmax, Adam
│   └── training/
│       └── cross_val.py        # k-fold CV, train_model, evaluate_model
├── checkpoints/                # Saved weights (git-ignored)
├── logs/                       # Output logs (git-ignored)
├── train_numpy.py              # Entry point
├── conftest.py
├── pytest.ini
└── requirements.txt
```

---

## Architecture

All layers are implemented manually in NumPy:

| Component | Detail |
|---|---|
| `Layer_Dense` | Linear transform with L1/L2 regularization on weights & biases |
| `Activation_ReLU` | Element-wise rectified linear unit |
| `Layer_Dropout` | Inverted dropout (disabled during inference) |
| `Activation_Softmax` + `Loss_CategoricalCrossentropy` | Fused for numerically stable backprop |
| `Optimizer_Adam` | Bias-corrected Adam with inverse learning rate decay |

Network: `784 → 256 → 128 → 10`

---

## Quick Start

### 1 – Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

### 2 – Train

```bash
python3 train_numpy.py
```

This will:
1. Download Fashion MNIST automatically
2. Run **5-fold cross-validation** on 60,000 training samples (10 epochs per fold)
3. Retrain on the full training set (15 epochs)
4. Report per-class accuracy on the 10,000-sample test set

---

## Hyperparameters

All hyperparameters are set inside `train_numpy.py` in the `build_model()` function:

| Parameter | Value |
|---|---|
| Hidden layers | 784 → 256 → 128 → 10 |
| L2 regularization | `5e-4` |
| Dropout rate | `0.1` |
| Learning rate | `0.001` |
| LR decay | `1e-4` |
| Adam β₁ / β₂ | `0.9` / `0.999` |
| Batch size | `256` |
| CV folds | `5` |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~97.1% |
| Optimizer | Adam with LR decay |
| Validation strategy | 5-fold cross-validation |

---

## Fashion MNIST Classes

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Stack

| Concern | Tool |
|---|---|
| Numerics / forward & backward pass | NumPy |
| Dataset download | torchvision (utility only) |
| Python version | 3.11+ |

---

## License

MIT
