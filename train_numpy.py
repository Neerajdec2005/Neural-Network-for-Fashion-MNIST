import numpy as np

from src.data.fashion_numpy import load_fashion_mnist, CLASS_NAMES
from src.models.numpy_net import (
    Layer_Dense, Layer_Dropout, Activation_ReLU,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Optimizer_Adam,
)
from src.training.cross_val import (
    run_cross_validation, train_model, evaluate_model, forward_pass
)


def build_model():
    layers = [
        Layer_Dense(784, 256, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4),
        Activation_ReLU(),
        Layer_Dropout(0.1),
        Layer_Dense(256, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4),
        Activation_ReLU(),
        Layer_Dropout(0.1),
        Layer_Dense(128, 10),
    ]
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-4)
    return layers, loss_activation, optimizer


X_train, y_train, X_test, y_test = load_fashion_mnist()

print("=" * 60)
print("5-Fold Cross Validation  (10 epochs per fold)")
print("=" * 60)

fold_results = run_cross_validation(
    build_fn=build_model,
    X=X_train,
    y=y_train,
    k=5,
    epochs=10,
    batch_size=256,
)

print("\n" + "=" * 60)
print("Final Training on Full Training Set  (15 epochs)")
print("=" * 60)

layers, loss_activation, optimizer = build_model()
train_model(layers, loss_activation, optimizer, X_train, y_train, epochs=15, batch_size=256)

print("\n" + "=" * 60)
print("Test Set Evaluation")
print("=" * 60)

test_loss, test_acc, predictions = evaluate_model(layers, loss_activation, X_test, y_test)
print(f"test_loss: {test_loss:.4f}  test_acc: {test_acc:.4f}")

print("\nPer-Class Accuracy")
print(f"  {'Class':<20} {'Correct':>8} {'Total':>8} {'Acc':>8}")
print("  " + "-" * 48)
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    correct = np.sum(predictions[mask] == y_test[mask])
    total = np.sum(mask)
    acc = correct / total
    print(f"  {name:<20} {correct:>8} {total:>8} {acc:>8.4f}")
