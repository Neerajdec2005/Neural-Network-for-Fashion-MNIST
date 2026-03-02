import numpy as np
from src.models.numpy_net import Layer_Dropout


def k_fold_split(n, k, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return np.array_split(indices, k)


def forward_pass(layers, X, training=True):
    output = X
    for layer in layers:
        if isinstance(layer, Layer_Dropout):
            layer.forward(output, training=training)
        else:
            layer.forward(output)
        output = layer.output
    return output


def backward_pass(layers, loss_activation, y_batch):
    loss_activation.backward(loss_activation.output, y_batch)
    dinputs = loss_activation.dinputs
    for layer in reversed(layers):
        layer.backward(dinputs)
        dinputs = layer.dinputs


def train_model(layers, loss_activation, optimizer, X_train, y_train, epochs=10, batch_size=256):
    dense_layers = [l for l in layers if hasattr(l, 'weights')]

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        epoch_loss = 0.0
        epoch_correct = 0

        for start in range(0, len(X_train), batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            output = forward_pass(layers, X_batch, training=True)
            data_loss = loss_activation.forward(output, y_batch)
            reg_loss = sum(loss_activation.regularization_loss(l) for l in dense_layers)
            loss = data_loss + reg_loss

            predictions = np.argmax(loss_activation.output, axis=1)
            epoch_correct += np.sum(predictions == y_batch)
            epoch_loss += data_loss * len(batch_idx)

            backward_pass(layers, loss_activation, y_batch)

            optimizer.pre_update_params()
            for layer in dense_layers:
                optimizer.update_params(layer)
            optimizer.post_update_params()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc = epoch_correct / len(X_train)
            avg_loss = epoch_loss / len(X_train)
            print(f"  epoch: {epoch + 1:3d}  loss: {avg_loss:.4f}  acc: {acc:.4f}  lr: {optimizer.current_learning_rate:.6f}")


def evaluate_model(layers, loss_activation, X, y):
    output = forward_pass(layers, X, training=False)
    loss = loss_activation.forward(output, y)
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)
    return loss, accuracy, predictions


def run_cross_validation(build_fn, X, y, k=5, epochs=10, batch_size=256):
    folds = k_fold_split(len(X), k)
    fold_results = []

    for fold_idx in range(k):
        print(f"\nFold {fold_idx + 1}/{k}")

        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        layers, loss_activation, optimizer = build_fn()

        train_model(layers, loss_activation, optimizer,
                    X_train_fold, y_train_fold, epochs, batch_size)

        val_loss, val_acc, _ = evaluate_model(layers, loss_activation, X_val_fold, y_val_fold)

        print(f"  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")
        fold_results.append({"fold": fold_idx + 1, "val_loss": val_loss, "val_acc": val_acc})

    accs = [r["val_acc"] for r in fold_results]
    losses = [r["val_loss"] for r in fold_results]
    print(f"\n{k}-Fold CV Results")
    print(f"  mean_acc:  {np.mean(accs):.4f}  std_acc:  {np.std(accs):.4f}")
    print(f"  mean_loss: {np.mean(losses):.4f}  std_loss: {np.std(losses):.4f}")

    return fold_results
