import numpy as np
from src.utils import softmax


def train(w1, w2, data, one_hot, CW, lr, EPOCHS):
    count = len(data)

    losses = []
    accuracies = []

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0

        for context, target in data:
            indices = [one_hot[word] for word in context]

            # Forward
            h = np.mean(w1[indices], axis=0)
            scores = w2.T @ h
            probs = softmax(scores)

            # Backward
            delta = probs.copy()
            delta[one_hot[target]] -= 1

            d_w2 = np.outer(h, delta)
            d_h = w2 @ delta

            # Update
            w2 -= lr * d_w2

            for idx in indices:
                w1[idx] -= lr * d_h / (2 * CW)

            # Calculate metrics
            loss = -np.log(probs[one_hot[target]]+1e-9)
            total_loss += loss
            predicted = np.argmax(probs)
            if predicted == one_hot[target]:
                correct += 1

        avg_loss = total_loss / count
        acc = correct / count
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {acc:.4f}")

    return losses, accuracies