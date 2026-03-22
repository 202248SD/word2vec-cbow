import numpy as np
from nltk.corpus import brown
from src.data import preprocess_data, one_hot_convert
from src.train import train
from src.utils import plot_training

# HYPERPARAMETERS
EPOCHS = 100
CW = 2
lr = 0.05
d = 50

# Load data
vocab, data = preprocess_data(brown, CW)

# Reduce vocab size
vocab = vocab[:10000]
one_hot = one_hot_convert(vocab)

data = [
    (context, target)
    for context, target in data
    if target in one_hot and all(w in one_hot for w in context)
]

V = len(vocab)

# Initialize weights
w1 = np.random.randn(V, d) * 0.1
w2 = np.random.randn(d, V) * 0.1

# Train
print(f"Training with a vocabulary of size {len(vocab)} and training data of size {len(data)}")
losses, accuracies = train(w1, w2, data, one_hot, CW, lr, EPOCHS)

# Plot
plot_training(losses, accuracies, save_path="training_plot.png")