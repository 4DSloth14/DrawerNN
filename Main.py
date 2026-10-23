# NumPy does the fast number and matrix calculations used by the network.
# From the project folder, install it in this project's VS Code environment with:
#     ./.venv/bin/python -m pip install numpy
import numpy as np

# Each drawing is represented by 64 numbers: 32 x-coordinates followed by
# 32 y-coordinates. The output class numbers follow this list's order.
hidden_layers = [32, 16]
input_size = 64
output_size = 4

# These are the default training settings; you can override them when creating
# the network or calling fit().
epochs = 50
learning_rate = 0.001
batch_size = 32

# Class number 0 means circle, 1 slash, 2 lightning, and 3 star.
labels = ["circle", "slash", "lightning", "star"]


class NeuralNetwork:
    def __init__(
        self,
        hidden_layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=None,
    ):
        # Store the training settings and the number of numbers at each stage.
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layer_sizes = [input_size] + list(hidden_layers) + [output_size]

        # A seeded random generator makes the initial network repeatable
        # when a seed is supplied.
        rng = np.random.default_rng(seed)

        # A weight is a learnable connection between two stages; a bias is an
        # extra learnable adjustment for each output in the next stage.
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            # Scale random starting weights to help signals stay at a useful
            # size as they pass through the network.
            if i < len(self.layer_sizes) - 2:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)

            self.weights.append(
                rng.standard_normal((fan_in, fan_out)) * scale
            )
            # Starting biases at zero gives every class a neutral starting point.
            self.biases.append(np.zeros((1, fan_out)))

    def forward(self, x):
        # Keep each stage's outputs because training needs them to work out
        # how each weight contributed to the final answer.
        activations = [x]
        current = x
        for i in range(len(self.weights) - 1):
            # Hidden stages use ReLU: negative values become zero, positive
            # values pass through unchanged.
            z = current @ self.weights[i] + self.biases[i]
            current = np.maximum(0, z)
            activations.append(current)

        # The final stage turns four scores into probabilities that add up
        # to 1. Subtracting the largest score keeps exponentials numerically safe.
        z = current @ self.weights[-1] + self.biases[-1]
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        probabilities = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        activations.append(probabilities)
        return activations

    def train_batch(self, X, y):
        # First make predictions. X has one drawing per row; y contains the
        # correct class number (0 through 3) for each row.
        activations = self.forward(X)
        predictions = activations[-1]

        batch_count = X.shape[0]

        # Start with the prediction error, then reduce the probability of the
        # correct class. Dividing by the batch size averages the batch errors.
        dz = predictions.copy()
        dz[np.arange(batch_count), y] -= 1
        dz /= batch_count

        # Work backwards through the network, measuring how each connection
        # affected the error and nudging it to make future predictions better.
        for i in reversed(range(len(self.weights))):
            previous_activation = activations[i]

            # These are the adjustments needed for this stage's weights
            # and biases.
            dW = previous_activation.T @ dz
            db = np.sum(dz, axis=0, keepdims=True)

            if i > 0:
                # Pass the error back to the previous stage. ReLU blocks error
                # for values that were zeroed during the forward pass.
                dz = dz @ self.weights[i].T
                dz *= activations[i] > 0

            # A small step in the direction that reduces the prediction error.
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def predict_proba(self, X):
        # Return one probability per class for every drawing.
        return self.forward(X)[-1]

    def predict(self, X):
        # The most likely class is the one with the highest probability.
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        # Accuracy is the share of drawings whose predicted class is correct.
        return np.mean(self.predict(X) == y)

    def loss(self, X, y):
        # Cross-entropy loss is small when the model gives the correct class
        # a high probability. Clipping prevents taking the logarithm of zero.
        probabilities = self.predict_proba(X)
        correct_probabilities = probabilities[
            np.arange(len(y)), y
        ]
        return -np.mean(
            np.log(np.clip(correct_probabilities, 1e-12, 1.0))
        )

    def fit(self, X, y, epochs=epochs, validation_data=None):
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        # Shuffle the examples each epoch so training does not depend on their
        # original order, then learn from manageable batches.
        rng = np.random.default_rng()
        for epoch in range(epochs):
            indices = rng.permutation(len(X))
            shuffled_X = X[indices]
            shuffled_y = y[indices]

            for start in range(0, len(X), self.batch_size):
                end = start + self.batch_size

                self.train_batch(
                    shuffled_X[start:end],
                    shuffled_y[start:end]
                )

            # Report progress on the training examples after each full pass.
            train_loss = self.loss(X, y)
            train_accuracy = self.accuracy(X, y)

            message = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Accuracy: {train_accuracy:.2%}"
            )

            if validation_data is not None:
                # Validation examples measure how well the model handles data
                # that was not used for the weight updates.
                val_X, val_y = validation_data
                val_accuracy = self.accuracy(val_X, val_y)
                message += f" | Val accuracy: {val_accuracy:.2%}"

            print(message)
if __name__ == "__main__":
    # The class is defined here, but this file does not load a dataset or train
    # a model by itself. Call fit() with numeric X data and integer class IDs.
    pass