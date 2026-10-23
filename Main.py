import argparse
import csv
from pathlib import Path

# NumPy does the fast number and matrix calculations used by the network.
# From the project folder, install it in this project's VS Code environment with:
#     ./.venv/bin/python -m pip install numpy
import numpy as np

# Each drawing is represented by 64 numbers: 32 x-coordinates followed by
# 32 y-coordinates. The output class numbers follow this list's order.
hidden_layers = [64, 32]
confidence_threshold = 0.8
input_size = 64
output_size = 5

# These are the default training settings; you can override them when creating
# the network or calling fit().
epochs = 2500
learning_rate = 0.005
batch_size = 64
random_seed = 42
early_stopping_patience = 60
minimum_loss_improvement = 1e-4
log_every = 25

# Class IDs follow this order; drawings from other datasets use the final class.
labels = ["circle", "triangle", "lightning", "star", "none"]
training_data_file = Path(__file__).with_name("spell_training_data.csv")
model_file = Path(__file__).with_name("spell_model.npz")


class NeuralNetwork:
    def __init__(
        self,
        hidden_layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=None,
    ):
        if not hidden_layers or any(width <= 0 for width in hidden_layers):
            raise ValueError("hidden_layers must contain positive layer sizes.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

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
                rng.standard_normal(
                    (fan_in, fan_out), dtype=np.float32
                ) * np.float32(scale)
            )
            # Starting biases at zero gives every class a neutral starting point.
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

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
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        for i in reversed(range(len(self.weights))):
            previous_activation = activations[i]

            # These are the adjustments needed for this stage's weights
            # and biases.
            weight_gradients[i] = previous_activation.T @ dz
            bias_gradients[i] = np.sum(dz, axis=0, keepdims=True)

            if i > 0:
                # Pass the error back to the previous stage. ReLU blocks error
                # for values that were zeroed during the forward pass.
                dz = dz @ self.weights[i].T
                dz *= activations[i] > 0

        # Apply updates only after all gradients are calculated, so every
        # gradient uses the same pre-update weights.
        for i in range(len(self.weights)):
            # A small step in the direction that reduces the prediction error.
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def predict_proba(self, X):
        # Return one probability per class for every drawing.
        return self.forward(X)[-1]

    def save(self, path=model_file):
        """Save the network architecture, weights, and biases to an NPZ file."""
        arrays = {
            "layer_sizes": np.asarray(self.layer_sizes, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.str_),
        }
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f"weights_{i}"] = weight
            arrays[f"biases_{i}"] = bias

        with Path(path).open("wb") as model_data:
            np.savez(model_data, **arrays)

    @classmethod
    def load(cls, path=model_file):
        """Load a saved network after checking its architecture and parameters."""
        with np.load(path, allow_pickle=False) as model_data:
            if "labels" not in model_data or not np.array_equal(
                model_data["labels"], np.asarray(labels, dtype=np.str_)
            ):
                raise ValueError(
                    f"{path} was trained with different class labels. "
                    "Regenerate the training data and retrain with "
                    "'python Main.py --train'."
                )

            layer_sizes = model_data["layer_sizes"]
            if (
                layer_sizes.ndim != 1
                or len(layer_sizes) < 3
                or not np.issubdtype(layer_sizes.dtype, np.integer)
                or layer_sizes[0] != input_size
                or layer_sizes[-1] != output_size
                or np.any(layer_sizes[1:-1] <= 0)
            ):
                raise ValueError(f"{path} contains an invalid network architecture.")

            model = cls(layer_sizes[1:-1].tolist())
            for i in range(len(model.weights)):
                weight = np.asarray(model_data[f"weights_{i}"], dtype=np.float32)
                bias = np.asarray(model_data[f"biases_{i}"], dtype=np.float32)
                if (
                    weight.shape != model.weights[i].shape
                    or bias.shape != model.biases[i].shape
                    or not np.all(np.isfinite(weight))
                    or not np.all(np.isfinite(bias))
                ):
                    raise ValueError(
                        f"{path} contains invalid parameters for layer {i}."
                    )
                model.weights[i] = weight.copy()
                model.biases[i] = bias.copy()

        return model

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

    def fit(
        self,
        X,
        y,
        epochs=epochs,
        validation_data=None,
        patience=early_stopping_patience,
        min_delta=minimum_loss_improvement,
        seed=random_seed,
    ):
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if X.ndim != 2 or X.shape[1] != input_size:
            raise ValueError(f"X must have shape (number_of_examples, {input_size}).")
        if y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("y must contain one class ID for every row in X.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer class IDs.")
        if np.any((y < 0) | (y >= output_size)):
            raise ValueError(f"class IDs must be from 0 to {output_size - 1}.")
        if validation_data is not None:
            validation_X, validation_y = validation_data
            if (
                validation_X.ndim != 2
                or validation_X.shape[1] != input_size
                or validation_y.ndim != 1
                or len(validation_X) != len(validation_y)
                or len(validation_y) == 0
            ):
                raise ValueError("validation_data must contain matching X and y arrays.")
            if not np.issubdtype(validation_y.dtype, np.integer):
                raise ValueError("validation labels must be integer class IDs.")
            if np.any((validation_y < 0) | (validation_y >= output_size)):
                raise ValueError(
                    f"validation class IDs must be from 0 to {output_size - 1}."
                )
        if patience <= 0:
            raise ValueError("patience must be positive.")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative.")

        # Shuffle the examples each epoch so training does not depend on their
        # original order, then learn from manageable batches.
        rng = np.random.default_rng(seed)
        best_validation_loss = float("inf")
        best_weights = None
        best_biases = None
        best_epoch = 0
        epochs_without_improvement = 0

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

            # Validation loss tracks how well the network predicts unseen
            # examples; retain the best version and stop if it stalls.
            if validation_data is not None:
                validation_loss = self.loss(validation_X, validation_y)
                if validation_loss < best_validation_loss - min_delta:
                    best_validation_loss = validation_loss
                    best_weights = [weight.copy() for weight in self.weights]
                    best_biases = [bias.copy() for bias in self.biases]
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if (epoch + 1) % log_every == 0 or epoch == 0:
                train_loss = self.loss(X, y)
                train_accuracy = self.accuracy(X, y)
                message = (
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Accuracy: {train_accuracy:.2%}"
                )
                if validation_data is not None:
                    message += (
                        f" | Val loss: {validation_loss:.4f}"
                        f" | Val accuracy: "
                        f"{self.accuracy(validation_X, validation_y):.2%}"
                    )
                print(message)

            if (
                validation_data is not None
                and epochs_without_improvement >= patience
            ):
                print(f"Early stopping after epoch {epoch + 1}.")
                break

        if best_weights is not None and best_biases is not None:
            self.weights = best_weights
            self.biases = best_biases
            print(
                f"Restored best model from epoch {best_epoch} "
                f"(validation loss {best_validation_loss:.4f}, "
                f"accuracy {self.accuracy(validation_X, validation_y):.2%})."
            )


def load_training_data(path=training_data_file):
    """Read the labeled drawing CSV and return numeric inputs and class IDs."""
    class_numbers = {label: number for number, label in enumerate(labels)}
    expected_header = (
        ["label"]
        + [f"x{i}" for i in range(input_size // 2)]
        + [f"y{i}" for i in range(input_size // 2)]
    )
    features = []
    targets = []

    with path.open(newline="", encoding="utf-8") as data_file:
        reader = csv.reader(data_file)
        header = next(reader, None)
        if header != expected_header:
            raise ValueError(
                f"{path} has an unexpected CSV header. "
                "Regenerate it by running quickdraw.py."
            )

        for line_number, row in enumerate(reader, start=2):
            if len(row) != input_size + 1:
                raise ValueError(
                    f"Expected a label and {input_size} values at "
                    f"{path}:{line_number}; found {len(row)} columns."
                )

            label = row[0]
            if label not in class_numbers:
                raise ValueError(
                    f"Unknown label {label!r} at {path}:{line_number}."
                )

            try:
                drawing = np.asarray(row[1:], dtype=np.float64)
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric drawing data at {path}:{line_number}."
                ) from error

            if not np.all(np.isfinite(drawing)):
                raise ValueError(
                    f"Non-finite drawing data at {path}:{line_number}."
                )

            # The CSV coordinates are approximately -100 to 100. Dividing by
            # 100 gives the network smaller, easier-to-learn input values.
            features.append((drawing / 100.0).astype(np.float32))
            targets.append(class_numbers[label])

    if not features:
        raise ValueError(f"No training examples were found in {path}.")

    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def split_by_class(X, y, validation_fraction=0.2, seed=42):
    """Make a repeatable train/validation split while retaining every class."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []

    for class_number in range(output_size):
        class_indices = np.flatnonzero(y == class_number)
        if len(class_indices) < 2:
            raise ValueError(
                f"Need at least two examples of {labels[class_number]} "
                "to create training and validation sets."
            )

        rng.shuffle(class_indices)
        validation_count = min(
            len(class_indices) - 1,
            max(1, int(len(class_indices) * validation_fraction)),
        )
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return (
        X[train_indices],
        y[train_indices],
        X[validation_indices],
        y[validation_indices],
    )


def parse_input(text):
    """Parse 64 pipe-separated drawing coordinates into a network input row."""
    values = text.strip().removeprefix("|").split("|")
    if len(values) != input_size:
        raise ValueError(
            f"Expected exactly {input_size} values separated by '|'; "
            f"received {len(values)}."
        )

    try:
        drawing = np.asarray([value.strip() for value in values], dtype=np.float32)
    except ValueError as error:
        raise ValueError("Every input must be a number.") from error

    if not np.all(np.isfinite(drawing)):
        raise ValueError("Inputs must be finite numbers.")

    # Match the coordinate scaling used when loading the training CSV.
    return (drawing / 100.0).reshape(1, input_size)


def train_and_save_model():
    X, y = load_training_data()
    train_X, train_y, validation_X, validation_y = split_by_class(X, y)

    print(
        f"Training on {len(train_y)} drawings; "
        f"validating on {len(validation_y)} drawings."
    )
    model = NeuralNetwork(hidden_layers, seed=random_seed)
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        validation_data=(validation_X, validation_y),
        seed=random_seed + 1,
    )
    model.save()
    print(f"Saved trained model to {model_file}.")


def predict_from_prompt():
    if not model_file.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_file}. "
            "Train one first with 'python Main.py --train'."
        )

    model = NeuralNetwork.load()
    while True:
        text = input(
            f"Paste {input_size} drawing coordinates separated by '|', "
            "then press Enter:\n"
        )
        try:
            drawing = parse_input(text)
        except ValueError as error:
            print(f"Invalid drawing input: {error}")
            continue
        break

    probabilities = model.predict_proba(drawing)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = probabilities[predicted_class]
    prediction = (
        labels[predicted_class]
        if confidence >= confidence_threshold
        else "None"
    )
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the drawing classifier or predict one pasted drawing."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train the network and save its weights for later predictions",
    )
    args = parser.parse_args()

    if args.train:
        train_and_save_model()
    else:
        predict_from_prompt()


if __name__ == "__main__":
    main()