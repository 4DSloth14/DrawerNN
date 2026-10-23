import numpy as np

hidden_layers = [32, 16]
input_size = 64
output_size = 4

# Training settings
epoch = 50
learning_rate = 0.001
batch_size = 32

labels = ["circle", "slash", "lightning", "star"]


class NeuralNetwork:
    def __init__(self, hidden_layers, learning_rate=0.001, batch_size=32):
        self.np = np
        self.learning_rate = learning_rate
        self.natch_size = batch_size
        self.layer_sizes = [64] + list(hidden_layers) + [4]
        rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes)-1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i+1]

            if i < len(self.layer_sizes)-2:
                scale = np.sqrt(2.0/fan_in)
            else:
                scale = np.sqrt(1.0/fan_in)

            self.weights.append(
                rng.standard_normal((fan_in,fan_out)) * scale
            )
            self.biases.append(
                rng.standard_normal((1,fan_out))
            )

    def forward (self, x):
        activations = [X]
        current = x
        for i in range(len(self.weights)-1)
            z = current @ self.weights[i] + self.biases[i]
            current = np.maximum(0,z)
            activations.append(current)

        z = current @ self.weights[-1] + self.biases[-1]

        z = z - np.max(z, axis = 1, keepdims=True)
        exp_z = np.exp(z)
        probabilities = exp_z / np.sum(
            exp_z, axis=1, keepdims=True
        )

        activations.append(probabilities)
        return activations

    def train_batch(self,X,y):
        activations = self.forward(X)
        predictions = activations[-1]

        batch_count = X.shape[0]

        dz = predictions.copy()
        dz[np.arrange(batch_count)]
        dz /= batch_count
        
        for i in reversed(range(len(self.weights))):
           previous_activation = activations[i]

           dW = previous_activation.T @ dz
           db = np.sum(dz, acis = 0, keepdims=True) 

           if i > 0:
               dz = dz @ self.weights[i].T

               dz *= (activations[i] > 0)
               
           self.weights[i] -= self.learning_rate * dW
           self.biases[i] -= self.learning_rate * db
    def predict_proba(self,X):
        return self.forward(X)[-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(x), axis = 1)

    def accuracy(self,X):
        return np.mean(self.predict(X)==y)
    def loss(self,X,y):
        probabilities = self.predict_proba(X)
        correct_probabilities = probabilities[
            np.arrange(len(y)),y
        ]
        return -np.mean(
            np.log(np.clip(correct_probabilities, 1e-12, 1.0))
        )

    def fit (self, X, y, epochs = 100, validation_data = None):
        if epochs <= 0:
            raise ValueError("Positive epoch")
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
            train_loss = self.loss(X,y)
            train_accuracy = self.accuracy(X,y)

            message = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Accuracy: {train_accuracy:.2%}"
            )

            if validation_data is not None:
                val_X, val_y = validation_data
                val_accuracy = self.accuracy(val_X,val_y)
                message += f" | Val accuracy: {val_accuracy:0.2%}"

            print(message)

            
if __name__ == "__main__":
    pass