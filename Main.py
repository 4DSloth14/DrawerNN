import numpy as np

hidden_layers = [32, 16]
layer_size = [64] + hidden_layers + [4]
input_size = 64
output_size = 4

# Training settings
epoch = 50
learning_rate = 0.001
batch_size = 32

labels = ["circle", "slash", "lightning", "star"]


class NeuralNetwork:
    def __init__(self, hidden_layers, learning_rate=0.001, batch_size=32):
        pass


if __name__ == "__main__":
    pass