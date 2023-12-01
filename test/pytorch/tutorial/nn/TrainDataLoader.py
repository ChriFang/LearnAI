from pathlib import Path
import pickle
import gzip
import torch

from matplotlib import pyplot


PATH = "data/mnist"
FILENAME = "mnist.pkl.gz"

def load_train_data():
    with gzip.open((PATH + "/" + FILENAME), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    #pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    #pyplot.show()
    return map(torch.tensor, (x_train, y_train, x_valid, y_valid))
