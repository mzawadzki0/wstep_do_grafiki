from lab2 import BaseImage
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


class Histogram:
    values: np.ndarray

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def plot(self) -> None:
        plt.figure(figsize=(10, 3))
        s1 = plt.subplot(1, 3, 1)
        plt.plot(self.values[0], 'r')
        plt.subplot(1, 3, 2, sharey=s1)
        plt.plot(self.values[1], 'g')
        plt.subplot(1, 3, 3, sharey=s1)
        plt.plot(self.values[2], 'b')
        plt.show()


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1


class ImageComparison(BaseImage):
    def histogram(self) -> Histogram:
        shape = (self.data.shape[0] * self.data.shape[1], self.data.shape[2])
        n_colors = np.zeros((3, 256), dtype=int)
        # print(n_colors)
        for r, g, b in self.data.reshape(shape):
            n_colors[0, r] += 1
            n_colors[1, g] += 1
            n_colors[2, b] += 1
        return Histogram(n_colors)

    def compare_to(self, other: BaseImage, method: ImageDiffMethod) -> float:
        pass
