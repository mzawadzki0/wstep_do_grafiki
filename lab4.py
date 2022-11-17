from lab2 import BaseImage, ColorModel
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


class Histogram:
    values: np.ndarray((256,), dtype='uint32')

    def __init__(self, values: np.histogram, is_grayscale: bool = False) -> None:
        self.values = values
        self.is_grayscale = is_grayscale

    def plot(self) -> None:
        if len(self.values.shape) == 1:
            plt.figure(figsize=(3, 3))
            plt.plot(self.values, 'gray')
        else:
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
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def histogram(self) -> Histogram:
        if self.color_model is ColorModel.gray:
            n_colors, bins = np.histogram(self.data.ravel(), range(256))
            return Histogram(n_colors)
        else:
            r, bins = np.histogram(self.data[:, :, 0].ravel(), range(256))
            g, bins = np.histogram(self.data[:, :, 1].ravel(), range(256))
            b, bins = np.histogram(self.data[:, :, 2].ravel(), range(256))
            n_colors = np.vstack((r, g, b))
        return Histogram(n_colors)

    def compare_to(self, other: BaseImage, method: ImageDiffMethod = ImageDiffMethod.mse) -> float:
        if self.color_model is not ColorModel.gray or other.color_model is not ColorModel.gray:
            print('Only grayscale images are compared!')
            return ...
        h_this = self.histogram()
        h_other = other.histogram()
        result_array = h_this.values.astype(float) - h_other.values.astype(float)
        result_array /= 16
        result_array = result_array ** 2
        result: float = sum(result_array)
        match method:
            case ImageDiffMethod.mse:
                return result
            case ImageDiffMethod.rmse:
                return sqrt(result)
