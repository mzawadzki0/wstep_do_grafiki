from lab2 import BaseImage
from lab3 import Image
from enum import Enum
import numpy as np
import matplotlib as plt


class Histogram:
    values: np.ndarray

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def plot(self) -> None:
        pass


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1


class ImageComparison(BaseImage):
    def histogram(self) -> Histogram:
        shape = (self.data.shape[0] * self.data.shape[1], self.data.shape[2])
        array = self.data.reshape(shape=shape)


    def compare_to(self, other: Image, method: ImageDiffMethod) -> float:
        pass
