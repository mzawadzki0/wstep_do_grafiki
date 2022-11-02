import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from enum import Enum
from math import acos, sqrt, degrees, cos


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4


class BaseImage:
    data: np.ndarray
    color_model: ColorModel

    def __init__(self, path: str) -> None:
        self.data = imread(path)
        self.color_model = ColorModel.rgb

    def save_img(self, path: str) -> None:
        imsave(path, self.data)

    def show_img(self) -> None:
        plt.imshow(self.data)
        plt.show()

    def get_layer(self, layer_id: int) -> 'BaseImage':
        shape = self.data.shape
        result = self
        if 0 <= layer_id <= 2:
            for i in range(3):
                if i is not layer_id:
                    result.data[:, :, i] = np.zeros(shape[0:2], dtype=type(self.data[0][0][0]))
            return result

    def to_hsv(self) -> 'BaseImage':
        shape = self.data.shape
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape)
        result.color_model = ColorModel.hsv
        for x in range(shape[0]):
            for y in range(shape[1]):
                mm = max(self.data[x][y])
                m = min(self.data[x][y])
                r, g, b = self.data[x, y, :].astype('float')
                k = (r - (g / 2) - (b / 2)) / sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)
                # k = k if k <= 1 else k - 2
                if self.data[x, y, 1] >= self.data[x, y, 2]:
                    result.data[x][y] = degrees(acos(k))
                else:
                    result.data[x][y] = 360 - degrees(acos(k))
                result.data[x, y, 1] = 1 - m / mm if mm > 0 else 0
                result.data[x, y, 2] = mm / 255
        return result

    def to_hsi(self) -> 'BaseImage':
        shape = self.data.shape
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape)
        result.color_model = ColorModel.hsi
        for x in range(shape[0]):
            for y in range(shape[1]):
                mm = max(self.data[x][y])
                m = min(self.data[x][y])
                r, g, b = self.data[x, y, :].astype('float')
                k = (r - (g / 2) - (b / 2)) / sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)
                if self.data[x, y, 1] >= self.data[x, y, 2]:
                    result.data[x][y] = degrees(acos(k))
                else:
                    result.data[x][y] = 360 - degrees(acos(k))
                result.data[x, y, 1] = 1 - m / mm if mm > 0 else 0
                result.data[x, y, 2] = (r + g + b) / 3
        return result

    def to_hsl(self) -> 'BaseImage':
        shape = self.data.shape
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape)
        result.color_model = ColorModel.hsl
        for x in range(shape[0]):
            for y in range(shape[1]):
                mm = max(self.data[x][y])
                m = min(self.data[x][y])
                r, g, b = self.data[x, y, :].astype('float')
                k = (r - (g / 2) - (b / 2)) / sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)
                if self.data[x, y, 1] >= self.data[x, y, 2]:
                    result.data[x][y] = degrees(acos(k))
                else:
                    result.data[x][y] = 360 - degrees(acos(k))
                d = (mm - m) / 255
                l = (0.5 * (mm + m)) / 255
                result.data[x, y, 1] = d / (1 - abs(2 * l - 1)) if l > 0 else 0
                result.data[x, y, 2] = l
        return result

    def to_rgb(self) -> 'BaseImage':
        if self.color_model is ColorModel.rgb:
            return self
        shape = self.data.shape
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape, dtype='uint8')
        result.color_model = ColorModel.rgb
        match self.color_model:
            case ColorModel.hsv:
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        mm = 255 * self.data[x, y, 2]
                        m = mm * (1 - self.data[x, y, 1])
                        z = (mm - m) * (1 - abs(((self.data[x, y, 0] / 60) % float(2)) - 1))
                        if z < 60:
                            result.data[x, y, 0] = mm
                            result.data[x, y, 1] = z + m
                            result.data[x, y, 2] = m
                        elif z < 120:
                            result.data[x, y, 0] = z + m
                            result.data[x, y, 1] = mm
                            result.data[x, y, 2] = m
                        elif z < 180:
                            result.data[x, y, 0] = m
                            result.data[x, y, 1] = mm
                            result.data[x, y, 2] = z + m
                        elif z < 240:
                            result.data[x, y, 0] = m
                            result.data[x, y, 1] = mm
                            result.data[x, y, 2] = z + m
                        elif z < 300:
                            result.data[x, y, 0] = z + m
                            result.data[x, y, 1] = m
                            result.data[x, y, 2] = mm
                        else:
                            result.data[x, y, 0] = mm
                            result.data[x, y, 1] = m
                            result.data[x, y, 2] = z + m

            case ColorModel.hsi:
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        h, s, i = self.data[x, y, :]
                        if h == 0:
                            result.data[x, y, 0] = i + 2 * i * s
                            result.data[x, y, 1] = i - i * s
                            result.data[x, y, 2] = i - i * s
                        elif h < 120:
                            result.data[x, y, 0] = i + i * s * cos(h) / cos(60 - h)
                            result.data[x, y, 1] = i + i * s * (1 - cos(h) / cos(60 - h))
                            result.data[x, y, 2] = i - i * s
                        elif h == 120:
                            result.data[x, y, 0] = i - i * s
                            result.data[x, y, 1] = i + 2 * i * s
                            result.data[x, y, 2] = i - i * s
                        elif h < 240:
                            result.data[x, y, 0] = i - i * s
                            result.data[x, y, 1] = i + i * s * cos(h - 120) / cos(180 - h)
                            result.data[x, y, 2] = i + i * s * (1 - cos(h - 120) / cos(180 - h))
                        elif h == 240:
                            result.data[x, y, 0] = i - i * s
                            result.data[x, y, 1] = i - i * s
                            result.data[x, y, 2] = i + 2 * i * s
                        else:
                            result.data[x, y, 0] = i + i * s * (1 - cos(h - 240) / cos(300 - h))
                            result.data[x, y, 1] = i - i * s
                            result.data[x, y, 2] = i + i * s * cos(h - 240) / cos(300 - h)

            case ColorModel.hsl:
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        h, l, s = self.data[x, y, :]
                        d = s * (1 - abs(2 * l - 1))
                        m = 255 * (l - d / 2)
                        z = d * (1 - abs(((h / 60) % float(2)) - 1))
                        if h < 60:
                            result.data[x, y, 0] = 255 * d + m
                            result.data[x, y, 1] = 255 * z + m
                            result.data[x, y, 2] = m
                        elif h < 120:
                            result.data[x, y, 0] = 255 * z + m
                            result.data[x, y, 1] = 255 * d + m
                            result.data[x, y, 2] = m
                        elif h < 180:
                            result.data[x, y, 0] = m
                            result.data[x, y, 1] = 255 * d + m
                            result.data[x, y, 2] = 255 * z + m
                        elif h < 240:
                            result.data[x, y, 0] = m
                            result.data[x, y, 1] = 255 * z + m
                            result.data[x, y, 2] = 255 * d + m
                        elif h < 300:
                            result.data[x, y, 0] = 255 * z + m
                            result.data[x, y, 1] = m
                            result.data[x, y, 2] = 255 * d + m
                        else:
                            result.data[x, y, 0] = 255 * d + m
                            result.data[x, y, 1] = m
                            result.data[x, y, 2] = 255 * z + m

        return result
