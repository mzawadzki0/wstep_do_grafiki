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
        shape_l = (shape[0] * shape[1], shape[2])
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape)
        result.color_model = ColorModel.hsv
        # widok do zmiany wartości bez zmiany kształtu oryginalnej tablicy
        hsv = result.data.view().reshape(shape_l)
        # iteracja po obu 2wym listach ( indeks, (kolory) )
        for i, ((r, g, b), hsv) in enumerate(zip(self.data.astype('float').reshape(shape_l), hsv)):
            mm = max(r, g, b)
            m = min(r, g, b)
            k = (r - (g / 2) - (b / 2)) / sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)
            if g >= b:
                hsv[0] = degrees(acos(k))
            else:
                hsv[0] = 360 - degrees(acos(k))
            hsv[1] = 1 - m / mm if mm > 0 else 0
            hsv[2] = mm / 255
        return result

    def to_hsi(self) -> 'BaseImage':
        shape = self.data.shape
        shape_l = (shape[0] * shape[1], shape[2])
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape)
        result.color_model = ColorModel.hsi
        hsi = result.data.view().reshape(shape_l)
        for i, ((r, g, b), hsi) in enumerate(zip(self.data.astype('float').reshape(shape_l), hsi)):
            mm = max(r, g, b)
            m = min(r, g, b)
            k = (r - (g / 2) - (b / 2)) / sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)
            if g >= b:
                hsi[0] = degrees(acos(k))
            else:
                hsi[0] = 360 - degrees(acos(k))
            hsi[1] = 1 - m / mm if mm > 0 else 0
            hsi[2] = (r + g + b) / 3
        return result

    def to_hsl(self) -> 'BaseImage':
        shape = self.data.shape
        shape_l = (shape[0] * shape[1], shape[2])
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape)
        result.color_model = ColorModel.hsl
        hsl = result.data.view().reshape(shape_l)
        for i, ((r, g, b), hsl) in enumerate(zip(self.data.astype('float').reshape(shape_l), hsl)):
            mm = max(r, g, b)
            m = min(r, g, b)
            k = (r - (g / 2) - (b / 2)) / sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)
            if g >= b:
                hsl[0] = degrees(acos(k))
            else:
                hsl[0] = 360 - degrees(acos(k))
            d = (mm - m) / 255
            l = (0.5 * (mm + m)) / 255
            hsl[1] = d / (1 - abs(2 * l - 1)) if l > 0 else 0
            hsl[2] = l
        return result

    def to_rgb(self) -> 'BaseImage':
        if self.color_model is ColorModel.rgb:
            return self
        shape = (self.data.shape[0], self.data.shape[1], 3)
        shape_l = (shape[0] * shape[1], shape[2])
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape, dtype='uint8')
        result.color_model = ColorModel.rgb
        rgb = result.data.view().reshape(shape_l)
        match self.color_model:
            case ColorModel.hsv:
                for i, ((h, s, v), rgb) in enumerate(zip(self.data.reshape(shape_l), rgb)):
                    mm = 255 * v
                    m = mm * (1 - s)
                    z = (mm - m) * (1 - abs(((h / 60) % float(2)) - 1))
                    if z < 60:
                        rgb[0] = mm
                        rgb[1] = z + m
                        rgb[2] = m
                    elif z < 120:
                        rgb[0] = z + m
                        rgb[1] = mm
                        rgb[2] = m
                    elif z < 180:
                        rgb[0] = m
                        rgb[1] = mm
                        rgb[2] = z + m
                    elif z < 240:
                        rgb[0] = m
                        rgb[1] = mm
                        rgb[2] = z + m
                    elif z < 300:
                        rgb[0] = z + m
                        rgb[1] = m
                        rgb[2] = mm
                    else:
                        rgb[0] = mm
                        rgb[1] = m
                        rgb[2] = z + m

            case ColorModel.hsi:
                for i, ((h, s, i), rgb) in enumerate(zip(self.data.reshape(shape_l), rgb)):
                    if h == 0:
                        rgb[0] = i + 2 * i * s
                        rgb[1] = i - i * s
                        rgb[2] = i - i * s
                    elif h < 120:
                        rgb[0] = i + i * s * cos(h) / cos(60 - h)
                        rgb[1] = i + i * s * (1 - cos(h) / cos(60 - h))
                        rgb[2] = i - i * s
                    elif h == 120:
                        rgb[0] = i - i * s
                        rgb[1] = i + 2 * i * s
                        rgb[2] = i - i * s
                    elif h < 240:
                        rgb[0] = i - i * s
                        rgb[1] = i + i * s * cos(h - 120) / cos(180 - h)
                        rgb[2] = i + i * s * (1 - cos(h - 120) / cos(180 - h))
                    elif h == 240:
                        rgb[0] = i - i * s
                        rgb[1] = i - i * s
                        rgb[2] = i + 2 * i * s
                    else:
                        rgb[0] = i + i * s * (1 - cos(h - 240) / cos(300 - h))
                        rgb[1] = i - i * s
                        rgb[2] = i + i * s * cos(h - 240) / cos(300 - h)

            case ColorModel.hsl:
                for i, ((h, s, l), rgb) in enumerate(zip(self.data.reshape(shape_l), rgb)):
                    d = s * (1 - abs(2 * l - 1))
                    m = 255 * (l - d / 2)
                    z = d * (1 - abs(((h / 60) % float(2)) - 1))
                    if h < 60:
                        rgb[0] = 255 * d + m
                        rgb[1] = 255 * z + m
                        rgb[2] = m
                    elif h < 120:
                        rgb[0] = 255 * z + m
                        rgb[1] = 255 * d + m
                        rgb[2] = m
                    elif h < 180:
                        rgb[0] = m
                        rgb[1] = 255 * d + m
                        rgb[2] = 255 * z + m
                    elif h < 240:
                        rgb[0] = m
                        rgb[1] = 255 * z + m
                        rgb[2] = 255 * d + m
                    elif h < 300:
                        rgb[0] = 255 * z + m
                        rgb[1] = m
                        rgb[2] = 255 * d + m
                    else:
                        rgb[0] = 255 * d + m
                        rgb[1] = m
                        rgb[2] = 255 * z + m
            case ColorModel.gray:
                result.data[:, :, 0] = result.data[:, :, 1] = result.data[:, :, 2] = self.data

        return result
