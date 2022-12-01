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

    def show_img(self, show_layers: bool = False) -> None:
        if self.color_model is ColorModel.gray:
            plt.imshow(self.data, cmap='gray')
        else:
            if show_layers:
                f, ax_arr = plt.subplots(1, 3)
                ax_arr[0].imshow(self.data[:, :, 0], cmap='gray')
                ax_arr[1].imshow(self.data[:, :, 1], cmap='gray')
                ax_arr[2].imshow(self.data[:, :, 2], cmap='gray')
            else:
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
            k: float = (r - (g / 2) - (b / 2)) / sqrt(r ** 2 + g ** 2 + b ** 2 - r * g - r * b - g * b)
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
            l = ((mm + m) / 2) / 255
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


class GreyScaleTransform(BaseImage):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def to_gray(self, high_contrast=True) -> BaseImage:
        shape = self.data.shape
        shape_l = (shape[0] * shape[1], shape[2])
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape[0:2], dtype='uint8')
        result.color_model = ColorModel.gray
        view = result.data.view().reshape(shape_l[0], 1)
        if high_contrast:
            for i, ((r, g, b), pixel) in enumerate(zip(self.to_rgb().data.reshape(shape_l), view)):
                pixel[...] = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            for i, ((r, g, b), pixel) in enumerate(zip(self.to_rgb().data.reshape(shape_l), view)):
                pixel[...] = (r + g + b) / 3
        return result

    def to_sepia(self, w: int = None, alpha_beta: tuple = (None, None)) -> BaseImage:
        result = self.to_gray().to_rgb()
        if w is not None:
            if 20 <= w <= 40:
                for l0 in np.nditer(result.data[:, :, 0], op_flags=['readwrite']):
                    l0[...] = 255 if l0 + w * 2 > 255 else l0 + w * 2
                for l1 in np.nditer(result.data[:, :, 1], op_flags=['readwrite']):
                    l1[...] = 255 if l1 + w > 255 else l1 + w
        else:
            if alpha_beta[0] > 1 and alpha_beta[1] < 1:
                for l0 in np.nditer(result.data[:, :, 0], op_flags=['readwrite']):
                    l0[...] = 255 if l0 * alpha_beta[0] > 255 else l0 * alpha_beta[0]
                for l2 in np.nditer(result.data[:, :, 2], op_flags=['readwrite']):
                    l2[...] = 255 if l2 * alpha_beta[1] > 255 else l2 * alpha_beta[1]
        return result


class Histogram:
    values: np.ndarray((256,), dtype='uint32')

    def __init__(self, values: np.histogram, is_grayscale: bool = False) -> None:
        self.values = values
        self.is_grayscale = is_grayscale

    def plot(self) -> None:
        if len(self.values.shape) == 1:
            plt.figure(figsize=(4, 4))
            plt.plot(self.values, 'gray')
        else:
            plt.figure(figsize=(12, 4))
            s1 = plt.subplot(1, 3, 1)
            plt.plot(self.values[0], 'r')
            plt.subplot(1, 3, 2, sharey=s1)
            plt.plot(self.values[1], 'g')
            plt.subplot(1, 3, 3, sharey=s1)
            plt.plot(self.values[2], 'b')
        plt.show()

    def to_cumulated(self) -> 'Histogram':
        result = Histogram.__new__(Histogram)
        result.values = np.ndarray((256,), dtype='uint32')
        for i in range(256):
            result.values[i] = sum(self.values[0:i])
        return result


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


class ImageAligning(BaseImage):
    def __init__(self, path: str):
        super().__init__(path)

    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
        if self.color_model is not ColorModel.gray:
            print('Only grayscale images are aligned!')
            return ...
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(self.data.shape, dtype='uint8')
        result.color_model = ColorModel.gray
        if tail_elimination:
            hist = self.histogram()
            hist = hist.to_cumulated()
            diff = hist.values[-1] - hist.values[0]
            for i, value in enumerate(hist.values):
                if value >= diff * 0.05:
                    m = i
                    break
            for i in range(255, 0, -1):
                if hist.values[i] <= diff * 0.95:
                    mm = i
                    break
        else:
            m = min(self.data[:, :].ravel())
            mm = max(self.data[:, :].ravel())
        if mm == m:
            result.data = self.data
        else:
            result.data = (self.data.astype(float) - m) * 255 // (mm - m)
        return result


class Image(GreyScaleTransform, ImageComparison, ImageAligning):
    def __init__(self, path: str) -> None:
        super().__init__(path)
