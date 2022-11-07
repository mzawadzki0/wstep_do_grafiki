from lab2 import ColorModel, BaseImage
from lab4 import ImageComparison
import numpy as np


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


class Image(GreyScaleTransform, ImageComparison):
    def __init__(self, path: str) -> None:
        super().__init__(path)
