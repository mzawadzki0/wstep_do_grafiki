from lab2 import ColorModel, BaseImage, np


class GreyScaleTransform(BaseImage):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def to_gray(self, high_contrast=True) -> BaseImage:
        shape = self.data.shape
        shape_l = (shape[0] * shape[1], shape[2])
        result = BaseImage.__new__(BaseImage)
        result.data = np.ndarray(shape, dtype='uint8')
        result.color_model = ColorModel.rgb
        view = result.data.view().reshape(shape_l)
        if high_contrast:
            for i, ((r, g, b), pixel) in enumerate(zip(self.to_rgb().data.reshape(shape_l), view)):
                pixel[0] = pixel[1] = pixel[2] = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            for i, ((r, g, b), pixel) in enumerate(zip(self.to_rgb().data.reshape(shape_l), view)):
                pixel[0] = pixel[1] = pixel[2] = (r + g + b) / 3
        return result

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        pass


class Image(GreyScaleTransform, BaseImage, enumerate):
    def __init__(self, path: str) -> None:
        super().__init__(path)
