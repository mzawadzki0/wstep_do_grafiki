from lab5 import Image
import numpy as np

lena = Image('lena.jpg')
gray = lena.to_gray()

blur = np.ones((5,5))

gaussian_blur = np.array(
    [[1, 4, 6, 2, 1],
     [4, 16, 24, 16, 4],
     [6, 24, 36, 24, 6],
     [4, 16, 24, 16, 4],
     [1, 4, 6, 4, 1]]
)

sharpen = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]]
)

edges_0 = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
)

Image.conv_2d(lena, blur, 0.04).show_img()
Image.conv_2d(lena, gaussian_blur, 1/2**8).show_img()
Image.conv_2d(lena, sharpen).show_img()
Image.conv_2d(lena, edges_0).show_img() # ok
