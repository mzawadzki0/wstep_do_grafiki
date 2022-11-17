from lab2 import BaseImage
import matplotlib.colors as colors


lena = BaseImage('lena.jpg')
hsv = lena.to_hsv()

hsv.data[:, :, 0] /= 360
view = hsv.data.view().reshape(512*512, 3)
for i, val in enumerate(view):
    view[i] = colors.hsv_to_rgb(val)

hsv.show_img()
