from lab5 import Image


lena = Image('lena.jpg')
# lena to grayscale
lena.data, lena.color_model = lena.to_gray().data, lena.to_gray().color_model

# lena to cumulated histogram
hist = lena.histogram().to_cumulated()
hist.plot()

# lena aligned
aligned = lena.align_image()
aligned.show_img()
