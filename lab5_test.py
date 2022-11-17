from lab5 import Image


lena = Image('lena.jpg')
lena.data, lena.color_model = lena.to_gray().data, lena.to_gray().color_model
ch = lena.histogram().to_cumulated()
# ch.plot()

aligned = lena.align_image()
# aligned.show_img()
