from lab5 import Image


lena = Image('lena.jpg')

# lena to grayscale
lena.data, lena.color_model = lena.to_gray().data, lena.to_gray().color_model
print(lena.data)

# lena to cumulated histogram
lena.histogram().plot()
lena.histogram().to_cumulated().plot()

# lena aligned
lena.data = lena.align_image().data
print(lena.data)
for i in lena.data.ravel():
    if i < 0 or i > 255:
        print(i)
lena.histogram().plot()
lena.show_img()
