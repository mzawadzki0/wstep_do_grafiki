from lab3 import Image


img = Image('lena.jpg')
gray = img.to_gray(high_contrast=True)
gray.show_img()
