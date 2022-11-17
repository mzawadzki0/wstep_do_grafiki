from lab3 import Image
from lab4 import ImageDiffMethod as DM


# test histogram & plot
lena: Image = Image('lena.jpg')
lena.histogram().plot()

# lena to grayscale & histogram
gray = lena.to_gray()
img = Image.__new__(Image)
img.data, img.color_model = gray.data, gray.color_model
img.histogram().plot()

# compare img to img
print(img.compare_to(img))

# img2 to grayscale & compare to img
img2: Image = Image("lena - Copy.jpg")
gray = img2.to_gray()
img2.data, img2.color_model = gray.data, gray.color_model
print(img2.compare_to(img))

# img3 to grayscale & compare to img
img3: Image = Image("Screenshot.jpg")
gray = img3.to_gray()
img3.data, img3.color_model = gray.data, gray.color_model
print(img3.compare_to(img))
print(img3.compare_to(img, method=DM.rmse))
