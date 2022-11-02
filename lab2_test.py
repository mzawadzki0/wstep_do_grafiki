from lab2 import BaseImage

img = BaseImage('lena.jpg')

# rgb to hsv & hsv to rgb
hsv = img.to_hsv()
hsv.to_rgb().show_img()

# rgb to hsi & hsi to rgb
hsi = img.to_hsi()
hsi.to_rgb().show_img()

# rgb to hsl & hsl to rgb
hsl = img.to_hsl()
hsl.to_rgb().show_img()
