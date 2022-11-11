from lab2 import BaseImage


lena = BaseImage('lena.jpg')

# rgb to hsv & hsv to rgb
hsv = lena.to_hsv()
hsv.to_rgb().show_img()

# rgb to hsi & hsi to rgb
hsi = lena.to_hsi()
hsi.to_rgb().show_img()

# rgb to hsl & hsl to rgb
hsl = lena.to_hsl()
hsl.to_rgb().show_img()
