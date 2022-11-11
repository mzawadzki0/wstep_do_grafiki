from lab3 import Image

lena = Image('lena.jpg')

# grayscale & grayscale to rgb
# samo grayscale zwraca pojedynczą warstwę
gray = lena.to_gray(high_contrast=True)
gray.to_rgb().show_img()

sepia = lena.to_sepia(alpha_beta=(1.5, 0.5))
sepia.show_img()
