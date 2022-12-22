from lab5 import Image

lena = Image('lena.jpg')

# grayscale & grayscale to rgb
# samo grayscale zwraca pojedynczą warstwę
lena.to_gray(high_contrast=False).to_rgb().show_img()
lena.to_gray().to_rgb().show_img()

lena.to_sepia(w=30).show_img()
lena.to_sepia(alpha_beta=(1.5, 0.5)).show_img()
