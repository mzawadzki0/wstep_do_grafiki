from lab3 import Image

img = Image('lena.jpg')

# grayscale
# gray = img.to_gray(high_contrast=True)
# gray.show_img()

sepia = img.to_sepia(alpha_beta=(1.5, 0.5))
sepia.show_img()
