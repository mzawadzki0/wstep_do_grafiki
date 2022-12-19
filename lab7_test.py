from lab5 import Image


lich = Image('lichtenstein.jpg')

lich.threshold(128).show_img()
lich.threshold(224).show_img()
