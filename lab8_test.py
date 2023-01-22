import cv2
import matplotlib.pyplot as plt
import numpy as np


out = cv2.imread('n.jpg', cv2.IMREAD_COLOR)
plt.imshow(out, cmap='gray')
plt.show()

out1 = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
plt.imshow(out1, cmap='gray')
plt.show()

out2 = cv2.cvtColor(out1, cv2.COLOR_RGB2GRAY)
plt.imshow(out2, cmap='gray')
plt.show()

clahe = cv2.createCLAHE(clipLimit=28, tileGridSize=(3,3))
out3 = clahe.apply(out2)
plt.imshow(out3, cmap='gray')
plt.show()

out4 = 255 - out3
plt.imshow(out4, cmap='gray')
plt.show()

a = out4 * 0.4
out5 = np.zeros((out4.shape[0], out4.shape[1], 3), 'uint8')
out5[:, :, 0], out5[:,:,1], out5[:,:,2] = a, a, out4
plt.imshow(out5)
plt.show()
