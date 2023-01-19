import numpy as np


arr = [1,1,1,1,5,100]

print(arr)

print(np.percentile(arr, 70, method='nearest'))
