import numpy as np

a1 = np.ones(50) * 5
print(a1)

a2 = np.arange(1, 26).reshape(5, 5)
print(a2)

a3 = np.arange(10, 51, 2)
print(a3)

a4 = np.eye(4) * 8
print(a4)

a5 = np.arange(0, 1, 0.01).reshape(10, 10)
print(a5)

a6 = np.linspace(0, 1, 50)
print(a6)

a7 = a2[2:5, 1:5]
print(a7)

a8 = a2[0:3, 4:5]
print(a8)

a9 = a2[3:5, :]
print(np.sum(a9))

x, y = np.random.randint(1, 10, 2)
a10 = np.random.randint(0, 100, (x, y))
print(a10)
