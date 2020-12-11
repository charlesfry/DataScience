import numpy as np

data:np.array = np.loadtxt('./1.txt')

data:np.array = data.reshape((-1,1)) # 200, 1
dt = np.transpose(data)             # 1, 200

matrix = data + dt

i, j = np.where(matrix == 2020)[0]

answer = data[i] * data[j]

print(int(answer[0]))