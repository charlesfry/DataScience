import numpy as np

data:np.array = np.loadtxt('./1.txt')

data:np.array = data.reshape((-1,1,1)) # 200, 1
dt = data.reshape((1,-1,1))             # 1, 200
dorth = data.reshape((1,1,-1))
matrix = data + dt
box = matrix + dorth

where_arr = np.where(box==2020)
i,j,k = where_arr[0][0], where_arr[1][0], where_arr[2][0]

print(int(data[i] + data[j] + data[k]))
print(int((data[i] * data[j] * data[k])[0]))