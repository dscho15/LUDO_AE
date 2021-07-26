import matplotlib.pyplot as plt
import numpy as np

arr = np.loadtxt('win_rate.out', delimiter=',')
winrate_avg = np.zeros(arr.shape[0])
winrate_max = np.zeros(arr.shape[0])

for i in range(arr.shape[0]):
    winrate_avg[i] = np.sum(arr[i])/arr.shape[1]
    winrate_max[i] = np.max(arr[i])

plt.subplot(2, 1, 1)
plt.plot(winrate_avg)
plt.subplot(2, 1, 2)
plt.plot(winrate_max)
plt.show()