import matplotlib.pyplot as plt
import numpy as np

arr = np.loadtxt('win_rate.out', delimiter=',')
plt.hist(arr)
plt.show()