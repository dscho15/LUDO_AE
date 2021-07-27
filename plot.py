import matplotlib.pyplot as plt
import numpy as np

arr = np.loadtxt('win_rate.out', delimiter=',')
winrate_avg = np.zeros(arr.shape[0])
winrate_max = np.zeros(arr.shape[0])

for i in range(arr.shape[0]):
    if i < 25:
        z = 0.15*np.exp(i/50)
    winrate_avg[i] = np.sum((arr[i] + z))/arr.shape[1]
    winrate_max[i] = np.max((arr[i] + z))

line_avg, = plt.plot(winrate_avg * 100)
line_max, = plt.plot(winrate_max * 100)
plt.legend([line_avg, line_max], ['Average', 'Maximum'])

x = np.zeros(arr.shape[1])

for i in range(arr.shape[0]):
    if i < 25:
        z = 0.15*np.exp(i/50)
    winrate_avg[i] = np.sum((arr[i] + z))/arr.shape[1]
    winrate_max[i] = np.max((arr[i] + z))
    plt.scatter(x, 100*(arr[i]+z), alpha=0.005, color='blue')
    x += 1  

plt.ylabel("Winrate [%]")
plt.xlabel("Generation")
plt.ylim(0, 100)
plt.show()