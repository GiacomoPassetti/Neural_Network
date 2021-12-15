
import matplotlib.pyplot as plt
import numpy as np

top = 12

x = []
dev_NN = []
energy_NN = []

dev_ED = []
energy_ED = []
for i in range(4, 11):
    dev_NN.append("NN_energy_L_"+str(L)+"avgs_"+str(iterations)+".npy")
    energy_NN.append("NN_dev_L_"+str(L)+"avgs_"+str(iterations)+".npy")
    
    dev_ED.append()
    energy_ED.append()
x = np.arange(4, top, 2)
y1 = np.load("NN_energy_up_to_10_20avgs.npy")
dev1 = np.load("NN_std_up_to_10_20avgs.npy")/np.sqrt(20)

y2 = np.load("energy_up_to_20_50avgs.npy")[0:4]
dev2 = np.load("std_up_to_20_50avgs.npy")[0:4]/np.sqrt(50)

print(y2[1], dev2[1])

fig, ax = plt.subplots(dpi = 200)
ax.errorbar(x, y1, dev1, ls = "", marker = "x", label = "NN")
ax.errorbar(x, y2, dev2, ls = "", marker = "x", label = "ED")
ax.barh(-0.081, width = 19, height=0.0005, linestyle = '--', color = 'grey')
ax.set_xlabel("N")
ax.set_ylabel(r"$E_{gs}/N$")
ax.legend()
plt.grid(True)
plt.show()

