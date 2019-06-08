import numpy as np
import matplotlib.pyplot as plt

s = np.genfromtxt('S8.mat')
A = s[0:1024]

# Calcula a Taxa Local de Flutuação
Amean = np.mean(A)
Asdev = np.std(A)
gamma = (A - Amean) / (Asdev**2)
# hist = the values of the histogram
# bin_edges = the bin edges (length(hist)+1)
hist, bin_edges = np.histogram(gamma)

# Display
plt.hist(gamma, density=True)
plt.show()
