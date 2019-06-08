# encoding: utf-8
# Exercício 1.3

import numpy as np
import powernoise as pn
import caotica as ct
import matplotlib.pyplot as plt

# Normaliza em torno da média e soma
# "Mean normalization"
# Fonte: Wikipedia(Feature_scaling)
# x' = ( x - average(x) ) / ( max(x) - min(x) )
def norm(x):
    x = ( (x - np.mean(x)) / (max(x) - min(x)) )
    return x

S1 = pn.powernoise(0, 2**12)
S2 = pn.powernoise(1, 2**12)
S3 = pn.powernoise(2, 2**12)

S4 = ct.caotica(2**12, 4.0, 0.001)

S5 = norm(S1) + norm(S4)
S6 = norm(S2) + norm(S4)
S7 = norm(S3) + norm(S4)

plt.plot(S5)
plt.show()
plt.plot(S6)
plt.show()
plt.plot(S7)
plt.show()


np.savetxt('S1.mat', [S1])
np.savetxt('S2.mat', [S2])
np.savetxt('S3.mat', [S3])
np.savetxt('S4.mat', [S4])
np.savetxt('S5.mat', [S5])
np.savetxt('S6.mat', [S6])
np.savetxt('S7.mat', [S7])
