import random
import matplotlib.pyplot as plt
import numpy as np

n = 5
r = []
p = []

for i in range(n) :
    # Roll each Die
    s = random.randint(1, 6) + random.randint(1, 6) + random.randint(1, 6)
    r.append(s)
for dice in range(3, 18) :
    na = r.count(dice)
    p.append(na / n)

# Display
p = np.array(p)
x = np.arange(15)
plt.bar(x, height = p)
plt.ylabel('Probability')
plt.xlabel('Result')
plt.show()