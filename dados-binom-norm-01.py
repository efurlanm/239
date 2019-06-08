import random
import matplotlib.pyplot as plt
import numpy as np

n_simulations = 10000
ne = 3
c = np.array([0, 0, 0, 0])
#def roll_the_dice(n_simulations)

for i in range(n_simulations) :
    # Roll each Die
    die = [random.randint(1, 6), random.randint(1, 6), random.randint(1, 6)]
    a = die.count(ne)

    # add it to the count
    c[a] += 1

#    print(die, c)

b = c / n_simulations

# Display
x = np.arange(4)
plt.bar(x, height = b)
plt.xticks(x, ['0','1','2','3']) 
plt.ylabel('Probability')
plt.xlabel('Same fixed number for 3 dice')
plt.show()