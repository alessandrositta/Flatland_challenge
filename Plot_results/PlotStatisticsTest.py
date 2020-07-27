#################################################################################
# Project 5 - Flatland challenge                                               #
#                                                                               #
# Plot and save results from metrics and action probailities files              #
#                                                                               #
# @Giovanni Montanari - Lorenzo Sarti - Alessandro Sitta                        #
#                                                                               #
# For any question contact us:                                                  #
#         mail : alessandro.sitta@live.it                                       #
#         GitHub: https://github.com/alessandrositta                            #
#                                                                               #
# INSTRUCTIONS: To select the desired file change the datapath at line 28.      #
#                                                                               #
#################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

present = 0

datapath = path.join('Test_Results' , 'test_metrics_malfunctions_multi5_new.csv')
scores_1 = pd.read_csv(datapath,delimiter=';', header=None)
scores = scores_1.drop([1,2,3,4],axis=1)
dones = scores_1.drop([0,1,3,4],axis=1)
deadlock = scores_1.drop([0,1,2,3],axis=1)
average_deadlocks = np.mean(deadlock)
average_dones = np.mean(dones)
average_score = np.mean(scores)
x = ['Average_dones',average_dones,'Average_score',average_score,'Average_deadlocks',average_deadlocks,'Number of Iterations',len(dones)]

print(average_dones)
print(average_score)
print(average_deadlocks)

#present = 1

if present:
    datapath = path.join('Test_Results' , 'test_action_prob_multi_global10.csv')
    action_1 = pd.read_csv(datapath,delimiter=';', header=None)
    average_probabilities = np.mean(action_1)
    x.append('Average action probabilities')
    x.append(average_probabilities)
    print(average_probabilities)


with open(path.join('Test_Results' , 'results.txt'), "w") as output:
    output.write(str(x))


dones.hist()
plt.title('Dones')
plt.show()

deadlock.hist()
plt.title('Deadlocks')
plt.show()

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(scores)
axs[0].set_title('Scores')
axs[0].set_xlabel('Number of iterations')
axs[0].set_ylabel('Scores')
fig.suptitle('Plot', fontsize=16)
axs[0].grid()

axs[1].plot(dones)
axs[1].set_xlabel('Number of iterations')
axs[1].set_title('Dones')
axs[1].set_ylabel('Dones')
axs[1].grid()

plt.show()


if present:
    action_1.hist()
    plt.show()

    plt.bar([1,2,3,4,5], height = average_probabilities,tick_label=['Do nothing','Move left','Move forward','Move right','Stop moving'])
    plt.grid()
    plt.show()





