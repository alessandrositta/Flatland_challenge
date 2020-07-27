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
# INSTRUCTIONS: To select the desired file change the datapath at line 32.      #
#                                                                               #
#################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
import sys
from pathlib import Path
import matplotlib

select_act = 0
select_cmp = 0


base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

datapath = path.join('Train_Results' , 'metrics_multi_global10.csv')
scores_1 = np.loadtxt(datapath,delimiter=';')


#select_act = 1
select_cmp = 1



if select_act:
    datapath = path.join('Train_Results' , 'action_prob_multi3_deadlock_invalid_action.csv')
    action_prob_1 = pd.read_csv(datapath,delimiter=';', header=None)
    average_probabilities = np.mean(action_prob_1[(len(action_prob_1)-100):len(action_prob_1)])

if select_cmp:
    datapath = path.join('Train_Results' , 'metrics_multi_reducing_d.csv')
    scores_2 = np.loadtxt(datapath,delimiter=';')

#datapath = path.join('Train_Results' , 'action_prob_reducing_d.csv')
#action_prob_2 = np.loadtxt(datapath,delimiter=';')
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)


fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(scores_1[:,1])
#axs[0].set_title('Dones')
axs[0].set_xlabel('Number of iterations')
axs[0].set_ylabel('Dones')
#fig.suptitle('Plot', fontsize=16)
axs[0].grid()

axs[1].plot(scores_1[:,0])
axs[1].set_xlabel('Number of iterations')
#axs[1].set_title('Scores')
axs[1].set_ylabel('Scores')
axs[1].grid()

plt.show()

plt.plot(scores_1[:,2])
plt.xlabel('Number of iterations')
#axs[2].set_title('Epsilon')
plt.ylabel('Epsilon')
plt.grid()

plt.show()

fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(scores_1[:,1])
#axs[0].set_title('Dones')
axs[0].set_xlabel('Number of iterations')
axs[0].set_ylabel('Dones')
#fig.suptitle('Plot', fontsize=16)
axs[0].grid()

axs[1].plot(scores_1[:,0])
axs[1].set_xlabel('Number of iterations')
#axs[1].set_title('Scores')
axs[1].set_ylabel('Scores')
axs[1].grid()

axs[2].plot(scores_1[:,2])
axs[2].set_xlabel('Number of iterations')
axs[2].set_title('Deadlock')
axs[2].set_ylabel('Deadlock')
axs[2].grid()

plt.show()

if select_act:
    plt.plot(action_prob_1)
    plt.title('Action probabilities')
    plt.xlabel('Number of iterations')
    plt.ylabel('Action probs')
    plt.grid()
    plt.show()
    plt.bar([1,2,3,4,5], height = average_probabilities,tick_label=['Do nothing','Move left','Move forward','Move right','Stop moving'])
    plt.grid()
    plt.show()

if select_cmp:
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(scores_1[0:min(len(scores_1),len(scores_2)),1],label='step penalty')
    axs[0].plot(scores_2[0:min(len(scores_1),len(scores_2)),1],label='reducing distance')
    axs[0].legend()
    #axs[0].set_title('Dones')
    axs[0].set_xlabel('Number of iterations')
    axs[0].set_ylabel('Dones')
    #fig.suptitle('Plot', fontsize=16)
    axs[0].grid()

    axs[1].plot(scores_1[0:min(len(scores_1),len(scores_2)),0],label='step penalty')
    axs[1].plot(scores_2[0:min(len(scores_1),len(scores_2)),0],label='reducing distance')
    axs[1].legend()
    #axs[1].set_title('Scores')
    axs[1].set_xlabel('Number of iterations')
    axs[1].set_ylabel('Scores')
    axs[1].grid()

    #axs[2].plot(scores_1[0:min(len(scores_1),len(scores_2)),2],label='Result1')
    #axs[2].plot(scores_2[0:min(len(scores_1),len(scores_2)),2],label='Result2')
    #axs[2].legend()
    #axs[2].set_xlabel('Number of iterations')
    #axs[2].set_title('Deadlocks')
    #axs[2].set_ylabel('Deadlocks')
    #axs[2].grid()

    plt.show()

