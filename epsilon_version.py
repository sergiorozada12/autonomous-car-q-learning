# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: The aim of this script is to solve a pathfinding problem on a 
grid-like world. Q learning is applied to search for the optimum path. Different
policies are tried

"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

###############################################################################
def getMaxQ(st,actions,Q):
    """ Gets max Q of a given state"""
    return max([Q[st,a] for a in actions])

###############################################################################
def getActionMaxQ(st,actions,Q):
    """ Get action where Q is max for a given state"""
    qActions = [Q[st,a] for a in actions]
    return actions[qActions.index(max(qActions))]

###############################################################################
def getPossibleActions(st,R):
    """ Get all possible actions from a given state"""
    return [a for a in range(len(R[st,:])) if not np.isnan(R[st,a])]

###############################################################################
def chooseAction(st,R,Q,epsilon):
    """ Selects next action following e-greedy policy"""
    possibleActions = getPossibleActions(st,R)
    
    if np.random.random() < epsilon:
        return np.random.choice(possibleActions)

    return getActionMaxQ(st,possibleActions,Q)

###############################################################################
def calculateNewQ(st,act,gamma,alpha,r,Q,R):
    """ Evaluates Q-learning update rule to calculate new Q""" 
    possibleActions = getPossibleActions(act,R)
    maxQ = getMaxQ(act,possibleActions,Q)
    
    return Q[st,act]+alpha*((r+gamma*maxQ)-Q[st,act])

###############################################################################
def updateEpsilon(epsilon):
    if epsilon < 0.5:   return epsilon*0.9999
    else : return epsilon*0.9999

###############################################################################
def displayAgent(image,st):
    """Displays the state of the agent on the world"""
    cv2.circle(image, (int(st[0]),int(st[1])), 10, (0,255,225), -1)
    cv2.imshow('road',image)
    cv2.waitKey(1000)

###############################################################################
def trainAgent(episodes,R,Q,gamma,alpha,epsilon,init_st,final_st):
    """ Train the agent during a given number of episodes"""
    cummR = []
    cummQ = []
    cR = 0
    for i in range(episodes):
        state = init_st
        print("EPISODE: ",i)
        print("Max Q: ",Q.max())
        
        # Count number of steps and append the cummulative reward collected so far
        cummR.append(cR)
        cummQ.append(np.sum(np.absolute(Q)))
        
        while True:
            
            # Choose action, and receive reward
            action = chooseAction(state,R,Q,epsilon)
            r = R[state,action]
            cR += r
            
            # Calculate new Q and perform action
            Q[state,action] = calculateNewQ(state,action,gamma,alpha,r,Q,R)
            state = action
            
            # New epsilon
            epsilon = updateEpsilon(epsilon)
            
            # Break if final state
            if state == final_st:
                break
            
    return (Q,cummQ,cummR)

###############################################################################
def testAgent(R,Q,init_state,final_state,limit,I,mN):
    """ Test wheter the agent takes the optimal path"""
    state = init_state
    path = []
    displayAgent(I.copy(),mN[state,:])
    
    for i in range(limit):
        path.append(state)
        action = chooseAction(state,R,Q,0)
        state = action
        displayAgent(I.copy(),mN[state,:])
        
        if state == final_state:
            path.append(state)
            break
    
    #Close the windows
    cv2.destroyAllWindows()  
    return path
            
###############################################################################

# Parameters of the agent and environment                
R = pd.read_csv('data/R.csv',sep=';',na_values=['-'],index_col = 0).as_matrix()
Q = np.zeros_like(R)
episodes = 1000
gamma = 0.5
alpha = 0.5
epsilon_1 = 0.7
epsilon_2 = 0.8
epsilon_3 = 0.9
initial_state = 25
final_state = 28
limit = 20

# Image to display the final performance
road = cv2.imread('images/roadGrid.PNG',cv2.IMREAD_COLOR)
markovNodes = pd.read_csv('data/gridPositions.csv',sep=';').as_matrix()

# Train to obtain the Q matrix and test greediest policy
Q_1, cummulative_Q_1, cummulative_R_1 = trainAgent(episodes,R,Q.copy(),gamma,alpha,epsilon_1,initial_state,final_state)
p_1 = testAgent(R,Q_1,initial_state,final_state,limit,road.copy(),markovNodes)

Q_2, cummulative_Q_2, cummulative_R_2 = trainAgent(episodes,R,Q.copy(),gamma,alpha,epsilon_2,initial_state,final_state)
p_2 = testAgent(R,Q_2,initial_state,final_state,limit,road.copy(),markovNodes)

Q_3, cummulative_Q_3, cummulative_R_3 = trainAgent(episodes,R,Q.copy(),gamma,alpha,epsilon_3,initial_state,final_state)
p_3 = testAgent(R,Q_3,initial_state,final_state,limit,road.copy(),markovNodes)

# Plot steps
plt.figure(1)
plt.plot(cummulative_Q_1,label="Epsilon: 0.7")
plt.plot(cummulative_Q_2,label="Epsilon: 0.8")
plt.plot(cummulative_Q_3,label="Epsilon: 0.9")
plt.title("Performance of the agent")
plt.xlabel("Episodes")
plt.ylabel("Q-cumulative")
plt.xlim([0,200])
plt.legend()

# Plot steps
plt.figure(2)
plt.plot(cummulative_R_1,label="Epsilon: 0.7")
plt.plot(cummulative_R_2,label="Epsilon: 0.8")
plt.plot(cummulative_R_3,label="Epsilon: 0.9")
plt.title("Performance of the agent")
plt.xlabel("Episodes")
plt.ylabel("Cummulative reward")
plt.legend()