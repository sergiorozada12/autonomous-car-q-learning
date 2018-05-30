# -*- coding: utf-8 -*-
"""
@author: Sergio

@description: The aim of this script is to solve a pathfinding problem on a 
grid-like world. Q learning is applied to search for the optimum path. Different
combinations of hyper parameters are used to study the performance of the system

"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

###############################################################################
def getData(path):
    """ Gets R and creates Q of the same size"""
    R = pd.read_csv(path,sep=';',na_values=['-'],index_col = 0).as_matrix()
    return (R,np.zeros_like(R))
    
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
    """ Update epsilon after each step"""
    if epsilon < 0.5:   return epsilon*0.9999
    elif epsilon<0.1:   return 0.1
    else : return epsilon*0.9999

###############################################################################
def displayAgent(image,st):
    """Displays the state of the agent on the world"""
    cv2.circle(image, (int(st[0]),int(st[1])), 10, (0,255,225), -1)
    cv2.imshow('road',image)
    cv2.waitKey(250)

###############################################################################
def trainAgent(episodes,R,Q,gamma,alpha,epsilon,init_st,final_st,limit,I,mN):
    """ Train the agent during a given number of episodes"""
    print("gamma: ",gamma,"alpha: ",alpha)
    cummR = []
    cummQ = []
    cR = 0
    for i in range(episodes):
        state = init_st
        
        # Count number of steps to check convergence and cummulative reward
        cummR.append(cR)
        cummQ.append(np.sum(np.absolute(Q)))
        
        while True:
            
            # Choose action, get reward
            action = chooseAction(state,R,Q,epsilon)
            r = R[state,action]
            cR += r
            
            # Calculate new Q
            Q[state,action] = calculateNewQ(state,action,gamma,alpha,r,Q,R)
            state = action
            
            # New epsilon
            epsilon = updateEpsilon(epsilon)
            
            # Break if final state
            if state == final_st:
                break
            
    print("path: ",testAgent(R,Q,init_st,final_st,limit,I,mN))
    return cummQ,cummR

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
    
    if len(path) == limit:
        return None
    
    #Close the windows
    cv2.destroyAllWindows() 
    return path
            
###############################################################################
 
# Parameters of the agent and the environment               
R,Q = getData('data/R.csv')
episodes = 1000
gamma = [0.2,0.5,0.8]
alpha = [0.2,0.5,0.8]
epsilon = 0.8
initial_state = 25
final_state = 28
limit = 2000

# Image to display the final performance
road = cv2.imread('images/roadGrid.PNG',cv2.IMREAD_COLOR)
markovNodes = pd.read_csv('data/gridPositions.csv',sep=';').as_matrix()

performance = []
rewards = []

# All possible combinations of gamma and alpha
for i in range(len(alpha)):
    for j in range(len(gamma)):       
        cummulative_Q, cummulative_R = trainAgent(episodes,R,Q.copy(),gamma[j],alpha[i],epsilon,initial_state,final_state,limit,road,markovNodes)
        performance.append((('gamma: '+str(gamma[j])+' alpha: '+str(alpha[i])),cummulative_Q))
        rewards.append((('gamma: '+str(gamma[j])+' alpha: '+str(alpha[i])),cummulative_R))

# Plot results         
plt.figure(1,figsize=(8,6))
plt.title("Performance of the agent")
plt.xlabel("Episodes")
plt.ylabel("Q-cumulative") 
plt.plot(performance[0][1],label=performance[0][0])
plt.plot(performance[4][1],label=performance[4][0])
plt.plot(performance[8][1],label=performance[8][0])  
plt.xlim([0,200])
plt.legend()  

plt.figure(2,figsize=(8,6))
plt.title("Performance of the agent")
plt.xlabel("Episodes")
plt.ylabel("Cummulative rewards") 
plt.plot(rewards[0][1],label=rewards[0][0])
plt.plot(rewards[4][1],label=rewards[4][0])
plt.plot(rewards[8][1],label=rewards[8][0])   
plt.legend() 