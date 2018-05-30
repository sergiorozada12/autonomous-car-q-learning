# -*- coding: utf-8 -*-
"""
@author: Sergio

@description: The aim of this script is to solve a pathfinding problem. 
Q learning is applied to search for the optimum path. To test the scalability
of the approach instead of considering a grid-like world, the agent is placed
in a pixel map. The Q function is approximated through an ANN.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance

from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers

np.seterr(divide='ignore')

###############################################################################
class Car:
    """ Represents the car by the state and orientation and stores last position"""
    def __init__(self,st,theta):
        self.st_prev = st
        self.st = st
        self.theta = theta
        
    def newCarOrientation(self):
        self.theta = np.arctan2(self.st[1]-self.st_prev[1],self.st[0]-self.st_prev[0])
    
    def newCarPosition(self,new_st):
        self.st_prev = self.st.copy()
        self.st = new_st
        self.newCarOrientation()

###############################################################################
def getPath(path):
    """ Return the coordinates of the center of the road"""
    return pd.read_csv(path,sep=';').as_matrix()

###############################################################################
def getSign(x,y,angle,dist):
    """ Rotate point around another to express distance from the road"""  
    yn = (y[0]-x[0])*np.sin(-angle)+(y[1]-x[1])*np.cos(-angle)
    
    if yn > 0:
        return dist
    
    return -dist    

###############################################################################
def getClosestPoint(st,path):
    """ Return the closest point of the path to the current state"""
    closest = distance.cdist([st], path).argmin()
    return closest

###############################################################################
def getDistance(st,path):
    """ Return the distance between the point and the path"""
    closest = path[getClosestPoint(st,path)]
    roadTheta = getRoadOrientation(st,path)
    dist = np.sqrt((st[0]-closest[0])**2+(st[1]-closest[1])**2)
    return getSign(closest,st,roadTheta,dist)

###############################################################################
def getRoadOrientation(st,path):
    """ Return the orientation of the road in the closest point to state"""
    closest = getClosestPoint(st,path)
    try:
        return np.arctan2((path[closest+30][1]-path[closest][1]),(path[closest+30][0]-path[closest][0]))
    except:
        return 1.6
            
###############################################################################
def getOrientation(path,car):
    """ Get relative Orientation between road and the car"""    
    orientation =  car.theta - getRoadOrientation(car.st,path)
    
    if np.abs(orientation) > 3.14:
        return orientation - 6.18
    
    return orientation 
    
###############################################################################
def getFeatures(car,path):
    """ Get the features correspondent to a given state"""
    return np.array([getDistance(car.st,path),getOrientation(path,car)])

###############################################################################
def getReward(st,orientation,path,goal):
    """ Get reward depending on the current state"""
    if np.abs(getDistance(st,path)) > 50:   return -50
    elif np.abs(orientation)>1.6:   return -10
    elif (np.abs(st[0]-goal[0])+np.abs(st[1]-goal[1])) < 50: return 100
    else:   return 10

###############################################################################
def getActions():
    """ Get the possible actions to take"""
    return [0,1,2] #Straight, turn right or turn left

###############################################################################
def performAction(car,a):
    """ Obtain the new state of the car based on the action selected"""
    if a==0:
        car.newCarPosition(checkBoundaries(np.array([car.st[0]+np.round(20*np.cos(car.theta)),
                                     car.st[1]+np.round(20*np.sin(car.theta))])))
    elif a==1:
        car.newCarPosition(checkBoundaries(np.array([car.st[0]+np.round(20*np.cos(car.theta+0.5)),
                                     car.st[1]+np.round(20*np.sin(car.theta+0.5))])))
    else:
        car.newCarPosition(checkBoundaries(np.array([car.st[0]+np.round(20*np.cos(car.theta-0.5)),
                                     car.st[1]+np.round(20*np.sin(car.theta-0.5))])))
          
    return car

###############################################################################
def getBrain(alpha,b_1,b_2,d):
    """ Instantiate a Keras ANN to approximate Q"""
    model = Sequential()
    
    model.add(Dense(100,  input_shape=(2,),activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='linear'))

    adam = optimizers.Adam(lr=alpha, beta_1=b_1, beta_2=b_2, epsilon=None, decay=d, amsgrad=True)
    model.compile(loss='mse', optimizer=adam)
    
    return model

###############################################################################
def chooseAction(car,path,model,epsilon):
    """ Selects next action following e-greedy policy"""
    possibleActions = getActions()
    
    if np.random.random() < epsilon:
        return np.random.choice(possibleActions)
    
    possibleQ = model.predict(np.array([getFeatures(car,path)]))[0]
    return possibleActions[np.argmax(possibleQ)]

###############################################################################
def updateEpsilon(epsilon):
    """ Update epsilon after each step"""
    if epsilon < 0.5:   return epsilon*0.999999
    elif epsilon < 0.1:   return 0.1
    else : return epsilon*0.99999
    
###############################################################################
def checkBoundaries(state):
    """ Check if state is inside the map"""
    if(state[0]<0): state[0]=0
    if(state[0]>625): state[0]=625
    if(state[1]<0): state[1]=0
    if(state[1]>627): state[1]=627
    return state

###############################################################################
def displayAgent(image,st):
    """Displays the state of the agent on the world"""
    cv2.circle(image, (int(st[0]),int(st[1])), 10, (0,255,225), -1)
    cv2.imshow('road',image)
    cv2.waitKey(1000)

###############################################################################
def testAgent(model,path,init,init_theta,goal,I):
    """Test the performance of the agent"""
    car = Car(init,init_theta)
    pathFound = []
    
    for i in range(200):
        pathFound.append(car.st)
        action = chooseAction(car,path,model,0)
        car = performAction(car,action)
        
        #Show the state
        displayAgent(I.copy(),car.st)
        
        if (np.abs(car.st[0]-goal[0])+np.abs(car.st[1]-goal[1])) < 50:
            pathFound.append(car.st)
            return np.array(pathFound)
    
    return np.array(pathFound)

###############################################################################
def trainAgent(alpha,b_1,b_2,decay,episodes,gamma,epsilon,path,init,init_theta,goal,limit,I):
    """ Train the agent"""
    model = getBrain(alpha,b_1,b_2,decay)
    
    for i in range(episodes):
        car = Car(init,init_theta)
        print("EPISODE: ",i)
        
        for j in range(limit):
            # Current state
            state = getFeatures(car,path)
            
            # Select and apply an action
            action = chooseAction(car,path,model,epsilon)
            Q = model.predict(np.array([state]))[0]
            targetQ = Q.copy()
                        
            car = performAction(car,action)
            newState = getFeatures(car,path)
            
            # Get return
            r = getReward(car.st,getOrientation(path,car),path,goal)
            maxQ = np.amax(model.predict(np.array([newState]))[0])
            target = r+gamma*maxQ
            targetQ[action] = target
            
            #Update the model
            model.fit(np.array([state]), np.array([targetQ]), batch_size=1, epochs=1, verbose=1)
            
            #Update epsilon and check if goal
            epsilon = updateEpsilon(epsilon)
            
            if (np.abs(car.st[0]-goal[0])+np.abs(car.st[1]-goal[1])) < 50:
                break
            
    return testAgent(model,path,init,init_theta,goal,I)
    
###############################################################################

# Parameters of the agent training   
path = getPath('data/center.csv')
episodes = 100
gamma = 0.5
init = np.array([110,627])
init_theta = -1.57
goal = np.array([510,627])
limit = 1000
epsilon = 0.9

# Parameters of the Neural Net to approximate the Q function
alpha = 0.001
b_1 = 0.9
b_2 = 0.999
decay = 0.0

# Image to display the final performance
road = cv2.imread('images/toyRoad.jpg',cv2.IMREAD_COLOR)

#Discovered path by the agent
pathFound = trainAgent(alpha,b_1,b_2,decay,episodes,gamma,epsilon,path,init,init_theta,goal,limit,road)
print("\nOptimal path found by the agent: ")
print(pathFound)

#Plot the paths
plt.figure(1)
plt.title("Optimal path vs Agent performance")
plt.xlabel("Pixels of the picture in x-axis")
plt.ylabel("Pixels of the picture in y-axis")
plt.plot(path[:,0],path[:,1],label='Optimum')
plt.plot(pathFound[:,0],pathFound[:,1],color='y',label='Agent')
plt.xlim([0,630])
plt.ylim([0,630])
plt.legend()

# Store the data of the found path
df = pd.DataFrame({'x': pathFound[:,0], 'y': pathFound[:,1]})
df.to_csv('agent.csv',sep=';',index=False)

# End up closing all windows
cv2.destroyAllWindows()