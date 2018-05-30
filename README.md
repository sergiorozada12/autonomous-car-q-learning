# autonomous-car-q-learning

The aim of this project is to show the basic implementation of Q-learning in a toy example. A car-like agent learns to interact with the environment, a grid-like top view map. Different implementations show the impact of different parametrizations. Ultimately, the behaviour of the agent is scaled up by using a Neural Network function approximator to deal with continuous states. In this case, the agent learns to plan ahead using the pixel values on the image. As long as  the amount of states is intractable for vanilla Q-learning, the agent needs to learn to generalize similar situations encountered in the past.

The project contains five different scripts:
1. vanilla_version.py: Plain version of Q-learning over a grid world	
2. gridsearch_version.py: Version with grid search of different combinations of alphas and gammas	
3. epsilon_version.py: Version with different policies implemented	
4. randomQ_version.py: Version with an initial non-zero Q-matrix	
5. ann_version.py: Version with the implementation of a neurla network as function approximator.

The libraries used in this project are:
* Pandas 0.20.3	
* Numpy 1.13.3	
* Matplotlib 2.1.0	
* Opencv 3.0.4	
* Scipy 0.19.1	
* Keras 2.1.5

PD: Each script is prepared to run autonomously
