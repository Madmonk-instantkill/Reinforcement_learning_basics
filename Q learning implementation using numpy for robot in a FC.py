#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


environment=np.ones(shape=(11,11))*(-100)
environment[0,5]=100
aisle={}
aisle[1]=[i for i in range(1,10)]
aisle[2]=[1,7,9]
aisle[3]=[i for i in range(1,8)]
aisle[3].append(9)
aisle[4]=[3,7]
aisle[5]=[i for i in range (0,11)]
aisle[6]=[5]
aisle[7]=[i for i in range (1,10)]
aisle[8]=[3,7]
aisle[9]=[i for i in range (0,11)]

for i in range(1,10):
    for j in aisle[i]:
        environment[i,j]=-1


# In[4]:


# actions= 4 ie left, right,up,down
q_values_matrix=np.zeros(shape=(11,11,4))
actions={0:"up",2:"down",3:"left",1:"right"}


# In[17]:


# helper function
def is_terminal(row,column):
    return (environment[row,column]==-100 or environment[row,column]==100)

def start_position():
    row,column=np.random.randint(11),np.random.randint(11)
    while is_terminal(row,column):
        row,column=np.random.randint(11),np.random.randint(11)
    return row,column
        
def choose_next_action(row,column,eps):
    if np.random.random() < eps:
        return np.argmax(q_values_matrix[row,column])
    else:
        return np.random.randint(4)
    
def get_new_location(current_row,current_col,action):
    new_row,new_col=current_row,current_col
    if actions[action]=="up" and new_row >0:
        new_row-=1
    elif actions[action]=="down" and new_row < 10:
        new_row+=1
    elif actions[action]=="right" and new_col < 10:
        new_col+=1
    elif actions[action]=="left" and new_col >0:
        new_col -=1
    return new_row,new_col  


def shortest_path(current_row,current_col):
    shortest_path=[]
    if is_terminal(current_row,current_col):
        return shortest_path
    else:
        shortest_path.append([current_row,current_row])
        while not is_terminal(current_row,current_col):
            #print(current_row,current_col)
            new_action=choose_next_action(current_row,current_col,1)
            new_row,new_col=get_new_location(current_row,current_col,new_action)
            current_row,current_col=new_row,new_col
            shortest_path.append([new_row,new_col])
            #print(is_terminal(shortest_path[-1][0],shortest_path[-1][1]))

    return shortest_path
        


# In[11]:


epochs=10000
epsilon=0.9
learning_rate=0.9
lam=0.9

for i in range(epochs):
    row,col=start_position()
    temp=[]
    temp.append([row,col])
    while not is_terminal(row,col):
        action=choose_next_action(row,col,epsilon)
        new_row,new_col=get_new_location(row,col,action)
        reward=environment[new_row,new_col]
        old_location_q_val=q_values_matrix[row,col,action]
        temporal_diff=reward-old_location_q_val+lam*np.max(q_values_matrix[new_row,new_col])    
        q_values_matrix[row,col,action]=old_location_q_val+learning_rate*temporal_diff
        row,col=new_row,new_col
        temp.append([row,col])
        #print(temp)
print("training complete")


# In[ ]:




