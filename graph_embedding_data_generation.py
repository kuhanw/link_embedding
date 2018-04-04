#!/usr/bin/env python
import pickle
import math
import random

import networkx as nx
import numpy as np
import tensorflow as tf

from joblib import Parallel, delayed
from graph_helpers import *

'''
randomWalk(graph, initial_node, step, max_step, path)

Function to take a random walk from a given node

graph: networkx graph, graph from which to random through
initial node: string, initial node to begin the walk
step: int, current step of walk
max_step: int, maximum number of steps to take in walk
path:, list, current path taken in the walk
'''
def randomWalk(initial_node, step, max_step, path):
 
    if step>= max_step: 
        return path
    
    adjacent_nodes = [i for i in web_graph[initial_node]]
    #Weights here are normalized with respect to global max as opposed to local max!
    #Therefore have to reweight

    if len(adjacent_nodes) == 0:
    
        while len(path)<=max_step:
            path.append(None)
        
        midpoint = len(path)//2
        if path[midpoint] is None:
            while path[midpoint] is None:
                path = np.roll(path, shift=1)
 
            #Give a 50% chance to switch target and context around
            #if random.choice([0, 1]) == 1:
            #    path = np.roll(path, shift=1)
 
        return path
    
    node_paths = web_graph[initial_node]
    
    adjacent_nodes_weights = [node_paths[i]['weight'] for i in node_paths]

    sum_weights = sum(adjacent_nodes_weights)

    adjacent_nodes_weights = [i/sum_weights for i in adjacent_nodes_weights]
       
    next_node = np.random.choice(adjacent_nodes, p=adjacent_nodes_weights)
    
    path.append(next_node)
    
    return randomWalk(next_node, step+1, max_step, path)

def main():

    max_step = 4# Window size and max_step must be connected
    vocabulary_size = len(nodes_only)
    n_epochs = 100 #This controls the number of walks from each node
    print ('start walks')
    
    for data in range(100):
        print ('data stack:%d' % data)
        random_walks = Parallel(n_jobs=-1, verbose=8, backend='multiprocessing')(delayed(randomWalk)(node, 0, max_step, [node]) for node in nodes_only for epoch in range(n_epochs))
   
        pickle.dump(random_walks, open('walk_data/random_walk_epoch_data_%d.pkl' % data, 'wb'))
        print('Walk saved')
         
if __name__ == "__main__":
    main()
