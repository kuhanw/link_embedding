import networkx as nx
import pickle

web_graph = pickle.load(open('graph_directed_weighted_v8.pkl', 'rb'))
nodes_only = pickle.load(open('nodes_only_directed_v8.pkl', 'rb'))

#These can only be made once!
domain_inv_map = pickle.load(open('graph_domain_inv_map_v8.pkl', 'rb'))
domain_map = pickle.load(open('graph_domain_map_v8.pkl', 'rb'))

keys = domain_map.keys()

def fixNone(path):
    return [i if i==i else None for i in path]

def mapValues(path):
    return [domain_map[i] if i in keys else -1 for i in path]
