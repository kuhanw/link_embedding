import networkx as nx
import pickle

web_graph = pickle.load(open('graph_directed_weighted_v8.pkl', 'rb'))
nodes_only = pickle.load(open('nodes_only_directed_v8.pkl', 'rb'))

list_of_nodes = [i for i in web_graph.nodes()]

domain_inv_map = {idx:i for idx, i in enumerate(list_of_nodes)}
domain_map = {i:idx for idx, i in enumerate(list_of_nodes)}
domain_map[None] = -1

print('%d nodes, %d key_domain dict, %d key_nodes only' % (len(list_of_nodes), len(domain_inv_map), len(nodes_only)))


