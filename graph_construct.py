#!/usr/bin/env python
import pickle

import tldextract
import os

import pandas as pd
import networkx as nx
import numpy as np
import datetime as dt

black_list = ['@', ':///', 'onlinecasinoreports']

'''
Fill a graph using crawled data

[DOCUMENTATION NEEDED]

'''

def fillGraph(web_graph, graph_file, edge_entries, nodes_only, domain_graph=True):
    
    for node in graph_file.keys():
        if domain_graph:
            domain_node = tldextract.extract(node)
            domain_node = domain_node.domain
        else:
            domain_node = node

        domain_node = domain_node.lower()

        for idx in range(0, len(graph_file[node]), 3):
            key = graph_file[node][idx]

            if domain_graph:
                domain_key = tldextract.extract(key)
                domain_key = domain_key.domain
            else:
                domain_key = key

            if domain_node == domain_key: 
                continue        

            if domain_node is None or domain_key is None: 
                continue
            
            if len(domain_node)==0 or len(domain_key)==0:
                #print ('domain:%s, node:%s, key:%s' % (domain_node, str(node), str(key)))
                continue
                
            if True in [i in domain_node for i in black_list] or True in [i in domain_key for i in black_list]:
                continue

            domain_key = domain_key.lower()

            edge_connection = domain_node + '--' + domain_key
            if edge_connection not in edge_entries.keys():
                edge_entries[edge_connection]=1
            else:
                edge_entries[edge_connection]+=1

            weight = edge_entries[edge_connection]

            web_graph.add_weighted_edges_from([(domain_node, domain_key, weight)])

            nodes_only.append(domain_node)
    
    return web_graph
    
def main():

    df_sp500 = pd.read_csv('sp500_links.csv')
    edge_entries = {}
    domain_graph = True
    nodes_only = []
    web_graph = nx.DiGraph()
    graph_file = []
    
    for idx, file in enumerate(os.listdir('./crawler_results/')):
        #if idx>5: break
        if 'final' not in file: 
            continue
        if 'depth_3' not in file:
            continue

        print ('Adding %s to graph' % file)
        graph_file = pickle.load(open('./crawler_results/' + file, 'rb'))

        if len(web_graph.nodes())==0:
            directed_graph = fillGraph(web_graph, graph_file, edge_entries, nodes_only, domain_graph)
        else:
            directed_graph = fillGraph(directed_graph, graph_file, edge_entries, nodes_only, domain_graph)

    #edges = weights_list[0]
    #weights = weights_list[1]/np.max(weights[1])#

    list_of_nodes = [i for i in web_graph.nodes()]

    domain_inv_map = {idx:i for idx, i in enumerate(list_of_nodes)}
    domain_map = {i:idx for idx, i in enumerate(list_of_nodes)}
    domain_map[None] = -1

    no_next_neighbors = []
    no_neighbors = []
    for node in list_of_nodes:
    
        neighbors = [i for i in web_graph.neighbors(node)]
        for neighbor in neighbors:
            next_neighbors = [i for i in web_graph.neighbors(neighbor)] 
        try:
            next_neighbor_counts = [len(i) for i in next_neighbors]
        except:
            next_neighbor_counts = [0]
        
        if sum(next_neighbor_counts)==0:
            no_next_neighbors.append(node)
        
        if len(neighbors)==0:
            no_neighbors.append(node)    

    nodes_only = list(set(nodes_only))

    print('%d nodes, %d key_domain dict, %d key_nodes only' % (len(list_of_nodes), len(domain_inv_map), len(nodes_only)))

    print ('Dump pickles')

    notes='depth_3'

    pickle.dump(web_graph, open('graph_directed_weighted_v8_%s.pkl' % notes, 'wb'))

    pickle.dump(nodes_only, open('nodes_only_directed_v8_%s.pkl' % notes, 'wb'))

    list_of_nodes = [i for i in web_graph.nodes()]

    domain_inv_map = {idx:i for idx, i in enumerate(list_of_nodes)}
    domain_map = {domain_inv_map[key]:key for key in domain_inv_map.keys()}
    domain_map[None] = -1
    pickle.dump(domain_inv_map, open('graph_domain_inv_map_%s_v8.pkl' % notes, 'wb'))
    pickle.dump(domain_map, open('graph_domain_map_%s_v8.pkl' % notes, 'wb'))

    nx.write_edgelist(web_graph, 'graph_edgelist_%s_v8.txt' % notes, delimiter='|')    

if __name__ == "__main__":
    main()
