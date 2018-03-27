#!/usr/bin/env python
'''
A small web crawling script
Kuhan Wang 18-01-15

This script will start from initial site and search for all 
absolute hrefs (no relative paths) and descend within. 
The return condition can be a max depth.
Alternatively, the crawler will also keep a record of 
domains it has visited (i.e. www.nytimes.com for sites such as www.nytimes.com/world)
once any domain has been visited "n" times it will continue the search laterally
as opposed to increasing the depth.

To do: Need to clean up error handling
'''

from lxml import html
from joblib import Parallel, delayed

import urllib
import urllib.request
import time
import tldextract

import datetime
import pickle
import numpy as np
import pandas as pd

filter_formats = ['.pdf', '.png', '.txt', '.svg', '.jpg', '.gz', '.md', '.zip']

filter_blacklist = ['t.co', 'mailto:?']

n_calls  = 0

def grabDomainRoot(url):
    base_url = "{0.scheme}://{0.netloc}/".format(urllib.parse.urlsplit(url))
    
    if 'http' in base_url:
        try:
            base_url = [i for i in base_url.split('/') if len(i)>0]
            base_url = base_url[1]
        except:
            #print ('NO BASE URL')
            return None
    
    return base_url

#Need to explore with the least amount of filters because otherwise the crawler may get stuck in closed loops easily
#Can prune during graph construction instead by applying domain filters
def grabLinks(dom, filter_domains):
    
    links = [i for i in dom.xpath('//a/@href') if 'http' in i]
    new_links = links
    
    #Filter not html pages
    new_links = [i for i in new_links if True not in [extension in i for extension in filter_formats]]
    #Filter black list of web pages
    new_links = [i for i in new_links if True not in [web in i for web in filter_blacklist]]
  
    return new_links


def recursiveDescent(root, initial_html, current_depth, max_depth, graph, max_graph_size, domains, max_domains):
    
    if len(graph)%20==0 and len(graph)>0:

        #print ('SAVING, n_calls %d' % n_calls)
     	#Why is this not deleting the prior iteration, it appears to be appending to graph?
        domain_entity = tldextract.extract(root)
        domain_entity = domain_entity.domain
        pickle.dump(graph, open('crawler_results/graph_size_%d_%s.pkl' % (len(graph), domain_entity), 'wb'))

    #Max retain a max depth to prevent stack overflow
    if len(graph)>max_graph_size: #this is a tunable parameter
        #print ('MAX GRAPH SIZE REACHED')
        return graph, domains

    #print ('CONNECT TO URL')
    try:
        connection = urllib.request.urlopen(initial_html, timeout=6)
    except:
        #print ('TIME OUT')
        return graph, domains
            
    #print ('HTML TO STRING')
    try:
        read_connect = connection.read()
    except:
        #print ('FAILED TO READ CONNECTION')
        return graph, domains

    #print ('PARSE HTML FROM STRING')
    try:
        dom =  html.fromstring(read_connect)
    except:
        #print ('FAILED TO PARSE FROM STRING')
        return graph, domains

    links = grabLinks(dom, domains)
    
    for link in links:

        base_url = grabDomainRoot(link)
    
        if base_url is None: 
            #print ('NO BASE URL PASSING')
            continue
        
        #print ('from:%s, to:%s, base_url:%s, depth:%d, max_depth:%d, graph_size:%d' %\
        #       (initial_html, link, base_url, current_depth, max_depth, len(graph)))
        
        if initial_html in graph.keys():
            connections = graph[initial_html].transpose()[0]
            if link in connections:
                #print ('PATH EXISTS, PASSING')
                continue

        if base_url in domains.keys(): 
            pass
        else: 
            domains[base_url]=1
    
        if initial_html not in graph.keys():
            graph[initial_html] = np.array([link, base_url, datetime.datetime.now()])

        else:
            graph[initial_html] = np.append(graph[initial_html], [link, base_url, datetime.datetime.now()])

        if current_depth+1>max_depth:
            #print ('MAX DEPTH EXCEEDED, PASSING')
            continue

        elif domains[base_url]>=max_domains:#this a tunable parameter
            #print ('BASE URL EXCEEDED, PASSING')
            continue   
     
        else:
            #print ('DESCEND')
            domains[base_url]+=1
            recursiveDescent(root, link, current_depth+1, max_depth, graph, max_graph_size, domains, max_domains)

        time.sleep(0.1)

    return graph, domains

def crawlLink(initial_html, max_depth, max_graph_size, max_domains):

    graph = {}
    domains = {}

    print ('CURRENTLY PROCESSING ##:%s' % initial_html)
 #   try:
    paths_list = recursiveDescent(initial_html, initial_html, 0, max_depth, graph, max_graph_size, domains, max_domains)
        
    domain_entity = tldextract.extract(initial_html)
    domain_entity = domain_entity.domain
    pickle.dump(graph, open('crawler_results/graph_calls_final_%s.pkl' % (domain_entity), 'wb'))
#    except:
#        print ('FAILED TO PROCESS ##:%s' % initial_html)
  
def main():
    
    max_depth = 10
    max_domains = 2
    max_graph_size = 20

    df = pd.read_csv('sp500_links.csv')
    df = df[120:124]
    
    results = Parallel(n_jobs=4, verbose=8)(delayed(crawlLink)(link, max_depth, max_graph_size, max_domains) for link in df['link'].values)


    #for idx_link, link in enumerate(df['link'].values):
        #if idx_link!=0:break
        #graph = {}
        #domains = {}
        #initial_html = link
        #print ('CURRENTLY PROCESSING ##:%s' % initial_html)
        #try:
        #    paths_list = recursiveDescent(initial_html, initial_html, 0, max_depth, graph, max_graph_size, domains, max_domains)
        
        #    domain_entity = tldextract.extract(link)
        #    domain_entity = domain_entity.domain
        #    pickle.dump(graph, open('crawler_results/graph_calls_final_%s.pkl' % (domain_entity), 'wb'))
        #except:
        #    print ('FAILED TO PROCESS ##:%s' % initial_html)
  
if __name__ == "__main__":
    main()
