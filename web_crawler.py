import requests
from lxml import html
import urllib
import time
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import json

import pickle

filter_formats = ['.pdf', '.png', '.txt', '.svg', '.jpg']

filter_blacklist = ['t.co']

def grabDomainRoot(url):
    base_url = "{0.scheme}://{0.netloc}/".format(urllib.parse.urlsplit(url))
    
    if 'http' in base_url:
        base_url = [i for i in base_url.split('/') if len(i)>0]
        base_url = base_url[1]
    
    #base_url = base_url.split('.')
    
    return base_url

#Need to explore with the least amount of filters because otherwise the crawler may get stuck in closed loops easily
#Can prune during graph construction instead by applying domain filters
def grabLinks(dom, base_url, filter_domains):
    
    links = list(set([i for i in dom.xpath('//a/@href') if 'http' in i]))
    new_links = links
    
    #This is redunant with the filter for domains
    #new_links = [i for i in links if base_url not in i]
    
    #Filter not html pages
    new_links = [i for i in new_links if True not in [extension in i for extension in filter_formats]]
    #Filter black list of web pages
    new_links = [i for i in new_links if True not in [web in i for web in filter_blacklist]]
    #Filter for domains
    #new_links = [i for i in new_links if True not in [domain in i for domain in filter_domains]]

    return new_links

def recursiveDescent(initial_html, current_depth, max_depth):
    global n_calls

    n_calls+=1
    
    if n_calls % 100 ==0:
        print (n_calls)
        pickle.dump(graph, open('graph.pkl', 'wb'))
        pickle.dump(domains, open('domains.pkl', 'wb'))

    #Break condition should go on top of recursive fn
    #In order to get diversity in the crawler we should break based on number of domain pages reached
    #not the max_depth
    #if current_depth>max_depth: 
        #print ('MAX DEPTH REACHED')
   #     return None

    base_url = grabDomainRoot(initial_html)
    
    if len(base_url) == 0 is None: 
        print ('NO BASE URL')
        return None
   
    if base_url in domains.keys(): 
        #print ('BASE URL IN DOMAIN KEYS')
 
        if domains[base_url]>=10:
            print ('BASE URL EXCEEDED')
            return None
        
        else:
            #print ('BASE URL INCREASED')
            domains[base_url]+=1
    
    else: 
        domains[base_url]=1
    
    try:
        connection = urllib.request.urlopen(initial_html)
        dom =  html.fromstring(connection.read())
    
    except:
        print ('FAILED TO CONNECT')
        return None
    
    links = grabLinks(dom, base_url, domains)
    
    if len(links)==0: return None
    
    #print ('GOING INTO LOOP', initial_html, links)
    
    for link in links:
        
        if initial_html in graph.keys():
            if link in graph[initial_html]:
                print ('PATH EXISTS, PASSING')
                continue
        
        print ('DESCEND', initial_html, link, domains, current_depth, max_depth, n_calls)
    
        if initial_html not in graph.keys():
            print ('APPEND NEW LINK')
            graph[initial_html] = set([link])
        else:
            print ('APPEND TO EXISTING LINK')
            graph[initial_html] = graph[initial_html].union(set([link]))   

        recursiveDescent(link, current_depth+1, max_depth)

        time.sleep(0.1)


if __name__ == "__main__":

	#graph = {}
	graph = pickle.load(open('graph_china.pkl', 'rb'))
	domains = {}
	n_calls  = 0

	recursiveDescent('http://www.yahoo.com', 0, 5)
