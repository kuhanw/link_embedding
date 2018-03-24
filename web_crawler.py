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
            print ('NO BASE URL')
            return None
    
    return base_url

#Need to explore with the least amount of filters because otherwise the crawler may get stuck in closed loops easily
#Can prune during graph construction instead by applying domain filters
def grabLinks(dom, base_url, filter_domains):
    
    links = [i for i in dom.xpath('//a/@href') if 'http' in i]
    new_links = links
    
    #Filter not html pages
    new_links = [i for i in new_links if True not in [extension in i for extension in filter_formats]]
    #Filter black list of web pages
    new_links = [i for i in new_links if True not in [web in i for web in filter_blacklist]]
  
    return new_links


def recursiveDescent(initial_html, current_depth, max_depth, max_domains, graph, domains, n_calls):
    #global n_calls
    print ('CURRENT DEPTH', current_depth)	
   
    #if n_calls % 2 == 0 and n_calls>0:
    if n_calls>5:
        print ('MAX CALLS REACHED')
        print ('SAVING, n_calls %d' % n_calls)
     	#Why is this not deleting the prior iteration, it appears to be appending to graph?
        domain_entity = tldextract.extract(initial_html)
        domain_entity = domain_entity.domain
        pickle.dump(graph, open('crawler_results/graph_calls_%d_%s.pkl' % (n_calls, domain_entity), 'wb'))

	#Testing a trigger	
        return graph, domains

    #Max retain a max depth to prevent stack overflow
    if current_depth>=max_depth: #this is a tunable parameter
        print ('MAX DEPTH REACHED')
        return graph, domains

    base_url = grabDomainRoot(initial_html)
    
    if base_url is None: 
        print ('NO BASE URL')
        return graph, domains
   
    if base_url in domains.keys(): 
        print ('BASE URL IN DOMAIN KEYS')
 
        if domains[base_url]>=max_domains:#this a tunable parameter
            print ('BASE URL EXCEEDED')
            return graph, domains
        
        else:
            print ('BASE URL INCREASED')
            domains[base_url]+=1
    
    else: 
        domains[base_url]=1
    
    
    print ('CONNECTING TO: ', initial_html)
    try:
        connection = urllib.request.urlopen(initial_html, timeout=6)
    except:
        print ('TIME OUT')
        return graph, domains
            
    print ('HTML TO STRING')
    try:
        read_connect = connection.read()
    except:
        print ('FAILED TO READ CONNECTION')
        return graph, domains

    print ('PARSE HTML FROM SRING')
    try:
        dom =  html.fromstring(read_connect)
    except:
        print ('FAILED TO PARSE FROM STRING')
        return graph, domains


    print ('GRAB LINKS')
    links = grabLinks(dom, base_url, domains)
    
    if len(links)==0: 
        print ('NO LINKS AFTER FILTERING')
        return graph, domains
    
    for link in links:
        
        if initial_html in graph.keys():
            connections = graph[initial_html].transpose()[0]
            if link in connections:
                print ('PATH EXISTS, PASSING')
                continue
        
        print ('DESCEND', initial_html, link, current_depth, max_depth, max_domains, n_calls)
    
        if initial_html not in graph.keys():
            graph[initial_html] = np.array([link, base_url, datetime.datetime.now()])

        else:
            graph[initial_html] = np.append(graph[initial_html], [link, base_url, datetime.datetime.now()])

        return recursiveDescent(link, current_depth+1, max_depth, max_domains, graph, domains, n_calls+1)

        time.sleep(0.1)

def main():
    
    max_depth = 5
    max_domains = 3  
    df = pd.read_csv('sp500_links.csv')

    for idx_link, link in enumerate(df['link'].values):
        if idx_link!=0:break
        graph = {}
        domains = {}
        initial_html = link
        print ('CURRENTLY PROCESSING ##:%s' % initial_html)
        n_calls = 0
        paths_list = recursiveDescent(initial_html, 0, max_depth, max_domains, graph, domains, n_calls)
        print (paths_list)

if __name__ == "__main__":
    main()
