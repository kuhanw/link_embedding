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
import time
import datetime
import pickle

filter_formats = ['.pdf', '.png', '.txt', '.svg', '.jpg', '.gz', '.md', '.zip']

filter_blacklist = ['t.co']

n_calls  = 0

graph = {}
domains = {}

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
    
    if base_url is None: 
        print ('NO BASE URL')
        return None
   
    if base_url in domains.keys(): 
        print ('BASE URL IN DOMAIN KEYS')
 
        if domains[base_url]>=10:
            print ('BASE URL EXCEEDED')
            return None
        
        else:
            print ('BASE URL INCREASED')
            domains[base_url]+=1
    
    else: 
        domains[base_url]=1
    
    try:
        print ('CONNECTING TO: ', initial_html)
        try:
            connection = urllib.request.urlopen(initial_html, timeout=6)
        except:
            print ('TIME OUT')
            return None
            
        print ('HTML TO STRING')
        try:
            try:
                read_connect = connection.read()
            except:
                print ('FAILED TO READ CONNECTION')
                return None
            try:
                dom =  html.fromstring(read_connect)
            except:
                print ('FAILED TO PARSE FROM STRING')
                return None
        except:
            print ('HTML TO STRING FAILED')
            return None
    except:
        print ('FAILED TO CONNECT')
        return None
    print ('GRAB LINKS')
    links = grabLinks(dom, base_url, domains)
    
    if len(links)==0: 
        print ('NO LINKS AFTER FILTERING')
        return None
    
   # print ('GOING INTO LOOP', initial_html, links)
    
    for link in links:
        
        if initial_html in graph.keys():
            connections = graph[initial_html].transpose()[0]
            if link in connections:
                print ('PATH EXISTS, PASSING')
                continue
        
        print ('DESCEND', initial_html, link, current_depth, max_depth, n_calls)
    
        if initial_html not in graph.keys():
            graph[initial_html] = np.array([link, datetime.datetime.now()])

        else:
            #connections = graph[initial_html].transpose()[0]
            #if link not in connections:
            graph[initial_html] = np.append(graph[initial_html], [link, datetime.datetime.now()])
            #else:
                #print('PATH EXISTS')
                #return None
            #graph[initial_html] = graph[initial_html].union(set([link]))   

        recursiveDescent(link, current_depth+1, max_depth)

        time.sleep(0.1)

def main():

	recursiveDescent('http://www.nytimes.com/', 0, 5)

if __name__ == "__main__":
    main()

