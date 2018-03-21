# link_embedding [Work in progress]
Embedding link connections from a webcrawler

This is a work in process. What happens if we try to create vector embeddings based on a graph where nodes are websites and edges are links?

First we will use a webcrawler to map out a tiny portion of the internet as data. We construct a directed graph from the link connections. 

To reduce the size of the graph, I construct the graph of domains as opposed to urls. I define the domain of an url according to Python as: 

```
def grabDomainRoot(url):
    base_url = \"{0.scheme}://{0.netloc}/\".format(urllib.parse.urlsplit(url))`    
    if 'http' in base_url:
        try:
            base_url = [i for i in base_url.split('/') if len(i)>0]
            base_url = base_url[1]
        except:
            return None
    
    return base_url,
```
thus it is possible to capture one website as two "domains", (i.e. corporate.website.com and press.website.com).

In the resultant graph nodes represents domains and edges connections between domains, as opposed to direct urls. A visualization of this can be seen below. [Replace with directed graph]

<p align="center">
<img src="./domain_graph_undirected.png">
</p>

We use the idea behind [arXiv:1403.6652 [cs.SI]](https://arxiv.org/abs/1403.6652) and take a random walk about nodes in the graph. The length of the walk represents the target and context window data for which we will embed in vector space.

It is important to take care here, as the link connections form a directed graph, is not in fact possible to walk from every node (as it would be in an undirected graph as in the original DeepWalk paper). Certain nodes in the graph will be terminal. Therefore, the set of "walkable" nodes is a fraction of all the nodes.

We construct a standard vector embedding analogous to word2vec in Tensorflow, the full code is as shown in `graph_embedding.ipynb`. The resultant vectors tell us the similarity between the items. 

In this case we would expect the similarity to be defined as how similar two domains are based on the connections between them. We can construct a full similarity matrix to look at this. 

In the table below, I output a few examples showing the embedding can reasonably capture the local similarity amongst nodes.

Source Node| Node 1| Score 1|Node 2 | Score 2|Node 3| Score 3| 
---|--- |---|--- |---|--- |---|
'marketplace.usatoday.com'|'developer.usatoday.com'|0.15|'arcade.usatoday.com'|0.23|'travel.usatoday'|0.35
'world.ign.com'|'world.ziffdavis.com'|0.25|'api.ign.com'|0.67|'africa.ign.com'|0.74
'www.cnbcstore.com'|'www.usanetworkstore.com'|0.39|'www.msnbcstore.com'|0.48|'www.shopbybravo.com'|0.48

It is clear the vector embedding can group together similar domains by their connections. The score represents the angle between their embedded vectors.

One flaw in this method is propensity to group together obvious connections such as various local variations of a website (as in the case of IGN.) This may be rectified by reconstructing the domain graph in a more nuauced way by merging variations of a domain into a single node through more sophisticated syntax processing.

