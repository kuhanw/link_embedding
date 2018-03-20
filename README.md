# link_embedding
Embedding link connections from a webcrawler

This is a work in process. What happens if we try to create vector embeddings based on a graph where nodes are websites and edges are links?
First we will use a webcrawler to map out a tiny portion of the internet as data. We construct an undirected graph from the link connections. To reduce the size of the graph, I construct the graph of domains as opposed to urls. Thus nodes represents domains and edges connections between domains, as opposed to direct urls. A visualization of this can be seen below.

<p align="center">
<img src="./domain_graph_undirected.png">
</p>
