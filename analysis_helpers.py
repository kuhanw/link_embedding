import numpy as np


def findTopN(vec, embeddings):
    inner_prod = [np.arccos(np.dot(vec, i)) for i in embeddings]
    inner_prod = [i if i==i else 0 for i in inner_prod]
    top_n = np.argsort(inner_prod)
    
    return top_n