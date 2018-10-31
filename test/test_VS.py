import numpy as np
import os, sys
import networkx as nx
from scipy import stats
from scipy.spatial.distance import pdist, squareform


SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from src.utils import VS, load_embeddings
from src.network import read_network

def compare(S1, S2):
    S1 /= np.sum(S1)
    S2 /= np.sum(S2)
    return np.mean((S1-S2)**2)

def resort(rank):
    res = np.zeros_like(rank)
    for i, j in enumerate(rank):
        res[j] = i
    return res

def similarity(x, y):
    return stats.kendalltau(x, y)[0]

if __name__ == '__main__':
    dataset_name = 'jazz'
    embedding_size = 32
    methods = ['deepwalk', 'line', 'node2vec', 'struc2vec']+['eni_{}_{}_{}_{}'.format(lr, embedding_size, alpha, lamb) for lr in [0.001, 0.0025, 0.005] for alpha in [0.0, 0.01, 0.1, 1.0] for lamb in [0.0, 0.01, 0.1, 1.0]]
    methods = ['graphsage']
    #methods += ['eni_0.01_16_0.0_0.0', 'eni_0.05_16_0.0_0.0']
    save_path = 'result/{}'.format(dataset_name)
    embedding_filenames = [os.path.join(save_path, "baseline_{}".format(embedding_size), 
        "{}.embeddings".format(m)) for m in methods if not m.startswith('eni_')]+\
        [os.path.join(save_path, "{}".format(m), 'embeddings.npy') for m in methods if m.startswith('eni_')]
    embedding_filenames = [os.path.join(save_path, "baseline_{}".format(embedding_size), "{}.npy".format(m)) for m in methods]

    embeddings = [load_embeddings(name) for name in embedding_filenames]

    G = read_network('dataset/{}.edgelist'.format(dataset_name))
    A = nx.to_scipy_sparse_matrix(G, dtype=float)
    S = VS(A, 0.9, 100)
    S = (S+S.T)/2
    aS = np.argsort(S, None)
    arS = resort(aS)
    for i, e in enumerate(embeddings):
        if methods[i].startswith('eni') or methods[i] == 'struc2vec':
            E = -squareform(pdist(e, 'euclidean'))
        else:
            E = e.dot(e.T)
        aE = np.argsort(E, None)
        arE = resort(aE)
        print(methods[i], similarity(arS, arE), sep='\t')
