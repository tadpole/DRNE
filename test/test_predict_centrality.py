import numpy as np
import sys, os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from src.utils import load_embeddings, MSE, print_array
from src.network import load_centrality

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    embedding_size = int(sys.argv[2])
    #methods = ['deepwalk', 'line', 'node2vec', 'struc2vec', 'SVD', 'eniws', 'eni_6_1', 'eni_6_2', 'eni_7_1', 'eni_8_1', 'eni_9_1']
    #methods = ['deepwalk', 'line', 'node2vec', 'struc2vec', 'SVD', 'eniws', 'eni_1']
    #methods = ['deepwalk', 'line', 'node2vec', 'struc2vec']+['eni_{}_{}_{}_{}'.format(lr, embedding_size, alpha, lamb) for lr in [0.001, 0.0025, 0.005] for alpha in [0.0, 0.01, 0.1, 1.0] for lamb in [0.0, 0.01, 0.1, 1.0]]

    methods = ['graphsage']
    centrality_types = ['degree', 'closeness', 'betweenness', 'eigenvector', 'kcore']
    #centrality_types = ['spread_number']
    centrality_path = 'result/{}/data'.format(dataset_name)
    save_path = 'result/{}'.format(dataset_name)
    embedding_filenames = [os.path.join(save_path, "baseline_{}".format(embedding_size), 
        "{}.embeddings".format(m)) for m in methods if not m.startswith('eni_')]+\
         [os.path.join(save_path, "{}".format(m), 'embeddings.npy') for m in methods if m.startswith('eni_')]

    embedding_filenames = [os.path.join(save_path, "baseline_{}".format(embedding_size), "{}.npy".format(m)) for m in methods]

    embeddings = [load_embeddings(name) for name in embedding_filenames]
    centralities = [load_centrality(centrality_path, c) for c in centrality_types]
    res = np.zeros((len(methods), len(centrality_types)))
    for i in range(len(methods)):
        for j in range(len(centrality_types)):
            lr = LinearRegression(n_jobs=-1)
            y_pred = cross_val_predict(lr, embeddings[i][centralities[j][:, 0].astype(int)], centralities[j][:, 1])
            res[i, j] = MSE(y_pred, centralities[j][:, 1])/np.mean(centralities[j][:, 1])
            #res[i, j] = np.mean(abs((y_pred-centralities[j][:, 1])/(centralities[j][:, 1]+1e-10)))
    print_array(res)
