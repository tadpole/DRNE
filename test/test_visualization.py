import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from src.utils import load_embeddings
from src.network import load_centrality


def plot_embedding_2(embeddings, labels=None, save_path=None):
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(embeddings)):
        embedding = embeddings[i]
        plt.figure(3)
        color_map = np.array(['xkcd:royal blue', 'xkcd:greenish', 'xkcd:blood red', 'xkcd:dark sky blue', 'xkcd:yellow ochre', 'xkcd:purple', 'xkcd:light grey'])
        index = [0]*9+[1, 2, 3, 4, 5, 6, 6, 5 , 4, 3 ,2 ,1]+[0]*9
        colors = color_map[index]
        """
        plt.scatter(embedding[:10, 0], embedding[:10, 1], color='r')
        plt.scatter(embedding[10:20, 0], embedding[10:20, 1], color='b')
        plt.scatter(embedding[20:, 0], embedding[20:, 1], color='y')
        """
        plt.scatter(embedding[:, 0], embedding[:, 1], color=colors)
        plt.savefig(os.path.join(save_path, "{}.png".format(labels[i])), bbox_inches='tight')

def plot_embedding(embeddings, labels=None, save_path=None, colors=None):
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(embeddings)):
        embedding = embeddings[i]
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=30, cmap=plt.cm.get_cmap('RdYlBu'))
        plt.savefig(os.path.join(save_path, "{}.png".format(labels[i])), bbox_inches='tight')

if __name__ =='__main__':
    methods = ['deepwalk', 'line', 'node2vec', 'struc2vec', 'eniws', 'eni_1', 'eni_2']
    save_path = 'result/karate'
    embedding_size = 2
    embedding_filenames = [os.path.join(save_path, "baseline_{}".format(embedding_size), 
        "{}.embeddings".format(m)) for m in methods[:5]]+\
         [os.path.join(save_path, "{}_{}".format(m, embedding_size), 'embeddings.npy') for m in methods[5:]]
    embeddings = [load_embeddings(name) for name in embedding_filenames]
    centrality = load_centrality(os.path.join(save_path, 'data'), 'kcore')
    print(centrality)
    plot_embedding(embeddings, labels=methods, colors=centrality, save_path=os.path.join(save_path, 'test'))
    #plot_embedding_2(embeddings, labels=methods, save_path=os.path.join(save_path, 'test'))
