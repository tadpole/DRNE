import numpy as np
import os, sys
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn import preprocessing

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, '..')))
from src.network import load_centrality
from src.utils import load_embeddings, MSE, print_array

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    embedding_size = int(sys.argv[2])
    methods = ['deepwalk', 'line', 'node2vec', 'struc2vec', 'eni_1_1']
    methods = ['graphsage']
    centrality_path = 'result/{}/data'.format(dataset_name)
    if dataset_name.endswith('flights'):
        Y = np.loadtxt('dataset/labels_{}.txt'.format(dataset_name)).astype(int)
        labels = Y[:, 1]
    else:
        num_class = 4
        ground_truth = 'spread_number'
        Y = load_centrality(centrality_path, ground_truth)
        rY = np.sort(Y[:, 1])
        threshold = [rY[int(i*len(rY)/num_class)] for i in range(num_class)]+[rY[-1]+1]
        labels = np.array([len(list(filter(lambda x: i<x, threshold)))-1 for i in Y[:, 1]])
    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    save_path = 'result/{}'.format(dataset_name)
    embedding_filenames = [os.path.join(save_path, "baseline_{}".format(embedding_size), 
        "{}.embeddings".format(m)) for m in methods if not m.startswith('eni_')]+\
        [os.path.join(save_path, "{}_{}".format(m, embedding_size), 'embeddings.npy') for m in methods if m.startswith('eni_')]
    embedding_filenames = [os.path.join(save_path, "baseline_{}".format(embedding_size), "{}.npy".format(m)) for m in methods]

    embeddings = [load_embeddings(name)[Y[:, 0].astype(int)] for name in embedding_filenames]

    centrality_types = ['closeness', 'betweenness', 'eigenvector', 'kcore']
    centralities = [load_centrality(centrality_path, c)[Y[:, 0].astype(int), 1].reshape(-1, 1) for c in centrality_types]
    for c in centralities:
        c = c.reshape(-1, 1)
    #res = np.zeros((len(methods), len(centrality_types)))
    combine_centrality = np.hstack(centralities)
    centralities.append(combine_centrality)
    acc = []
    for _ in range(100):
        radio = 0.8
        index = np.random.permutation(range(len(labels)))
        th = int(radio*len(labels))
        temp_res = []
        for i in range(len(methods)):
            #lr = OneVsRestClassifier(svm.SVC(kernel='linear', C=0.025, probability=True))
            data = embeddings[i]
            lr = OneVsRestClassifier(LogisticRegression())
            lr.fit(data[index[:th]], labels[index[:th]])
            y_pred = lr.predict_proba(data[index[th:]])
            y_pred = lb.transform(np.argmax(y_pred, 1))
            #y_pred = lr.predict(embeddings[i][index[th:]])
            temp_res.append(np.sum(np.argmax(y_pred, 1) == np.argmax(labels[index[th:]], 1))/len(y_pred))
            #print(np.sum(y_pred == labels[index[th:]])/len(y_pred))
        for i in range(len(centralities)):
            data = centralities[i]
            lr = OneVsRestClassifier(LogisticRegression())
            lr.fit(data[index[:th]], labels[index[:th]])
            y_pred = lr.predict_proba(data[index[th:]])
            y_pred = lb.transform(np.argmax(y_pred, 1))
            #y_pred = lr.predict(embeddings[i][index[th:]])
            temp_res.append(np.sum(np.argmax(y_pred, 1) == np.argmax(labels[index[th:]], 1))/len(y_pred))
            #print(np.sum(y_pred == labels[index[th:]])/len(y_pred))
        acc.append(temp_res)
    acc = np.array(acc)
    #print(acc)
    print(np.mean(acc, 0))
