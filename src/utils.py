import numpy as np
import operator
import tensorflow as tf
import scipy
import networkx as nx
import sys, time, os

def load_from_wv_format(filename):
    with open(filename) as f:
        l = f.readline().split()
        total_num, embedding_size = int(l[0]), int(l[1])
        ls = list(map(lambda x: x.strip().split(), f.readlines()))
        total_num = max([int(line[0]) for line in ls])+1
        res = np.zeros((total_num, embedding_size), dtype=float)
        for line in ls:
            res[int(line[0])] = list(map(float, line[1:]))
    return res

def save_as_wv_format(filename, data):
    with open(filename, 'w') as f:
        nums, embedding_size = data.shape
        print(nums, embedding_size, file=f)
        for j in range(nums):
            print(j, *data[j], file=f)

def load_embeddings(filename, file_type=None):
    #print("load embeddings ", filename)
    if file_type is None:
        file_type = filename.strip().split('.')[-1]
    if file_type == 'embeddings':
        return load_from_wv_format(filename)
    elif file_type == 'npy':
        return np.load(filename)
    else:
        print('unsupported file type!')
        return None

def MSE(x, y):
    return np.mean((x-y)**2)

def print_array(X):
    a, b = X.shape
    print("\n".join(["\t".join(["{:.6e}"]*b)]*a).format(*X.flatten()))

def regularize_dataset(filename, output_filename, uniq=False, sort=True, label_filename=None, label_output_filename=None):
    data = np.loadtxt(filename).astype(int)
    if uniq:
        data = np.array(list(filter(lambda x: x[0] < x[1], data)))
    data_ids = list(set(data.flatten()))
    if sort:
        data_ids.sort()
    mapping = dict(zip(data_ids, range(len(data_ids))))
    res = np.vectorize(mapping.get)(data)
    np.savetxt(output_filename, res, fmt='%d')
    if label_filename is not None:
        labels = np.loadtxt(label_filename).astype(int)
        res_label_id = np.vectorize(mapping.get)(labels[:, 0])
        res_label = np.vstack((res_label_id, labels[:, 1])).T
        np.savetxt(label_output_filename, res_label, fmt='%d')

def init_embedding(degrees, degree_max, emb_size):
    return np.vstack([np.random.normal(i*1.0/degree_max, 0.001, emb_size) for i in degrees])

def selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.contrib.keras.activations.elu(x, alpha)

def gather_col(X, batch_size, l):
    if type(l) == list or type(l) == tuple:
        return tf.gather_nd(X, [[[i, j] for j in l] for i in range(batch_size)])
    else:
        return tf.gather_nd(X, [[i, l] for i in range(batch_size)])

def VS(A, alpha, iter_num=100):
    """the implement of Vertex similarity in networks"""
    assert 0 < alpha < 1
    assert type(A) is scipy.sparse.csr.csr_matrix
    lambda_1 = scipy.sparse.linalg.eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
    n = A.shape[0]
    d = np.array(A.sum(1)).flatten()
    d_inv = np.diag(1./d)
    dsd = np.random.normal(0, 1/np.sqrt(n), (n, n))
    #dsd = np.zeros((n, n))
    I = np.eye(n)
    for i in range(iter_num):
        dsd = alpha/lambda_1*A.dot(dsd)+I
        if i % 10 == 0:
            print('VS', i, '/', iter_num)
    return d_inv.dot(dsd).dot(d_inv)

def generate_graph(N, d, p):
    return nx.random_graphs.watts_strogatz_graph(N, d, p)


class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0
        self.last_time = 0.0
        self.last_percent = 0.0

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        now_time = time.time()
        left_time = (now_time-self.last_time)/(percent-self.last_percent)*(100.0-percent)
        self.last_time, self.last_percent = now_time, percent
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                + '%.2f' % percent + '%, left time: ' + '%.2f' % left_time + 's\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()

    def close(self, words='done'):
        print('')
        print(words)
        self.i = 0

def load_graphsage_embedding(data_dir, result_dir):
    X = np.load(os.path.join(data_dir, 'val.npy'))
    ind = np.loadtxt(os.path.join(data_dir, 'val.txt'), dtype=int)
    emd_size = X.shape[1]
    X = X[ind]
    np.save(os.path.join(result_dir, 'baseline_{}/graphsage.npy'.format(emd_size)), X)

if __name__ == '__main__':
    #with tf.Session().as_default():
    #    print(selu(np.array([1.2, 2.1, 21.1, -3.23])).eval())
    #regularize_dataset('dataset/jazz.edgelist', 'dataset/jazz.edgelist')
    #regularize_dataset('dataset/brazil-fligths.edgelist.txt', 'dataset/brazil-flights.edgelist', label_filename='dataset/labels-brazil-airports.txt', label_output_filename='dataset/labels_brazil-flights.txt')
    #regularize_dataset('dataset/europe-flights.edgelist.txt', 'dataset/europe-flights.edgelist', label_filename='dataset/labels-europe-airports.txt', label_output_filename='dataset/labels_europe-flights.txt')
    #regularize_dataset('/home/tuke/Centrality_Data/Flickr2_1.txt', 'dataset/flickr.edgelist', uniq=True)
    #regularize_dataset('/mnt/data/zhangziwei/WeChat_Dynamic/WeChat_time0')
    load_graphsage_embedding('../code/GraphSAGE/Jazz/unsup-dataset/graphsage_mean_small_0.000010/', 'result/jazz/')
    #G = generate_graph(10**7, 300, 0.1)
    #with open('dataset/toy.edgelist', 'w') as f:
    #    for i, j in G.edges():
    #        print(i, j, file=f)
