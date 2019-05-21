import argparse
import numpy as np
import time

import tensorflow as tf

import network
from eni import eni

def parse_args():
    parser = argparse.ArgumentParser("Deep Recursive Network Embedding with Regular Equivalence")
    parser.add_argument('--data_path', type=str, help='Directory to load data.')
    parser.add_argument('--save_path', type=str, help='Directory to save data.')
    parser.add_argument('--save_suffix', type=str, default='eni', help='Directory to save data.')
    parser.add_argument('-s', '--embedding_size', type=int, default=16, help='the embedding dimension size')
    parser.add_argument('-e', '--epochs_to_train', type=int, default=20, help='Number of epoch to train. Each epoch processes the training data once completely')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Number of training examples processed per step')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0025, help='initial learning rate')
    parser.add_argument('--undirected', type=bool, default=True, help='whether it is an undirected graph')
    parser.add_argument('-a', '--alpha', type=float, default=0.0, help='the rate of structure loss and orth loss')
    parser.add_argument('-l', '--lamb', type=float, default=0.5, help='the rate of structure loss and guilded loss')
    parser.add_argument('-g', '--grad_clip', type=float, default=5.0, help='clip gradients')
    parser.add_argument('-K', type=int, default=1, help='K-neighborhood')
    parser.add_argument('--sampling_size', type=int, default=100, help='sample number')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--index_from_0', type=bool, default=True, help='whether the node index is from zero')
    return parser.parse_args()

def main(args):
    np.random.seed(int(time.time()) if args.seed == -1 else args.seed)
    graph = network.read_from_edgelist(args.data_path, index_from_zero=args.index_from_0)
    network.sort_graph_by_degree(graph)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config) as sess, tf.device('/gpu:2'):
        alg = eni(graph, args, sess)
        print("max degree: {}".format(alg.degree_max))
        alg.train()
        alg.save()

if __name__ == '__main__':
    main(parse_args())
