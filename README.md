# DRNE
The Implementation of "[Deep Recursive Network Embedding with Regular Equivalence](https://dl.acm.org/citation.cfm?doid=3219819.3220068)"(KDD 2018).

### Requirements
```
Python >= 3.5.2
scipy >= 0.19.1
numpy >= 1.13.1
tensorflow == 1.2.0
networkx >= 1.11
```

### Usage
##### Example Usage
```
python src/main.py --data_path dataset/barbell.edgelist --save_path result/barbell --save_suffix test \
      -s 16 -b 256 -lr 0.0025 --index_from_0 True
```
##### Full Command List
```
usage: Deep Recursive Network Embedding with Regular Equivalence
       [-h] [--data_path DATA_PATH] [--save_path SAVE_PATH]
       [--save_suffix SAVE_SUFFIX] [-s EMBEDDING_SIZE] [-e EPOCHS_TO_TRAIN]
       [-b BATCH_SIZE] [-lr LEARNING_RATE] [--undirected UNDIRECTED]
       [-a ALPHA] [-l LAMB] [-g GRAD_CLIP] [-K K]
       [--sampling_size SAMPLING_SIZE] [--seed SEED]
       [--index_from_0 INDEX_FROM_0]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Directory to load data.
  --save_path SAVE_PATH
                        Directory to save data.
  --save_suffix SAVE_SUFFIX
                        Directory to save data.
  -s EMBEDDING_SIZE, --embedding_size EMBEDDING_SIZE
                        the embedding dimension size
  -e EPOCHS_TO_TRAIN, --epochs_to_train EPOCHS_TO_TRAIN
                        Number of epoch to train. Each epoch processes the
                        training data once completely
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of training examples processed per step
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        initial learning rate
  --undirected UNDIRECTED
                        whether it is an undirected graph
  -a ALPHA, --alpha ALPHA
                        the rate of structure loss and orth loss
  -l LAMB, --lamb LAMB  the rate of structure loss and guilded loss
  -g GRAD_CLIP, --grad_clip GRAD_CLIP
                        clip gradients
  -K K                  K-neighborhood
  --sampling_size SAMPLING_SIZE
                        sample number
  --seed SEED           random seed
  --index_from_0 INDEX_FROM_0
                        whether the node index is from zero
```
### Cite
If you find this code useful, please cite our paper:
```
@inproceedings{tu2018deep,
  title={Deep recursive network embedding with regular equivalence},
  author={Tu, Ke and Cui, Peng and Wang, Xiao and Yu, Philip S and Zhu, Wenwu},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \&amp; Data Mining},
  pages={2357--2366},
  year={2018},
  organization={ACM}
}
```
