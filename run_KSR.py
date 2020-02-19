# -*- coding: utf-8 -*-

import sys
import argparse
import pickle

sys.path.append('./src/')

import numpy as np, time
import pandas as pd
import torch
from evaluation import evaluate_sessions
import KVMN4rec as MN4rec

np.random.seed(666)


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge-Enhanced Sequential Representaion.")
    parser.add_argument('--data', default='year15',
                        help='dataset folder name under ../benchmarks')
    parser.add_argument('--type', default='B',
                        help='our dataset is divided')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size.')
    parser.add_argument('--dropout', type=int, default=0.2,
                        help='Dropout rate.')
    parser.add_argument('--out_dim', type=int, default=100,
                        help='Embedding size of output.')
    parser.add_argument('--num_neg', type=int, default=10,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--activation', nargs='?', default='tanh',
                        help='Specify activation function: sigmoid, relu, tanh, identity')
    parser.add_argument('--momentum', type=float, default=0.05,
                        help='Momentum as hyperprameter.')
    parser.add_argument('--argument', action='store_true',
                        help='use the method called "data argument" if true')
    parser.add_argument('--retrain', action='store_true',
                        help='use kg emb, the pretrained output of OpenKE (trans-E)')
    parser.add_argument('--reload', action='store_true',
                        help='restore saved params if true')
    parser.add_argument('--eval', action='store_true',
                        help='only eval once, non-train')
    parser.add_argument('--save', action='store_true',
                        help='if save model or not')
    parser.add_argument('--savepath',
                        help='for customization')
    parser.add_argument('--cuda', default='4',
                        help='gpu No.')
    return parser.parse_args()


args = parse_args()
if not args.savepath:
    args.savepath = args.data + args.type + '_' + ('retrain' if args.retrain else 'pretrain') + '.model'
print(args)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

dataset_path = '../benchmarks/kg_{}/{}/'.format(args.data, args.type)
train_data = dataset_path + 'train.csv'
valid_data = dataset_path + 'valid.csv'
test_data = dataset_path + 'test.csv'
File_ItemEmbedding = dataset_path + 'item_factors.txt'  # sys.argv[1]
File_KBItemEmbedding = dataset_path + 'e_emb.txt'  # sys.argv[2]
File_r_matrix = dataset_path + 'r_emb.txt'  # sys.argv[3]

size_dict = {'year20': {'A': 52126, 'B': 94794},
             'year15': {'A': 50531, 'B': 91701},
             'month15': {'A': 36845, 'B': 63937}}


def read_ItemEmbedding(File_ItemEmbedding, isItem=True):
    ItemEmb = {}
    f = open(File_ItemEmbedding)
    if isItem:
        reindex = pd.Series(data=reindex_dic['ItemId'].values, index=reindex_dic['OldItemId'].values)
    # length = int(f.readline().strip().split()[1])
    # a = set()
    for line in f.readlines():
        ss = line.strip().split()
        # if ss[0][0] != 'm':
        #    continue
        ItemId = reindex[int(ss[0])] if isItem else int(ss[0])
        if not isItem:
            if ItemId not in ItemEmbedding:
                continue
        t = []
        for i in range(1, len(ss)):
            t.append(float(ss[i]))
        ItemEmb[ItemId] = np.array(t, dtype=np.float32)
        # a.add(len(ss) - 1)
    # print(set(a))
    f.close()
    length = len(ss) - 1
    return ItemEmb, length


def read_r_matrix(filename):
    f = open(filename)
    r_matrix = []
    for line in f.readlines():
        ss = line.strip().split()
        t = []
        for i in ss:
            t.append(float(i))
        r_matrix.append(np.array(t, dtype=np.float32))
    f.close()
    return np.array(r_matrix, dtype=np.float32)


def process(data, argument=False):
    data.columns = ['SessionId', 'OldItemId']
    data['Time'] = data.index
    data = pd.merge(data, reindex_dic, on='OldItemId')
    data = data[['SessionId', 'ItemId', 'Time']].sort_values(['SessionId', 'Time'])
    data.ItemId += 1
    size = data.groupby("SessionId").size()
    max_len = max(size)
    prepare = []
    for s in data.groupby("SessionId"):
        sid = s[1].SessionId.values[0]
        if len(s[1]) == 1:
            continue
        # l = [sid] + s[1].ItemId.tolist() + [0] * (max_len - len(s[1]))
        # prepare.append(l)
        l = s[1].ItemId.tolist()
        for k in range(1, len(l)) if argument else [len(l) - 1]:
            for i in l[:k]:
                prepare.append([sid, i])
            for i in range(max_len - 1 - k):
                prepare.append([sid, 0])
            prepare.append([sid, l[k]])
    data = pd.DataFrame(prepare)
    data.columns = ['SessionId', 'ItemId']
    data['Time'] = data.index
    return data


def load(filename, generate, *args):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        res = generate(*args)
        with open(filename, 'wb') as f:
            pickle.dump(res, f)
            return res


if __name__ == '__main__':
    train = pd.read_csv(train_data, sep=',', dtype={'ItemId': np.int64})
    valid = pd.read_csv(valid_data, sep=',', dtype={'ItemId': np.int64})
    test = pd.read_csv(test_data, sep=',', dtype={'ItemId': np.int64})
    reindex_dic = pd.read_csv('../benchmarks/kg_{}/item_index_old2new_{}.txt'.format(args.data, args.type)
                              , delimiter='\t', header=None, names=['OldItemId', 'ItemId']
                              , dtype={'ItemId': np.int64})  # item_id -> kg_id
    if args.type == 'B':
        if args.data == 'year15':
            reindex_dic['OldItemId'] -= 29207
        else:
            reindex_dic['OldItemId'] -= 36845

    if os.path.exists('ItemEmb' + args.data + args.type) and os.path.exists('KBEmb' + args.data + args.type):
        with open('ItemEmb' + args.data + args.type, 'rb') as f0, open('KBEmb' + args.data + args.type, 'rb') as f1:
            ItemEmbedding = pickle.load(f0)
            KBItemEmbedding = pickle.load(f1)
            length = len(list(ItemEmbedding.values())[-1])
            KBlength = len(list(KBItemEmbedding.values())[-1])
    else:
        ItemEmbedding, length = read_ItemEmbedding(File_ItemEmbedding)
        KBItemEmbedding, KBlength = read_ItemEmbedding(File_KBItemEmbedding, isItem=False)
        with open('ItemEmb' + args.data + args.type, 'wb') as f0, open('KBEmb' + args.data + args.type, 'wb') as f1:
            pickle.dump(ItemEmbedding, f0)
            pickle.dump(KBItemEmbedding, f1)
    # for key in ItemEmbedding:
    #     ItemEmbedding[key] = np.random.rand(256)
    print("the length and number of data ItemEmbedding", length, len(ItemEmbedding))
    print("the length and number of data ItemEmbedding", KBlength, len(KBItemEmbedding))
    r_matrix = read_r_matrix(File_r_matrix)
    print("the shape of r_matrix", r_matrix.shape)

    valid_sum = len(test["SessionId"].unique())
    test_sum = len(test["SessionId"].unique())
    items = train["ItemId"].unique()
    valid = valid[valid["ItemId"].isin(items)]
    test = test[test["ItemId"].isin(items)]

    argument = args.argument

    valid = valid[valid.SessionId.isin((valid.groupby('SessionId').size() > 1).index)]
    test = test[test.SessionId.isin((test.groupby('SessionId').size() > 1).index)]
    train = process(train, argument)
    valid = process(valid, argument)
    test = process(test, argument)
    # train = load('train', process, train, argument)
    # test = load('test', process, test, argument)

    # Reproducing results from "Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)

    gru = MN4rec.KVMN(layers=[256], batch_size=args.batch_size, embedding=length,
                      KBembedding=KBlength, dropout_p_hidden=args.dropout, n_sample=args.num_neg,
                      learning_rate=args.lr, momentum=0.1, sample_alpha=0, time_sort=True,
                      train_random_order=True, out_dim=args.out_dim, MN_nfactors=r_matrix.shape[0],
                      MN_dims=r_matrix.shape[1], )

    gru.fit(train, ItemEmbedding, KBItemEmbedding, r_matrix, n_epochs=args.epochs,
            retrain=args.retrain, only_eval=args.eval, valid=valid, test=test,
            valid_sum=valid_sum, test_sum=test_sum, savepath=args.savepath, reload=args.reload)
