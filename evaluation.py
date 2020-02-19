# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# title_df = pd.read_csv('../benchmarks/kg_year15/kitchen_entity_title.txt', delimiter='\t', header=None, names=['asin', 'title'])
# old2new_df = pd.read_csv('../benchmarks/kg_year15/item_index_old2new_B.txt', delimiter='\t', header=None, names=['oldId', 'ItemId'])
# reindex_df = pd.read_csv('../benchmarks/kg_year15/entity2item_B.txt', delimiter='\t', header=None, names=['asin', 'eid', 'oldId'])
# title_df = title_df.merge(reindex_df.merge(old2new_df))
# title_df['ItemId'] += 1

def evaluate_sessions(pr, test_data, train_data, sum, cut_off=20, session_key='SessionId', item_key='ItemId',
                      time_key='Time', batch_size=100):
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)

    '''
    items = train_data[item_key].unique()
    test_data = pd.merge(test_data, pd.DataFrame({pr.item_key: items, 'ItemIdx': pr.itemidmap[items].values}),
                         on=pr.item_key, how='inner')
    test_data.sort_values([session_key, time_key], inplace=True)
    # test_data = test_data[[session_key, 'ItemIdx']]
    # test_data.columns = [session_key, item_key]
    test = test_data.groupby("SessionId").ItemIdx.apply(list)
    test_items = np.array(list(test.values))
    # items_to_predict = np.arange(item_size)
    evalutation_point_count = 0
    # prev_iid, prev_sid = -1, -1
    mrr, recall = [0, 0, 0], [0, 0, 0]
    cut_off = [5, 10, 20]
    # dic = pd.merge(title_df, pd.DataFrame({pr.item_key: items, 'ItemIdx': pr.itemidmap[items].values}),
    #                      on=pr.item_key, how='inner')
    # trans = dic.set_index('ItemIdx')['title']
    i = 0
    while i < len(test_items):
        sid = test.index[i: i + batch_size].values
        pid = test_items[i: i + batch_size][:, 0:-1]
        iid = test_items[i: i + batch_size][:, -1]
        # sid = test_data[i: i+batch_size].SessionId
        # pid = test_data[i: i+batch_size].ItemId
        # iid = test_data[i+1: i+batch_size+1].ItemId
        preds = pr.predict_next_batch(sid, pid, items, batch=len(pid))
        preds[np.isnan(preds)] = 0
        # preds += 1e-8 * np.random.rand(len(preds))  # Breaking up ties
        # preds.apply(lambda x: x + 1e-8)

        preds = preds[preds.index != 0]
        ranks = (preds > np.diag(preds.T[iid])).sum(axis=0) + 1
        for k in [0, 1, 2]:
            rank_ok = (ranks <= cut_off[k])
            # pred_res += list(rank_ok)
            recall[k] += rank_ok.sum()
            mrr[k] += (1.0 / ranks[rank_ok]).sum()

        # for j in range(len(iid)):
        #     if iid[j] == 0:
        #         continue
        #     rank = ((preds[j] > preds[j][iid[j]]).sum() + 1).item()
        #     # assert rank > 0
        #     for k in [0, 1, 2]:
        #         if rank <= cut_off[k]:
        #             recall[k] += 1
        #             mrr[k] += 1.0 / rank
        # evalutation_point_count += len(iid)
        i += batch_size



    # print("ranks mean: ", np.mean(ranks))
    return np.array(recall, dtype=float) / sum, np.array(mrr, dtype=float) / sum
