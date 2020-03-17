# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch

def evaluate_sessions(pr, test_data, train_data, sum, cut_off=20, session_key='SessionId', item_key='ItemId',
                      time_key='Time', batch_size=100):
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


def evaluate_sessions_gpu(pr, test_data, train_data, sum, cut_off=20, session_key='SessionId', item_key='ItemId',
                      time_key='Time', batch_size=100):
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
        # preds[np.isnan(preds)] = 0
        # preds += 1e-8 * np.random.rand(len(preds))  # Breaking up ties
        # preds.apply(lambda x: x + 1e-8)

        preds = preds[:, 1:]
        ranks = (preds.t() > torch.diag(preds[:, iid-1])).sum(0) + 1
        for k in [0, 1, 2]:
            rank_ok = (ranks <= cut_off[k])
            # pred_res += list(rank_ok)
            recall[k] += rank_ok.sum().item()
            mrr[k] += (1.0 / ranks[rank_ok].float()).sum().item()

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

def evaluate_sessions_minibatch(pr, test_data, train_data, sum, cut_off=20, session_key='SessionId', item_key='ItemId',
                      time_key='Time', batch_size=100):

    test_data.sort_values([session_key, time_key], inplace=True)
    test_data['iid'] = test_data[item_key].shift(-1)
    test_data = test_data.reset_index()
    test_data = test_data[~test_data.index.isin(test_data.groupby(session_key).size().cumsum() - 1)]
    test_data = test_data.astype(int)

    items = train_data[item_key].unique()
    # items_to_predict = np.arange(item_size)
    evalutation_point_count = 0
    # prev_iid, prev_sid = -1, -1
    mrr, recall = [0, 0, 0], [0, 0, 0]
    cut_off = [5, 10, 20]
    i = 0
    while i < len(test_data):
        # sid = sessions[i: i+batch_size]
        # pid = item_ids[i: i+batch_size]
        # iid = item_ids2[i: i+batch_size]
        sid = test_data[session_key][i: i+batch_size].values
        pid = test_data[item_key][i: i+batch_size].values
        iid = test_data['iid'][i: i+batch_size].values
        preds = pr.predict_next_batch(sid, pid, items, batch=len(pid))
        preds[np.isnan(preds)] = 0
        # preds += 1e-8 * np.random.rand(len(preds))  # Breaking up ties
        # preds.apply(lambda x: x + 1e-8)
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
        evalutation_point_count += len(iid)
        i += batch_size

    print("ranks mean: ", np.mean(ranks))
    # print(np.array(preds.max()))
    return np.array(recall, dtype=float) / evalutation_point_count, np.array(mrr) / evalutation_point_count
