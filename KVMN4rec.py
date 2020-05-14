import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import pickle
import os
from evaluation import *


class KVMN(nn.Module):
    def __init__(self, layers,
                 batch_size=50, dropout_p_hidden=0.5, dropout_p_embed=0.0,
                 learning_rate=0.05, momentum=0.0, lmbd=0.0, embedding=0, KBembedding=0,
                 n_sample=0, sample_alpha=0.75, smoothing=0, decay=0.9, grad_cap=0, sigma=0,
                 init_as_normal=False, reset_after_session=True, train_random_order=False,
                 time_sort=False, session_key='SessionId', item_key='ItemId', time_key='Time',
                 out_dim=64, MN_nfactors=10, MN_dims=64):

        super(KVMN, self).__init__()
        self.layers = layers
        self.batch_size = batch_size
        self.dropout_p_hidden = nn.Dropout(dropout_p_hidden)
        self.dropout_p_embed = nn.Dropout(dropout_p_embed)
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.sigma = sigma
        self.init_as_normal = init_as_normal
        self.reset_after_session = reset_after_session
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.grad_cap = grad_cap
        self.train_random_order = train_random_order
        self.lmbd = lmbd
        self.embedding = embedding
        self.time_sort = time_sort
        # self.final_act = final_act
        self.loss = nn.NLLLoss
        self.hidden_activation = torch.tanh
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.smoothing = smoothing
        ## add memory network info
        self.KBembedding = KBembedding
        self.MN_nfactors = MN_nfactors
        self.MN_dims = KBembedding  # MN_dims
        self.out_dim = out_dim
        ## parameters for read operator
        self.MN_u = nn.Linear(self.layers[-1], self.MN_dims)
        self.MN_u2 = nn.Linear((self.MN_dims + self.layers[-1]), self.layers[-1])
        ## parameters for write operator
        self.MN_ea = nn.Linear(self.KBembedding, self.MN_dims)

        self.gru = nn.GRUCell(self.embedding, self.layers[0])
        self.gru2 = nn.GRUCell(self.out_dim, self.out_dim)

        self.mlp1 = nn.Linear(self.layers[-1], self.out_dim)

    def ini(self, data, ItemEmbedding, KBItemEmbedding, r_matrix, retrain=False):

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(666)
        else:
            torch.manual_seed(666)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        data.sort_values([self.session_key, self.time_key], inplace=True)
        # offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        # offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        # if os.path.exists('ItemE') and os.path.exists('KBE'):
        #     with open('ItemE', 'rb') as f0, open('KBE', 'rb') as f1:
        #         ItemE = pickle.load(f0)
        #         ItemKBE = pickle.load(f1)
        # else:
        ItemE = self.generate_embedding(ItemEmbedding, True)
        ItemKBE = self.generate_embedding(KBItemEmbedding, retrain)
        # with open('ItemE', 'wb') as f0, open('KBE', 'wb') as f1:
        #     pickle.dump(ItemE, f0)
        #     pickle.dump(ItemKBE, f1)

        self.E = nn.Parameter(torch.tensor(ItemE))  # shape : self.init_weights((self.n_items, self.embedding))

        self.KBE = nn.Parameter(torch.tensor(ItemKBE))
        # self.MergeE = nn.Parameter(torch.tensor(np.hstack([ItemE, ItemKBE])))
        ### add memory network
        self.r_matrix = torch.tensor(r_matrix, device=self.device)
        self.mlp2 = nn.Linear(ItemKBE.shape[1] + ItemE.shape[1], self.out_dim)
        # self.mlp2 = nn.Linear(self.out_dim, self.n_items)

        self.constant_ones = torch.ones((self.MN_dims,), device=self.device)

        self.to(self.device)

        # return offset_sessions

    def generate_embedding(self, embedding, rand=False):  # embedding : dict {name:vector}
        embedding_new = {}
        for i in embedding:
            if i + 1 not in self.itemidmap:
                continue
            vector = embedding[i]
            embedding_new[self.itemidmap[i + 1]] = np.random.randn(len(vector)) if rand else vector
            # embedding_new[self.itemidmap[i + 1]] = np.random.randn(len(vector))

        embedding_new[self.itemidmap[0]] = np.zeros(len(vector))
        embedding = []
        for i in range(len(embedding_new)):
            embedding.append(embedding_new[i])
        return np.array(embedding, dtype=np.float32)

    def model_step(self, y, X, hp, MN):
        h = self.gru(y, hp)
        h = self.dropout_p_hidden(h)
        y = h
        ### add memory network part
        ## write operator
        mask = torch.ones_like(self.KBE, device=self.device)
        mask[0] = 0
        KBItem = (self.KBE * mask)[X]  # shape:b*KBembedding
        EA = self.hidden_activation(self.MN_ea(KBItem))  # shape:b*d
        EA = EA.unsqueeze(1).repeat(1, self.MN_nfactors, 1) + self.r_matrix  # shape:b*k*d
        MN_gate = torch.sigmoid(torch.matmul(MN * EA, self.constant_ones.unsqueeze(1)))
        # MN_gate = MN_gate.unsqueeze(2)  # shape:dot(b*k*d, d*1) = b*k*1
        MN = MN * (1 - MN_gate) + EA * MN_gate  # shape:b*k*d
        ## read operator
        U_trans = self.hidden_activation(self.MN_u(y))  # We need to make the same dimension. (shape:b*d)
        MN_AW = F.softmax(torch.matmul(U_trans, self.r_matrix.transpose(1, 0)), dim=1)  # shape:b*k
        MN_AC = torch.matmul(MN.transpose(2, 1), MN_AW.unsqueeze(2))  # shape: b*d*1
        Ut = torch.cat((y, MN_AC[:, :, 0]), -1)  # shape:b*(dim+d)
        Ut = self.hidden_activation(self.MN_u2(Ut))  # shape : b*layers[-1]
        # H_new[-1] = Ut #self.dropout(Ut, drop_p_hidden)
        # add Dense layer
        y = self.hidden_activation(self.mlp1(Ut)) # b*out_dim
        return y, h, MN

    def forward(self, X, Y, predict=False):
        MN = torch.zeros((X.shape[0], self.MN_nfactors, self.MN_dims), device=self.device)
        # self.E[0] = 0
        mask = torch.ones_like(self.E, device=self.device)
        mask[0] = 0
        Sx = (self.E * mask)[X]  # b*l*d
        h = None
        h2 = None
        y_sum = 0
        for t in range(0, X.shape[1]):
            y, h, MN = self.model_step(Sx[:, t, :], X[:, t], h, MN)
            # h2 = self.gru2(y, h2)
            y_sum += y
        y = y_sum / X.shape[1]
        # y = h2
        if Y is not None:
            # self.MergeE[0] = 0
            # SBy = self.By[Y]
            MergeE = torch.cat((self.E, self.KBE), 1)
            mask = torch.ones_like(MergeE, device=self.device)
            mask[0] = 0
            Sy = (MergeE * mask)
            Sy = self.hidden_activation(self.mlp2(Sy))  # b * n * out_dim
            if predict:
                # y = F.softmax(self.mlp2(y), dim=1)

                y = F.softmax(torch.matmul(y, Sy.transpose(-1, -2)), dim=1)
            else:
                # y = F.log_softmax(self.mlp2(y), dim=1)

                y = F.log_softmax(torch.matmul(y, Sy.transpose(-1, -2)), dim=1)
            return y
        else:  ## output user embedding
            if predict == True:
                return [h], MN, y, Sx
            else:
                return "error"

    def fit(self, data, ItemEmbedding, KBItemEmbedding, r_matrix, n_epochs, valid, test, valid_sum, test_sum,
            retrain=False, only_eval=False, sample_store=10000000, item_size=None, savepath=None, reload=False):
        '''
        Trains the network.

        Parameters
        --------
        data : pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        retrain : boolean
            If False, do normal train. If True, do additional train (weigths from previous trainings are kept as the initial network) (default: False)
        sample_store : int
            If additional negative samples are used (n_sample > 0), the efficiency of GPU utilization can be sped up, by precomputing a large batch of negative samples (and recomputing when necessary).
            This parameter regulizes the size of this precomputed ID set. Its value is the maximum number of int values (IDs) to be stored. Precomputed IDs are stored in the RAM.
            For the most efficient computation, a balance must be found between storing few examples and constantly interrupting GPU computations for a short time vs. computing many examples and interrupting GPU computations for a long time (but rarely).

        '''
        itemids = sorted(data[self.item_key].unique())
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                        on=self.item_key, how='inner')
        data = data.sort_values([self.session_key, self.time_key])
        train = data.groupby("SessionId").ItemIdx.apply(list)
        train_items = np.array(list(train.values))
        self.pos_len = train_items.shape[1]
        self.ini(data, ItemEmbedding, KBItemEmbedding, r_matrix, retrain)
        # base_order = np.argsort(
        #     data.groupby(self.session_key)[self.time_key].min().values) if self.time_sort else np.arange(
        #     len(offset_sessions) - 1)
        if self.n_sample:
            pop = data.groupby('ItemId').size()
            pop = pop[self.itemidmap.index.values].values
            pop = 1.0 * pop.cumsum() / pop.sum()
            pop[-1] = 1
            # session_samples = self.generate_neg_samples_sessionBased(data, pop)
            # print("We sample n_sample items for every session/user who never see them before")
        resample = True

        if reload:
            # restore_param = {k: v for k, v in torch.load(savepath).items() if k not in ["gru2.weight_ih", "gru2.weight_hh", "gru2.bias_ih", "gru2.bias_hh"]}
            # self.load_state_dict(restore_param)
            self.load_state_dict(torch.load(savepath))

        if only_eval:
            print('valid', evaluate_sessions_gpu(self, valid, data, sum=valid_sum))
            print('test', evaluate_sessions_gpu(self, test, data, sum=test_sum))
            return

        if resample == True:
            if self.n_sample:
                session_samples = self.generate_neg_samples_sessionBased(data, pop)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        best_mrr, best_epoch = 0, 0
        for epoch in range(n_epochs):
            self.train()
            c = []
            session_idx_arr = np.random.permutation(len(data[self.session_key].unique()))
            i = 0
            while i < len(train):
                self.optimizer.zero_grad()
                sis = session_idx_arr[i: i + self.batch_size] if i + self.batch_size < len(train) else \
                    session_idx_arr[-self.batch_size:]
                in_idx = train_items[sis]
                out_idx = in_idx[:, -1]
                in_idx = in_idx[:, 0:-1]
                neg_samples = session_samples[sis]
                i += self.batch_size

                # For bpr, modify "forward" not only the below

                # if self.n_sample > 0:
                #     y = np.hstack([out_idx[:, np.newaxis], neg_samples])
                # else:
                #     y = out_idx

                Y_pred = self.forward(torch.tensor(in_idx, device=self.device, dtype=torch.int64),
                                      torch.tensor(range(len(itemids)), device=self.device, dtype=torch.int64))
                # cost = self.bpr(Y_pred.gather(1, torch.tensor(y).to(self.device)))
                cost = F.nll_loss(Y_pred, target=torch.tensor(out_idx, device=self.device))
                if np.isnan(cost.item()):
                    print(str(epoch) + ': NaN error!')
                    self.error_during_train = True
                    return
                c.append(cost.item())
                cost.backward()
                self.optimizer.step()
            avgc = np.mean(c)
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            print('Epoch{}\tloss: {:.6f}'.format(epoch, avgc))

            if epoch >= 20:
                res = evaluate_sessions_gpu(self, valid, data, sum=valid_sum)
                if savepath and best_mrr < res[1][-1]:
                    best_mrr = res[1][-1]
                    best_epoch = epoch
                    # torch.save(self.state_dict(), savepath)
                print('best_epoch', best_epoch)
                print('valid', res)
                print('test', evaluate_sessions_gpu(self, test, data, sum=test_sum))

    def predict_next_batch(self, session_ids, input_item_ids, predict_for_item_ids=None, batch=100):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        predict_for_item_ids : 1D array (optional)
            IDs of items for which the network should give prediction scores. Every ID must be in the training set. The default value is None, which means that the network gives prediction on its every output (i.e. for all items in the training set).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        self.eval()
        with torch.no_grad():
            # in_idxs = self.itemidmap[input_item_ids]
            in_idxs = torch.tensor(input_item_ids, device=self.device, dtype=torch.int64)
            if predict_for_item_ids is not None:
                iIdxs = torch.tensor(self.itemidmap[predict_for_item_ids], device=self.device, dtype=torch.int64)
                yhat = self(in_idxs, iIdxs, predict=True)
                # preds = np.asarray(yhat.cpu()).T
                # return pd.DataFrame(data=preds)
                return yhat

    def generate_neg_samples(self, pop, length):
        if self.sample_alpha:
            sample = np.searchsorted(pop, np.random.rand(self.n_sample * length))
        else:
            sample = np.random.choice(self.n_items, size=self.n_sample * length)
        if length > 1:
            sample = sample.reshape((length, self.n_sample))
        return sample

    def generate_neg_samples_sessionBased(self, data, pop):
        session_ItemIdxs = data.groupby(self.session_key)['ItemIdx'].unique()
        session_ids = data[self.session_key].unique()
        session_samples = []
        for i in range(len(session_ids)):
            session_id = session_ids[i]
            ItemIdxs = session_ItemIdxs[session_id]
            samples = []
            for k in range(self.n_sample):
                # t = np.random.choice(self.n_items)
                t = np.searchsorted(pop, np.random.rand())
                while t in ItemIdxs:
                    # t = np.random.choice(self.n_items)
                    t = np.searchsorted(pop, np.random.rand())
                samples.append(t)
            session_samples.append(samples)
        return np.array(session_samples)

    def bpr(self, yhat):
        ans = - torch.log(torch.sigmoid(yhat[:, 0].unsqueeze(1) - yhat))  # .reshape((3, -1))
        return ans.mean()
