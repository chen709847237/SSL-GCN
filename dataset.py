#coding=utf-8
import os
import torch
import pickle
import numpy as np
from rdkit import Chem
import dgl.backend as F
from rdkit.Chem import AllChem
from dgl import save_graphs, load_graphs
from joblib import Parallel, delayed, cpu_count


def pmap(pickleable_fn, data, n_jobs=None, verbose=1, **kwargs):
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    return Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(pickleable_fn)(d, **kwargs) for d in data
    )

class Dataset(object):
    def __init__(self, df, smiles_to_graph, node_featurizer, edge_featurizer, smiles_column,
                 cache_file_path, task_names=None, load=True, log_every=300, init_mask=True, # load=True
                 n_jobs=1):
        self.df = df
        self.smiles = self.df[smiles_column].tolist()
        if task_names is None:
            self.task_names = self.df.columns.drop([smiles_column]).tolist()
        else:
            self.task_names = task_names
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer,
                          load, log_every, init_mask, n_jobs)
        self._task_pos_weights = None

    def _pre_process(self, smiles_to_graph, node_featurizer,
                     edge_featurizer, load, log_every, init_mask, n_jobs=1):
        if os.path.exists(self.cache_file_path) and load:
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
            self.labels = label_dict['labels']
            if init_mask:
                self.mask = label_dict['mask']
            self.valid_ids = label_dict['valid_ids'].tolist()
        else:
            print('Processing dgl graphs from scratch...')
            if n_jobs > 1:
                self.graphs = pmap(smiles_to_graph,
                                   self.smiles,
                                   node_featurizer=node_featurizer,
                                   edge_featurizer=edge_featurizer,
                                   n_jobs=n_jobs)
            else:
                self.graphs = []
                for i, s in enumerate(self.smiles):
                    if (i + 1) % log_every == 0:
                        print('Processing molecule {:d}/{:d}'.format(i+1, len(self)))
                    self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer))

            self.valid_ids = []
            graphs = []
            for i, g in enumerate(self.graphs):
                if g is not None:
                    self.valid_ids.append(i)
                    graphs.append(g)
            self.graphs = graphs
            _label_values = self.df[self.task_names].values

            self.labels = F.zerocopy_from_numpy(np.nan_to_num(_label_values, nan=-1).astype(np.float32))[self.valid_ids]
            valid_ids = torch.tensor(self.valid_ids)
            if init_mask:
                self.mask = F.zerocopy_from_numpy((~np.isnan(_label_values)).astype(np.float32))[self.valid_ids]
                save_graphs(self.cache_file_path, self.graphs, labels={'labels': self.labels, 'mask': self.mask, 'valid_ids': valid_ids})
            else:
                self.mask = None
                save_graphs(self.cache_file_path, self.graphs, labels={'labels': self.labels, 'valid_ids': valid_ids})

        self.smiles = [self.smiles[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.mask is not None:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]

    def __len__(self):
        return len(self.smiles)

def save_ecfp(save_path, data):
    with open(save_path, 'wb') as f:
        try:
            pickle.dump(data, f)
            f.close()
        except:
            print('ERROR save ecfp !')

def load_ecfp(cache_path):
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
        f.close()
    features = data.pop('features')
    return features, data

class DatasetECFP4(object):
    def __init__(self, df, smiles_column, cache_file_path, task_names=None,
                 load=False, log_every=300, init_mask=True):
        self.df = df
        self.smiles = self.df[smiles_column].tolist()

        if task_names is None:
            self.task_names = self.df.columns.drop([smiles_column]).tolist()
        else:
            self.task_names = task_names
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        self._pre_process(load, log_every, init_mask)
        self._task_pos_weights = None

    def _pre_process(self, load, log_every, init_mask):
        if os.path.exists(self.cache_file_path) and load:

            print('Loading previously saved ECFP4 fingerprint...')
            self.descriptors, label_dict = load_ecfp(self.cache_file_path)
            self.labels = label_dict['labels']
            if init_mask:
                self.mask = label_dict['mask']
            self.valid_ids = label_dict['valid_ids'].tolist()
        else:
            print('Processing ECFP4  from scratch...')

            self.descriptor = []
            for i, s in enumerate(self.smiles):
                if (i + 1) % log_every == 0:
                    print('Processing molecule {:d}/{:d}'.format(i+1, len(self)))
                self.descriptor.append([i for i in list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), nBits=2048, radius=2, useFeatures=False))])

            self.valid_ids = []
            descriptors = []
            for i, ecfp in enumerate(self.descriptor):
                if ecfp is not None:
                    self.valid_ids.append(i)
                    descriptors.append(ecfp)
            self.descriptors = descriptors

            _label_values = self.df[self.task_names].values

            self.labels = F.zerocopy_from_numpy(np.nan_to_num(_label_values, nan=-1).astype(np.float32))[self.valid_ids]
            valid_ids = torch.tensor(self.valid_ids)
            if init_mask:
                self.mask = F.zerocopy_from_numpy((~np.isnan(_label_values)).astype(np.float32))[self.valid_ids]
                save_ecfp(self.cache_file_path, data={'features':self.descriptors, 'labels': self.labels, 'mask': self.mask, 'valid_ids': valid_ids})
            else:
                self.mask = None
                save_ecfp(self.cache_file_path, data={'features':self.descriptors, 'labels': self.labels, 'valid_ids': valid_ids})
        self.smiles = [self.smiles[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.mask is not None:
            return self.smiles[item], self.descriptors[item], self.labels[item], self.mask[item]
        else:
            return self.smiles[item], self.descriptors[item], self.labels[item]

    def __len__(self):
        return len(self.smiles)

    def task_pos_weights(self, indices):
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = F.sum(self.labels[indices], dim=0)
        num_indices = F.sum(self.mask[indices], dim=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]

        return task_pos_weights
