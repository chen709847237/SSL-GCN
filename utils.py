#coding=utf-8
import os
import dgl
import json
import torch
import errno
import numpy as np
from functools import partial
import torch.nn.functional as F
from dgllife.utils import smiles_to_bigraph, ScaffoldSplitter, RandomSplitter, SingleTaskStratifiedSplitter, mol_to_bigraph

from dataset import  Dataset, DatasetECFP4


def init_featurizer(args):
    if args['model'] in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                         'gin_supervised_edgepred', 'gin_supervised_masking']:
        from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
        args['atom_featurizer_type'] = 'pre_train'
        args['bond_featurizer_type'] = 'pre_train'
        args['node_featurizer'] = PretrainAtomFeaturizer()
        args['edge_featurizer'] = PretrainBondFeaturizer()
        return args

    if args['atom_featurizer_type'] == 'canonical':
        # Atom Featurizer
        from dgllife.utils import CanonicalAtomFeaturizer
        args['node_featurizer'] = CanonicalAtomFeaturizer()
    elif args['atom_featurizer_type'] == 'attentivefp':
        from dgllife.utils import AttentiveFPAtomFeaturizer
        args['node_featurizer'] = AttentiveFPAtomFeaturizer()
    else:
        return ValueError(
            "Expect node_featurizer to be in ['canonical', 'attentivefp'], "
            "got {}".format(args['atom_featurizer_type']))

    if args['model'] in ['Weave', 'MPNN', 'AttentiveFP']:
        if args['bond_featurizer_type'] == 'canonical':
            from dgllife.utils import CanonicalBondFeaturizer
            args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
        elif args['bond_featurizer_type'] == 'attentivefp':
            from dgllife.utils import AttentiveFPBondFeaturizer
            args['edge_featurizer'] = AttentiveFPBondFeaturizer(self_loop=True)
    else:
        args['edge_featurizer'] = None

    return args

def load_dataset(args, df):
    if args['cache_path'] == None:
        dataset = Dataset(df=df,
                                     smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                     node_featurizer=args['node_featurizer'],
                                     edge_featurizer=args['edge_featurizer'],
                                     smiles_column=args['smiles_column'],
                                     cache_file_path=args['result_path'] +'/graph.bin',
                                     task_names=args['task_names'],
                                     n_jobs=args['num_workers'])
    else:
        dataset = Dataset(df=df,
                                     smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                     node_featurizer=args['node_featurizer'],
                                     edge_featurizer=args['edge_featurizer'],
                                     smiles_column=args['smiles_column'],
                                     cache_file_path=args['cache_path'] +'/graph.bin',
                                     task_names=args['task_names'],
                                     n_jobs=args['num_workers'])
    return dataset

def get_self_configure(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        f.close()
    return config

def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def collate_molgraphs(data):
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def load_model(exp_configure):
    if exp_configure['model'] == 'GCN':
        from dgllife.model import GCNPredictor
        model = GCNPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks'])
    else:
        return ValueError("Expect model to be from ['GCN'], "
                          "got {}".format(exp_configure['model']))
    return model

def predict(args, model, bg):
    bg = bg.to(args['device'])
    if args['edge_featurizer'] is None:
        node_feats = bg.ndata.pop('h').to(args['device'])
        return model(bg, node_feats)
    elif args['bond_featurizer_type'] == 'pre_train':
        node_feats = [
            bg.ndata.pop('atomic_number').to(args['device']),
            bg.ndata.pop('chirality_type').to(args['device'])
        ]
        edge_feats = [
            bg.edata.pop('bond_type').to(args['device']),
            bg.edata.pop('bond_direction_type').to(args['device'])
        ]
        return model(bg, node_feats, edge_feats)
    else:
        node_feats = bg.ndata.pop('h').to(args['device'])
        edge_feats = bg.edata.pop('e').to(args['device'])
        return model(bg, node_feats, edge_feats)

def data_unpack(dataset):
    smiles_array = np.array([data[0] for data in dataset])
    descriptors_array = np.array([data[1] for data in dataset])
    labels_array = np.array([data[2] for data in dataset])
    mask_array = np.array([data[3] for data in dataset])
    return smiles_array, descriptors_array, labels_array, mask_array

def load_ecfp_dataset(args, df):
    dataset = DatasetECFP4(df=df,
                                 smiles_column=args['smiles_column'],
                                 cache_file_path=args['cache_path'] +'/ecfp_data.pkl',
                                 task_names=args['task_names'],
                                 load=True)
    return dataset