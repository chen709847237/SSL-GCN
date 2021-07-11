#coding=utf-8
import os
import torch
import joblib
import numpy as np
import pandas as pd
import dgl.backend as F
from dgl.data.utils import Subset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from utils import init_featurizer,  load_dataset, get_self_configure, mkdir_p, collate_molgraphs, load_model, predict, data_unpack, load_ecfp_dataset


def evaluate(args, exp_config, test_set):
    exp_config.update({
        'model': args['model'],
        'n_tasks': args['n_tasks'],
        'atom_featurizer_type': args['atom_featurizer_type'],
        'bond_featurizer_type': args['bond_featurizer_type']})
    if args['atom_featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
    if args['edge_featurizer'] is not None and args['bond_featurizer_type'] != 'pre_train':
        exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()

    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'], collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    model = load_model(exp_config).to(args['device'])
    model.load_state_dict(torch.load(args['model_data_path']+'/model.pth', map_location=torch.device('cpu'))['model_state_dict'])

    val_prob_list = []
    val_target_list = []
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            logits = predict(args, model, bg)
            proba = torch.sigmoid(logits)
            val_prob_list.extend(proba.detach().cpu().data)
            val_target_list.extend(labels.detach().cpu().data)
    auc_score= roc_auc_score(val_target_list, val_prob_list)
    return auc_score

def evaluate_std(args, exp_config, test_set):
    # Record settings
    exp_config.update({
        'model': args['model'],
        'n_tasks': args['n_tasks'],
    })
    test_smiles, test_descriptors, test_labels, test_masks = data_unpack(test_set)
    model = joblib.load(args['model_data_path']+'/'+args['model']+'_model.skl')
    val_prob_list = model.predict_proba(test_descriptors)[:, 1]
    auc_score= roc_auc_score(test_labels, val_prob_list)
    return auc_score

def norm_model_evaluate(data_folder, root_data_folder, output_data_folder):

    data_files = os.listdir(root_data_folder)
    if '.DS_Store' in data_files:
        data_files.remove('.DS_Store')
    data_files = np.unique(np.array([i[0:-3] for i in data_files]))

    final_result_dict = {'task': [], 'auc': []}
    for model_data_folder_p1 in data_files:
        task = model_data_folder_p1.split('_')[0]
        task_result_dict = {'file': [], 'auc': []}
        for version in range(5):
            model_data_folder = model_data_folder_p1 + '_v' + str(version + 1)
            pretrain_folder_path = root_data_folder + model_data_folder
            local_train_folder_path = output_data_folder + model_data_folder

            args = {'csv_path': data_folder + 'csv\\' + task + '.csv',
                    'cache_path': data_folder + 'cache\\' +  task,
                    'task_names': task,
                    'smiles_column': 'SMILES',
                    'model': 'GCN',
                    'atom_featurizer_type': 'canonical',
                    'split': 'scaffold_smiles',
                    'split_ratio': '0.8,0.1,0.1',
                    'num_workers': 0,
                    'print_every': 3,
                    'metric': 'roc_auc_score',
                    'result_path': local_train_folder_path,
                    'model_data_path': pretrain_folder_path,
                    'atom_featurizer_type': 'canonical',
                    'bond_featurizer_type': 'canonical'
                    }
            args = init_featurizer(args)

            if torch.cuda.is_available():
                args['device'] = torch.device(args['cuda_define'])
            else:
                args['device'] = torch.device('cpu')


            if args['task_names'] is not None:
                args['task_names'] = args['task_names'].split(',')

            df = pd.read_csv(args['csv_path'])
            mkdir_p(args['result_path'])

            df_label = df[args['task_names']].values
            labels = F.zerocopy_from_numpy(np.nan_to_num(df_label, nan=-1).astype(np.float32))
            label_idx = np.argwhere(np.array(labels).squeeze(1) != -1).squeeze(1)

            label_df = df.iloc[label_idx]
            lab_dataset = load_dataset(args, label_df)

            args['n_tasks'] = lab_dataset.n_tasks
            data_indices_dict = np.load(args['model_data_path'] + '/data_indices_dict.npy', allow_pickle=True).item()
            test_set = Subset(lab_dataset, data_indices_dict['test_indices'])

            print('Use pre-selected hyperparameters\nPATH: {}'.format(args['model_data_path']))
            exp_config = get_self_configure(args['model_data_path'] + '/configure.json')
            auc_score = evaluate(args, exp_config, test_set)

            task_result_dict['file'].append(model_data_folder)
            task_result_dict['auc'].append(auc_score)
            os.rmdir(local_train_folder_path)

            print('----------------------------------------------------------------\n')

        task_result_df = pd.DataFrame(task_result_dict)
        task_result_df.to_csv(output_data_folder + task + '_detail_result.csv', index=False)
        final_result_dict['task'].append(task)
        final_result_dict['auc'].append(np.array(task_result_dict['auc']).mean())

    final_result_df = pd.DataFrame(final_result_dict)
    final_result_df.to_csv(output_data_folder + '_SL_local_avg_result.csv', index=False)

def semi_model_evaluate(data_folder, root_data_folder, output_data_folder):
    data_files = os.listdir(root_data_folder)
    if '.DS_Store' in data_files:
        data_files.remove('.DS_Store')
    data_files = np.unique(np.array([i[0:-3] for i in data_files]))
    final_result_dict = {'task': [], 'ra': [], 'auc': []}

    for model_data_folder_p1 in data_files:
        task = model_data_folder_p1.split('_')[0]
        ra = model_data_folder_p1.split('_ssl_')[1]
        task_result_dict = {'file': [], 'auc': []}
        for version in range(5):
            model_data_folder = model_data_folder_p1 + '_v' + str(version + 1)

            pretrain_folder_path = root_data_folder + model_data_folder
            local_train_folder_path = output_data_folder + model_data_folder

            args = {'csv_path': data_folder + 'csv\\' + task + '.csv',
                    'cache_path': data_folder + 'cache\\' + task,
                    'task_names': task,
                    'smiles_column': 'SMILES',
                    'model': 'GCN',
                    'atom_featurizer_type': 'canonical',
                    'split': 'scaffold_smiles',
                    'split_ratio': '0.8,0.1,0.1',
                    'num_workers': 0,
                    'print_every': 3,
                    'metric': 'roc_auc_score',

                    'result_path': local_train_folder_path,
                    'model_data_path': pretrain_folder_path,

                    'atom_featurizer_type': 'canonical',
                    'bond_featurizer_type': 'canonical'
                    }
            args = init_featurizer(args)

            if torch.cuda.is_available():
                args['device'] = torch.device(args['cuda_define'])
            else:
                args['device'] = torch.device('cpu')

            if args['task_names'] is not None:
                args['task_names'] = args['task_names'].split(',')

            df = pd.read_csv(args['csv_path'])
            mkdir_p(args['result_path'])

            df_label = df[args['task_names']].values
            labels = F.zerocopy_from_numpy(np.nan_to_num(df_label, nan=-1).astype(np.float32))
            label_idx = np.argwhere(np.array(labels).squeeze(1) != -1).squeeze(1)

            label_df = df.iloc[label_idx]
            lab_dataset = load_dataset(args, label_df)

            args['n_tasks'] = lab_dataset.n_tasks
            data_indices_dict = np.load(args['model_data_path'] + '/data_indices_dict.npy', allow_pickle=True).item()
            test_set = Subset(lab_dataset, data_indices_dict['test_indices'])

            print('Use pre-trained  specified hyperparameters\nPATH: {}'.format(args['model_data_path']))
            exp_config = get_self_configure(args['model_data_path'] + '/configure.json')
            auc_score = evaluate(args, exp_config, test_set)

            task_result_dict['file'].append(model_data_folder)
            task_result_dict['auc'].append(auc_score)
            os.rmdir(local_train_folder_path)

            print('----------------------------------------------------------------\n')

        task_result_df = pd.DataFrame(task_result_dict)
        task_result_df.to_csv(output_data_folder + task + '_task_' + ra + '_detail_result.csv', index=False)

        final_result_dict['task'].append(task)
        final_result_dict['ra'].append(ra)
        final_result_dict['auc'].append(np.array(task_result_dict['auc']).mean())

    final_result_df = pd.DataFrame(final_result_dict)
    final_result_df.to_csv(output_data_folder + '_SSL_local_avg_result.csv', index=False)

def std_model_evaluate(data_folder, root_data_folder, output_data_folder):
    data_files = os.listdir(root_data_folder)
    if '.DS_Store' in data_files:
        data_files.remove('.DS_Store')

    data_files = np.unique(np.array([i[0:-3] for i in data_files]))
    final_result_dict = {'task': [], 'model': [], 'auc': []}

    for model_data_folder_p1 in data_files:

        task = model_data_folder_p1.split('_')[0]
        model_type = model_data_folder_p1.split('_')[-1]
        task_result_dict = {'file': [], 'auc': []}

        for version in range(5):
            model_data_folder = model_data_folder_p1 + '_v' + str(version + 1)

            pretrain_folder_path = root_data_folder + model_data_folder
            local_train_folder_path = output_data_folder + model_data_folder

            args = { 'csv_path': data_folder + 'csv\\' + task + '.csv',
                'cache_path': data_folder + 'cache\\' + task,
                'task_names': task,
                'smiles_column': 'SMILES',
                'model': model_type,
                'split': 'scaffold_smiles',
                'split_ratio': '0.8,0.1,0.1',
                'num_workers': 0,
                'metric': 'roc_auc_score',

                'result_path': local_train_folder_path,
                'model_data_path': pretrain_folder_path,

                'num_evals': None,
                'model_save': False}

            if args['task_names'] is not None:
                args['task_names'] = args['task_names'].split(',')

            df = pd.read_csv(args['csv_path'])
            mkdir_p(args['result_path'])

            df_label = df[args['task_names']].values
            labels = F.zerocopy_from_numpy(np.nan_to_num(df_label, nan=-1).astype(np.float32))
            label_idx = np.argwhere(np.array(labels).squeeze(1) != -1).squeeze(1)

            label_df = df.iloc[label_idx]
            lab_dataset = load_ecfp_dataset(args, label_df)

            args['n_tasks'] = lab_dataset.n_tasks
            data_indices_dict = np.load(args['model_data_path'] + '/data_indices_dict.npy', allow_pickle=True).item()
            test_set = Subset(lab_dataset, data_indices_dict['test_indices'])

            print('Use pre-trained  specified hyperparameters\nPATH: {}'.format(args['model_data_path']))
            exp_config = get_self_configure(args['model_data_path'] + '/configure.json')
            exp_config['model_type'] = model_type

            auc_score = evaluate_std(args, exp_config, test_set)

            task_result_dict['file'].append(model_data_folder)
            task_result_dict['auc'].append(auc_score)
            os.rmdir(local_train_folder_path)

            print('----------------------------------------------------------------\n')

        task_result_df = pd.DataFrame(task_result_dict)
        task_result_df.to_csv(output_data_folder + task + '_' + model_type + '_detail_result.csv', index=False)

        final_result_dict['task'].append(task)
        final_result_dict['model'].append(model_type)
        final_result_dict['auc'].append(np.array(task_result_dict['auc']).mean())


    final_result_df = pd.DataFrame(final_result_dict)
    final_result_df.to_csv(output_data_folder + '_CM_local_avg_result.csv', index=False)


if __name__ == '__main__':

    parser = ArgumentParser('Local reproduction experiments of CM model, SL-GCN model and SSL-GCN model')
    parser.add_argument('-d', '--data-path', type=str, required=True, help='The path to the data folder (with "/" or "\\" at the end)')
    parser.add_argument('-m', '--model-path', type=str, required=True, help='The path to the model folder (with "/" or "\\" at the end)')
    parser.add_argument('-mt', '--model-type', choices=['cm', 'sl', 'ssl'], default='cm',
                                    help='define the type of model (default: CM models).'
                                            '"cm" --- Conventional Machine Learning Models, include KNN, NN, RF, SVM, and XGBoost.'
                                            '"sl" --- SL-GCN models.'
                                            '"ssl" --- SSL-GCN models.')
    parser.add_argument('-o', '--output-path', default=None, type=str, help='The path to an empty output folder where the experiment results will be stored (with "/" or "\\" at the end)')
    start_args = parser.parse_args().__dict__

    # REMOVE BEFORE UPLOAD
    '''start_args = {
        'model_type': 'cm',
        'data_path': 'D:\mol_net_2021\git_reproduce\data\\',
        'model_path': 'D:\mol_net_2021\git_reproduce\model\\',
        'output_path': 'D:\mol_net_2021\git_reproduce\\std_result\\'
    }'''

    if start_args['model_type'] == 'cm':
        std_model_evaluate(start_args['data_path'], start_args['model_path']+'cm\\', start_args['output_path'])
    elif start_args['model_type'] == 'sl':
        norm_model_evaluate(start_args['data_path'], start_args['model_path']+'sl\\', start_args['output_path'])
    elif start_args['model_type'] == 'ssl':
        semi_model_evaluate(start_args['data_path'], start_args['model_path']+'ssl\\', start_args['output_path'])









