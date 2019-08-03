# Copyright (C) 2019 Karsten Roth and Biagio Brattoli
#
# This file is part of metric-learning-mining-interclass-characteristics.
#
# metric-learning-mining-interclass-characteristics is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# metric-learning-mining-interclass-characteristics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""=================================================================="""
#################### LIBRARIES #################
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json, csv, pandas as pd
import matplotlib.pyplot as plt


import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import auxiliaries as aux

from tqdm import tqdm





"""=================================================================================================================="""
def evaluate(dataset, LOG, **kwargs):
    """
    Given a dataset name, applies the correct evaluation function.

    Args:
        dataset: str, name of dataset.
        LOG:     aux.LOGGER instance, main logging class.
        **kwargs: Input Argument Dict, depends on dataset.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    if dataset in ['cars196', 'cub200', 'online_products']:
        ret = evaluate_one_dataset(LOG, **kwargs)
    elif dataset in ['in-shop']:
        ret = evaluate_query_and_gallery_dataset(LOG, **kwargs)
    elif dataset in ['vehicle_id']:
        ret = evaluate_multiple_datasets(LOG, **kwargs)
    else:
        raise Exception('No implementation for dataset {} available!')

    return ret



"""========================================================="""
def evaluate_one_dataset(LOG, dataloader, model, opt, evaltype='Class', save=True, give_return=False, epoch=0):
    """
    Compute evaluation metrics, update LOGGER and print results.

    Args:
        LOG:         aux.LOGGER-instance. Main Logging Functionality.
        dataloader:  PyTorch Dataloader, Testdata to be evaluated.
        model:       PyTorch Network, Network to evaluate.
        opt:         argparse.Namespace, contains all training-specific parameters.
        save:        str, name of embedding to evaluate.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
        epoch:       int, current epoch, required for logger.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    start = time.time()
    image_paths = np.array(dataloader.dataset.image_list)

    with torch.no_grad():
        #Compute Metrics
        F1, NMI, recall_at_ks, feature_matrix_all = aux.eval_metrics_one_dataset(model, dataloader, device=opt.device, k_vals=opt.k_vals, evaltype=evaltype)
        #Make printable summary string.
        result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(opt.k_vals, recall_at_ks))
        result_str = 'Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(epoch, NMI, F1, result_str)

        if LOG is not None :
            if save:
                if not len(LOG[evaltype].progress_saver['val']['Recall @ 1']) or recall_at_ks[0]>np.max(LOG[evaltype].progress_saver['val']['Recall @ 1']):
                    aux.set_checkpoint(model, opt, LOG[evaltype].progress_saver, LOG[evaltype].prop.save_path+'/checkpoint_{}.pth.tar'.format(evaltype))
                    aux.recover_closest_one_dataset(feature_matrix_all, image_paths, LOG[evaltype].prop.save_path+'/sample_recoveries.png')

            #Update logs.
            LOG[evaltype].log('val', LOG[evaltype].metrics_to_log['val'], [epoch, np.round(time.time()-start), NMI, F1]+recall_at_ks)

    prefix = '{}: '.format(evaltype.upper())
    print(prefix+' '+result_str)

    if give_return:
        return recall_at_ks, NMI, F1
    else:
        None



"""========================================================="""
def evaluate_query_and_gallery_dataset(LOG, query_dataloader, gallery_dataloader, model, opt, evaltype='Class', save=True, give_return=False, epoch=0):
    """
    Compute evaluation metrics, update LOGGER and print results, specifically for In-Shop Clothes.

    Args:
        LOG:         aux.LOGGER-instance. Main Logging Functionality.
        query_dataloader:    PyTorch Dataloader, Query-testdata to be evaluated.
        gallery_dataloader:  PyTorch Dataloader, Gallery-testdata to be evaluated.
        model:       PyTorch Network, Network to evaluate.
        opt:         argparse.Namespace, contains all training-specific parameters.
        save:        str, name of the embedding to use.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
        epoch:       int, current epoch, required for logger.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    start = time.time()
    query_image_paths   = np.array([x[0] for x in query_dataloader.dataset.image_list])
    gallery_image_paths = np.array([x[0] for x in gallery_dataloader.dataset.image_list])

    with torch.no_grad():
        #Compute Metrics.
        F1, NMI, recall_at_ks, query_feature_matrix_all, gallery_feature_matrix_all = aux.eval_metrics_query_and_gallery_dataset(model, query_dataloader, gallery_dataloader, device=opt.device, k_vals = opt.k_vals, evaltype=evaltype)
        #Generate printable summary string.
        result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(opt.k_vals, recall_at_ks))
        result_str = 'Epoch (Test) {0}: NMI [{1:.4f}] | F1 [{2:.4f}] | Recall [{3}]'.format(epoch, NMI, F1, result_str)

        if LOG is not None:
            if save:
                if not len(LOG[evaltype].progress_saver['val']['Recall @ 1']) or recall_at_ks[0]>np.max(LOG[evaltype].progress_saver['val']['Recall @ 1']):
                    aux.set_checkpoint(model, opt, LOG[evaltype].progress_saver, LOG[evaltype].prop.save_path+'/checkpoint_{}.pth.tar'.format(evaltype))
                    aux.recover_closest_inshop(query_feature_matrix_all, gallery_feature_matrix_all, query_image_paths, gallery_image_paths, LOG[evaltype].prop.save_path+'/sample_recoveries.png')

            #Update logs.
            LOG[evaltype].log('val', LOG[evaltype].metrics_to_log['val'], [epoch, np.round(time.time()-start), NMI, F1]+recall_at_ks)

    prefix = '{}: '.format(evaltype.upper())
    print(prefix+' '+result_str)

    if give_return:
        return recall_at_ks, NMI, F1
    else:
        None






"""========================================================="""
def evaluate_multiple_datasets(LOG, dataloaders, model, opt, evaltype='Class', save=True, give_return=False, epoch=0):
    """
    Compute evaluation metrics, update LOGGER and print results, specifically for Multi-test datasets s.a. PKU Vehicle ID.

    Args:
        LOG:         aux.LOGGER-instance. Main Logging Functionality.
        dataloaders: List of PyTorch Dataloaders, test-dataloaders to evaluate.
        model:       PyTorch Network, Network to evaluate.
        opt:         argparse.Namespace, contains all training-specific parameters.
        save:        str, name of embedding to evaluate.
        save:        bool, if True, Checkpoints are saved when testing metrics (specifically Recall @ 1) improve.
        give_return: bool, if True, return computed metrics.
        epoch:       int, current epoch, required for logger.
    Returns:
        (optional) Computed metrics. Are normally written directly to LOG and printed.
    """
    start = time.time()

    csv_data = [epoch]


    with torch.no_grad():
        for i,dataloader in enumerate(dataloaders):
            print('Working on Set {}/{}'.format(i+1, len(dataloaders)))
            image_paths = np.array(dataloader.dataset.image_list)
            #Compute Metrics for specific testset.
            F1, NMI, recall_at_ks, feature_matrix_all = aux.eval_metrics_one_dataset(model, dataloader, device=opt.device, k_vals = opt.k_vals, evaltype=evaltype)
            #Generate printable summary string.
            result_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(opt.k_vals, recall_at_ks))
            result_str = 'SET {0}: Epoch (Test) {1}: NMI [{2:.4f}] | F1 {3:.4f}| Recall [{4}]'.format(i+1, epoch, NMI, F1, result_str)

            if LOG is not None:
                if save:
                    if not len(LOG['Class'].progress_saver['val']['Set {} Recall @ 1'.format(i)]) or recall_at_ks[0]>np.max(LOG['Class'].progress_saver['val']['Set {} Recall @ 1'.format(i)]):
                        #Save Checkpoint for specific test set.
                        aux.set_checkpoint(model, opt, LOG['Class'].progress_saver, LOG['Class'].prop.save_path+'/checkpoint_set{}.pth.tar'.format(i+1))
                        aux.recover_closest_one_dataset(feature_matrix_all, image_paths, LOG['Class'].prop.save_path+'/sample_recoveries_set{}.png'.format(i+1))

                csv_data += [NMI, F1]+recall_at_ks
            print(result_str)

    csv_data.insert(0, np.round(time.time()-start))
    #Update logs.
    LOG[evaltype].log('val', LOG[evaltype].metrics_to_log['val'], csv_data)


    if give_return:
        return csv_data[2:]
    else:
        None
