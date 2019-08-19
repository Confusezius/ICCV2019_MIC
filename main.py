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

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn

import datasets        as data
import auxiliaries     as aux
import netlib          as netlib
import losses          as losses
import evaluate        as eval

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')





"""=================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

####### Main Parameter: Dataset to use for Training
parser.add_argument('--dataset',      default='cub200',   type=str,
                    help='Dataset to use. Select from [cub200, cars196, online_products, in-shop and vehicle_id].')


### Network parameters
parser.add_argument('--arch',           default='resnet50',  type=str,
                    help='Choice of architecture. Limited to resnet50.')
parser.add_argument('--not_pretrained', action ='store_true',
                    help='If set, no pretraining is used for initialization. Uncommon to use.')


### Evaluation Parameters
parser.add_argument('--k_vals',        nargs='+', default=[], type=int,
                    help='Recall @ Values. If set, default values for datasets are overwritten.')


### General Training Parameters
parser.add_argument('--n_epochs',   default=130,         type=int,
                    help='Number of training epochs.')
parser.add_argument('--kernels',    default=-1,           type=int,
                    help='Number of workers for pytorch dataloader.')
parser.add_argument('--seed',       default=1,           type=int,
                    help='Random seed for reproducibility.')
parser.add_argument('--scheduler',  default='step',      type=str,
                    help='Type of learning rate scheduling. Available: [step, exp]')
parser.add_argument('--gamma',      default=0.3,         type=float,
                    help='Learning rate reduction after tau epochs. Should be close to 1 for exponential scheduling.')
parser.add_argument('--decay',      default=0.0004,      type=float,
                    help='Weight decay for optimizer. Normally untouched for all runs.')
parser.add_argument('--tau',        default=[80], nargs='+',type=int,
                    help='Stepsize(s) before reducing learning rate.')
parser.add_argument('--task_p',     nargs='+', type=float, default=[1,0.8],
                    help='Prob. of [main task, aux. task] to be included in one iteration.')


### Parameters shared by label- and cluster-based tasks (main task/aux. task)
parser.add_argument('--lr',           default=1e-5, type=float,
                    help='Initial learning rate.')
parser.add_argument('--bs',           default=112,  type=int,
                    help='Mini-Batchsize to use. Set to 112 to fit on a 1080Ti (11GB).')
parser.add_argument('--cs_per_bs',    default=[4,4],        nargs='+', type=int,
                    help='Num. samples taken from one class before switching to next when filling batch. List of values for [main task, aux. task].')
parser.add_argument('--embed_sizes',  default=[128, 128],   nargs='+', type=int,
                    help='Output embedding sizes of the respective embeddings. List of values for [main task, aux. task].')
parser.add_argument('--losses',       default=['marginloss', 'marginloss'], nargs='+', type=str,
                    help='Criterion to use to train the resp. embeddings. List of values for [main task, aux. task].')
parser.add_argument('--sampling',     default=['distance',   'distance'],   nargs='+', type=str,
                    help='Sampling to use to train the resp. embeddings. List of values for [main task, aux. task].')

### Default Criterion parameters for provided loss functions (main task/aux. task).
### NOTE: The lists require two elements always, regardless of whether the loss function is used for both tasks.
parser.add_argument('--proxy_lr',          default=[1e-5, 1e-5], nargs='+', type=float,
                    help='PROXYNCA: Learning rates for proxies for [main task, aux. task].')
parser.add_argument('--beta',              default=[1.2, 1.2],   nargs='+', type=float,
                    help='MARGIN:   Initial beta-margin values for [main task, aux. task].')
parser.add_argument('--beta_lr',           default=[5e-4, 5e-4], nargs='+', type=float,
                    help='MARGIN:   Learning rate for beta-margin values for [main task, aux. task].')
parser.add_argument('--nu',                default=[0,0],        nargs='+', type=float,
                    help='MARGIN:   Regularisation value on betas in Margin Loss for [main task, aux. task].')
parser.add_argument('--margin',            default=[0.2, 0.2],   nargs='+', type=float,
                    help='TRIPLETS: Fixed Margin value for Triplet-based loss functions for [main task, aux. task].')

### Adversarial Loss function parameters (Projection Network R)
parser.add_argument('--adversarial',       default=['Class-Shared'], nargs='+', type=str,
                    help="Directions of adversarial loss ['target-source']: 'Class-Shared' (as used in the paper) and 'Shared-Class'. Can contain both directions.")
parser.add_argument('--adv_weights',       default=[2500], nargs='+', type=float,
                    help='Weighting parameter for adversarial loss. Needs to be the same length as the number of adv. loss directions.')
parser.add_argument('--adv_dim',           default=512,    type=int,
                    help='Dimension of linear layers in adversarial projection network.')

### Interclass Mining: Parameters
parser.add_argument('--shared_num_classes',      default=30,     type=int,
                    help='Number of clusters for auxiliary interclass mining task.')
parser.add_argument('--cluster_update_freq',     default=3,      type=int,
                    help='Number of epochs to train before updating cluster labels. E.g. 1 -> every other epoch.')
parser.add_argument('--cluster_mode',            default='mean', type=str,
                    help='Clustering mode: Without normalization (no_norm) or with mean-subtraction (mean) or mean-std-norm (mstd).')
parser.add_argument('--random_cluster_pick_p',   default=0.2, type=float,
                    help='Probability of assigning a random image to a cluster label to reduce overfitting to aux. task.')

### Setup Parameters
parser.add_argument('--gpu',               default=-1,           type=int,
                    help='GPU-ID for GPU to use.')
parser.add_argument('--savename',          default='',          type=str,
                    help='Specific save folder name. Will override default name based on start time.')
parser.add_argument('--make_graph',        action ='store_true',
                    help='If set, will include a computational graph of the underlying network.')

### Paths to datasets and storage folder
parser.add_argument('--source_path',  default=os.getcwd()+'/Datasets', type=str,
                    help='Path to folder containing the dataset folders.')
parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str,
                    help='Where to save everything.')

###
opt = parser.parse_args()



"""============================================================================"""
######## Adjust default parameters
# Set path to specific dataset folder
opt.source_path += '/'+opt.dataset
# Set path to specific dataset save-folder
opt.save_path   += '/'+opt.dataset
# Default Recall@k - values
if len(opt.k_vals)==0:
    if opt.dataset=='online_products':
        opt.k_vals = [1,10,100,1000]
    if opt.dataset=='in-shop':
        opt.k_vals = [1,10,20,30,50]
    if opt.dataset=='vehicle_id':
        opt.k_vals = [1,5]
    if opt.dataset=='cub200' or opt.dataset=='cars196':
        opt.k_vals = [1,2,4,8]
# Sanity Check to ensure that all input arguments are set correctly.
aux.sanity_check(opt)
# Names for Output Embedding Dictionary.
opt.tasks = ['Class','Shared']
# Adjusting and asserting loss-specific batch values (ProxyNCA requires drawing only one sample per class).
for i,loss in enumerate(opt.losses):
    if opt.losses[i]=='proxynca': opt.cs_per_bs[i]=1
    assert not opt.bs%opt.cs_per_bs[i], 'Batchsize has to be divisible by samples per class for {}.'.format(opt.losses[i])


if opt.kernels == -1:
    import multiprocessing
    opt.kernels = multiprocessing.cpu_count()

"""============================================================================"""
################### GPU SETTINGS ###########################
if opt.gpu > -1:
    os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)



"""============================================================================"""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)



"""============================================================================"""
##################### NETWORK SETUP ##################
# Load Network of choice
model         = netlib.NetworkSuperClass_ResNet50(opt)
# Network Info
print('{} Setup for {} with {} sampling on {} complete with #weights: {}'.format(' | '.join(x.upper() for x in opt.losses), opt.arch.upper(), ' | '.join(x.upper() for x in opt.sampling), \
                                                                                 opt.dataset.upper(), aux.gimme_params(model)))
print('Embeddings: {}, Sample Probs: {}'.format(' | '.join(str(x).upper() for x in opt.embed_sizes), ' | '.join(str(x).upper() for x in opt.task_p)))
# Torch device
opt.device    = torch.device('cuda')
_             = model.to(opt.device)
# List of optimization parameters. Will be appended by loss functions layer if they have learnable parameters.
to_optim   = [{'params':model.parameters(),'lr':opt.lr, 'weight_decay':opt.decay}]



"""============================================================================"""
#################### DATALOADERS SETUP ##################
#opt.all_num_classes simply collects the number of target classes for each task.
dataloaders, opt.all_num_classes = {task:{} for task in opt.tasks},[]

#### CLASS
opt.samples_per_class         = opt.cs_per_bs[0]
dataloaders['Class']          = data.give_dataloaders(opt.dataset, opt)
opt.all_num_classes.append(len(dataloaders['Class']['training'].dataset.avail_classes))

#### SHARED
opt.samples_per_class                    = opt.cs_per_bs[1]
dataloaders['Shared']['label_generator'] = dataloaders['Class']['evaluation']
# Compute initial clusters using features throughout the network (i.e. not only the final embedding.
# This allows better grouping based on both low and high-level features.)
shared_labels, image_paths               = aux.initcluster(opt, dataloaders['Shared']['label_generator'], model, num_cluster=opt.shared_num_classes)
# Using those labels, generate a new PyTorch dataloader for the auxiliary task.
dataloaders['Shared']['cluster']         = data.ClusterDataset(image_paths, shared_labels, opt.samples_per_class, opt)
dataloaders['Shared']['training']        = torch.utils.data.DataLoader(dataloaders['Shared']['cluster'], batch_size=opt.bs, num_workers=opt.kernels, shuffle=True, pin_memory=True, drop_last=True)
opt.all_num_classes.append(len(dataloaders['Shared']['training'].dataset.avail_classes))



"""============================================================================"""
#################### CREATE LOGGING FILES ###############
#Each dataset usually has a set of standard metrics to log. aux.metrics_to_examine()
#returns a dict which lists metrics to log for training ('train') and validation/testing ('val')
metrics_to_log = aux.metrics_to_examine(opt.dataset, opt.k_vals)

# example output: {'train': ['Epochs', 'Time', 'Train Loss', 'Time'],
#                  'val': ['Epochs','Time','NMI','F1', 'Recall @ 1','Recall @ 2','Recall @ 4','Recall @ 8']}

#Using the provided metrics of interest, we generate a LOGGER instance.
#Note that 'start_new' denotes that a new folder should be made in which everything will be stored.
#This includes network weights as well.
LOG = {}
LOG['Class'] = aux.LOGGER(opt, metrics_to_log, name='Class', start_new=True)
# For Logger-Settings, please refer directly to the LOGGER class in auxiliaries.py
#If graphviz is installed on the system, a computational graph of the underlying
#network can be made as well.
try:
    if opt.make_graph:
        aux.save_graph(opt, model)
    else:
        print('Not generating graph!')
except:
    # Will be thrown if graphviz is not installed (correctly).
    print('Cannot generate graph!')


"""============================================================================"""
#################### LOSS SETUP - Collecting all criterions ####################
Criterions = nn.ModuleDict()
# Add Class/Shared loss criterion to Criterion dictionary.
for i,task in enumerate(opt.tasks):
    Criterions[task], to_optim     = losses.loss_select(opt.losses[i], opt, to_optim, i)

# Add adversarial loss in given directions.
for i,mutual_task in enumerate(opt.adversarial):
    idx_target = np.where(np.array(opt.tasks)==mutual_task.split('-')[0])[0][0]
    idx_source = np.where(np.array(opt.tasks)==mutual_task.split('-')[1])[0][0]
    opt.embed_dim_target, opt.embed_dim_source = opt.embed_sizes[idx_target], opt.embed_sizes[idx_source]
    Criterions['MutualInfo-{}'.format(mutual_task)], to_optim = losses.loss_select('adversarial', opt, to_optim, i)

### Move learnable parameters to GPU
for _, loss in Criterions.items():
    _ = loss.to(opt.device)



"""============================================================================"""
#################### OPTIMIZER & SCHEDULING SETUP ####################
optimizer  = torch.optim.Adam(to_optim)

if opt.scheduler  =='exp':
    scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
elif opt.scheduler=='step':
    scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
elif opt.scheduler=='none':
    print('Not using any scheduling!')
else:
    raise Exception('No scheduling option for input: {}'.format(opt.scheduler))




"""============================================================================"""
#################### TRAINER FUNCTION ############################
def train_one_epoch(dataloaders, model, optimizer, opt, epoch, Criterions):
    start = time.time()
    # Loss collection per iteration
    loss_collect = []

    dataloader_collection = [dataloaders[task]['training'] for task in opt.tasks]
    data_iterator = tqdm(zip(*dataloader_collection), desc='Epoch {} Training...'.format(epoch), total=len(dataloader_collection[0]))

    # Iterate over both dataloaders in sequence with prob. of using it given in opt.task_p.
    for i,data in enumerate(data_iterator):
        for j,task in enumerate(opt.tasks):
            run_step = np.random.choice(2, p=[1-opt.task_p[j], opt.task_p[j]])
            if run_step:
                #### Train Class Embedding
                features  = model(data[j][1].to(opt.device))
                labels    = data[j][0]

                ## Basic DML Loss
                loss  = Criterions[task](features[task], labels)

                ### Mutual Information Loss between both embeddings
                for mutual_weight, mutual_task in zip(opt.adv_weights, opt.adversarial):
                    target, source = mutual_task.split('-')
                    mut_info_loss  = Criterions['MutualInfo-{}'.format(mutual_task)](features[target], features[source])
                    loss           = loss + mutual_weight*mut_info_loss

                ### Gradient Computation and Parameter Updating
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### Data Logging
                loss_collect.append(loss.item())

    #### MORE Data Logging
    if not len(loss_collect): loss_collect = [0]
    LOG['Class'].log('train', LOG['Class'].metrics_to_log['train'], [epoch, np.round(time.time()-start,4), np.mean(loss_collect)])



"""==========================================================================================================="""
"""==========================================================================================================="""
"""==========================================================================================================="""
#################### MAIN PART ############################
print('\n-----\n')
# Counter for cluster updates, i.e. rewriting the cluster labels used for the aux. training task.
# Rewritten happends if this value hits opt.cluster_update_freq.
opt.cluster_update_counter = 0

for epoch in range(opt.n_epochs):
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))


    ### Train one epoch
    _ = model.train()
    train_one_epoch(dataloaders, model, optimizer, opt, epoch, Criterions)


    ### Evaluate  - Give required information to evaluation function.
    _ = model.eval()
    if opt.dataset in ['cars196', 'cub200', 'online_products']:
        eval_params = {'dataloader':dataloaders['Class']['testing'], 'model':model, 'opt':opt}
    elif opt.dataset=='in-shop':
        eval_params = {'query_dataloader':dataloaders['Class']['testing_query'], 'gallery_dataloader':dataloaders['Class']['testing_gallery'], 'model':model, 'opt':opt}
    elif opt.dataset=='vehicle_id':
        eval_params = {'dataloaders':[dataloaders['Class']['testing_set1'], dataloaders['Class']['testing_set2'], dataloaders['Class']['testing_set3']], 'model':model, 'opt':opt}
    eval_params['epoch'] = epoch

    eval.evaluate(opt.dataset, LOG, evaltype='Class', save=True,  **eval_params)

    # Update Summary/Performance plot
    LOG['Class'].update_info_plot()

    ### Update Cluster Information
    if opt.cluster_update_counter==opt.cluster_update_freq:
        new_shared_labels = aux.deepcluster(opt, dataloaders['Shared']['label_generator'], model, num_cluster=opt.all_num_classes[1])
        dataloaders['Shared']['training'].dataset.update_labels(new_shared_labels)
        opt.cluster_update_counter = 0
    else:
        opt.cluster_update_counter+= 1


    ### Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()

    print('\n-----\n')


    ### Write Training Summary
    LOG['Class'].write_summary()
