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

import numpy as np, os, sys, pandas as pd, csv, copy
import torch, torch.nn as nn, matplotlib.pyplot as plt, random

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import pretrainedmodels.utils as utils
import auxiliaries as aux



"""=================================================================="""
################ FUNCTION TO RETURN ALL DATALOADERS NECESSARY ####################
def give_dataloaders(dataset, opt):
    ### ImageNet Properties
    opt.mean, opt.std, opt.input_space, opt.input_range = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 'RGB', [0,1]
    if 'class_samples_per_class' in vars(opt).keys():
        opt.samples_per_class = opt.class_samples_per_class

    if opt.dataset=='cub200':
        datasets = give_CUB200_datasets(opt)
    elif opt.dataset=='cars196':
        datasets = give_CARS196_datasets(opt)
    elif opt.dataset=='online_products':
        datasets = give_OnlineProducts_datasets(opt)
    elif opt.dataset=='in-shop':
        datasets = give_InShop_datasets(opt)
    elif opt.dataset=='vehicle_id':
        datasets = give_VehicleID_datasets(opt)
    else:
        raise Exception('No Dataset >{}< available!'.format(dataset))

    dataloaders = {}
    for key,dataset in datasets.items():
        is_val = dataset.is_validation
        dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.kernels, shuffle=not is_val, pin_memory=True, drop_last=not is_val)

    return dataloaders


################# FUNCTIONS TO RETURN TRAIN/VAL PYTORCH DATASETS FOR CUB200, CARS196 AND STANFORD ONLINE PRODUCTS ####################################
def give_CUB200_datasets(opt):
    """
    This function generates a training, testing and training-evaluation dataloader for Metric Learning on the CUB200-2011 dataset.
    For Metric Learning, the dataset is sorted by name, with the first half used for training while the last half is used for testing.
    So no random shuffling of classes. Note that the training-evaluation dataloader is required for the cluster generation as we do not want
    to perform augmentation during that time.
    """
    #### The following lines aggregate all image-paths into a list of form [(class_idx, path_to_image),...]
    image_sourcepath  = opt.source_path+'/images'
    image_classes     = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0]))
    conversion        = {int(x.split('.')[0]):x.split('.')[-1] for x in image_classes}
    image_list        = {int(key.split('.')[0]):sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes}
    image_list        = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list        = [x for y in image_list for x in y]

    ####
    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)


    ####
    keys = sorted(list(image_dict.keys()))
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]
    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}


    ####
    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    val_dataset   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    train_dataset.conversion = conversion
    val_dataset.conversion   = conversion
    eval_dataset.conversion   = conversion

    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}


def give_CARS196_datasets(opt):
    """
    This function generates a training, testing and training-evaluation dataloader for Metric Learning on the CARS196 dataset.
    For Metric Learning, the dataset is sorted by name, with the first half used for training while the last half is used for testing.
    So no random shuffling of classes. Note that the training-evaluation dataloader is required for the cluster generation as we do not want
    to perform augmentation during that time.
    """
    #### The following lines aggregate all image-paths into a list of form [(class_idx, path_to_image),...]
    image_sourcepath  = opt.source_path+'/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    conversion    = {i:x for i,x in enumerate(image_classes)}
    image_list    = {i:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key)]) for i,key in enumerate(image_classes)}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    ####
    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    ####
    keys = sorted(list(image_dict.keys()))
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]
    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}

    ####
    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    val_dataset   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    train_dataset.conversion = conversion
    val_dataset.conversion   = conversion
    eval_dataset.conversion   = conversion

    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}


def give_OnlineProducts_datasets(opt):
    """
    This function generates a training, testing, training-evaluation dataloader for Metric Learning on the Stanford Online Products dataset.
    The required training/validation(test) splitting is provided in accompanying text-files. Note that the training-evaluation dataloader
    is required for the cluster generation as we do not want to perform augmentation during that time.
    """
    #### Load image names and respective classes from provided text files.
    image_sourcepath  = opt.source_path+'/images'
    training_files = pd.read_table(opt.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(opt.source_path+'/Info_Files/Ebay_test.txt', header=0, delimiter=' ')

    #### Create dictionaries converting class-ids to string-names.
    conversion = {}
    for class_id, path in zip(training_files['class_id'],training_files['path']):
        conversion[class_id] = path.split('/')[0]
    for class_id, path in zip(test_files['class_id'],test_files['path']):
        conversion[class_id] = path.split('/')[0]

    #### Create Training/Test/Evaluation dictionaries {class_id:[image_paths in this class]}
    train_image_dict, val_image_dict, super_train_image_dict  = {},{},{}
    for key, img_path in zip(training_files['class_id'],training_files['path']):
        key = key-1
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(test_files['class_id'],test_files['path']):
        key = key-1
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_image_dict[key].append(image_sourcepath+'/'+img_path)


    ####
    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    val_dataset   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    train_dataset.conversion = conversion
    val_dataset.conversion   = conversion
    eval_dataset.conversion   = conversion

    return {'training':train_dataset, 'testing':val_dataset, 'evaluation':eval_dataset}


def give_InShop_datasets(opt):
    """
    This function generates a training, query- and gallery as well as a training-evaluation dataloader for Metric Learning on the In-Shop Clothes dataset.
    The required training/validation(test) splitting is provided in accompanying text-files. In this specific case, testing is done using provided query images,
    which are used to fetch images from a gallery dataset. Note that the training-evaluation dataloader is required for the cluster generation as we do not want to perform augmentation during that time.
    """
    #### Load image names and respective classes from provided text files.
    data_info = np.array(pd.read_table(opt.source_path+'/Eval/list_eval_partition.txt', header=1, delim_whitespace=True))[1:,:]
    train, query, gallery   = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]
    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
    train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])
    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
    query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
    gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

    ####
    train_image_dict    = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(opt.source_path+'/'+img_path)

    query_image_dict    = {}
    for img_path, key in query:
        if not key in query_image_dict.keys():
            query_image_dict[key] = []
        query_image_dict[key].append(opt.source_path+'/'+img_path)

    gallery_image_dict    = {}
    for img_path, key in gallery:
        if not key in gallery_image_dict.keys():
            gallery_image_dict[key] = []
        gallery_image_dict[key].append(opt.source_path+'/'+img_path)

    ####
    train_dataset       = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    eval_dataset      = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    query_dataset     = BaseTripletDataset(query_image_dict, opt, is_validation=True)
    gallery_dataset   = BaseTripletDataset(gallery_image_dict, opt, is_validation=True)

    return {'training':train_dataset, 'testing_query':query_dataset, 'evaluation':eval_dataset, 'testing_gallery':gallery_dataset}



def give_VehicleID_datasets(opt):
    """
    This function generates a training, testing, training-evaluation dataloader for Metric Learning on the PKU Vehicle-ID dataset.
    The required training/validation(test) splitting is provided in accompanying text-files. For this dataset, multiple provided test-sets in different
    sizes are already giving. Note that the training-evaluation dataloader is required for the cluster generation as we do not want to perform augmentation during that time.
    """
    #### Load image names and respective classes from provided text files.
    train = np.array(pd.read_table(opt.source_path+'/train_test_split/train_list.txt', header=None, delim_whitespace=True))
    small_test  = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_800.txt', header=None, delim_whitespace=True))
    medium_test = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_1600.txt', header=None, delim_whitespace=True))
    big_test    = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_2400.txt', header=None, delim_whitespace=True))
    lab_conv = {x:i for i,x in enumerate(np.unique(train[:,1]))}
    train[:,1] = np.array([lab_conv[x] for x in train[:,1]])
    lab_conv = {x:i for i,x in enumerate(np.unique(np.concatenate([small_test[:,1], medium_test[:,1], big_test[:,1]])))}
    small_test[:,1]  = np.array([lab_conv[x] for x in small_test[:,1]])
    medium_test[:,1] = np.array([lab_conv[x] for x in medium_test[:,1]])
    big_test[:,1]    = np.array([lab_conv[x] for x in big_test[:,1]])

    ####
    train_image_dict    = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    small_test_dict = {}
    for img_path, key in small_test:
        if not key in small_test_dict.keys():
            small_test_dict[key] = []
        small_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    medium_test_dict    = {}
    for img_path, key in medium_test:
        if not key in medium_test_dict.keys():
            medium_test_dict[key] = []
        medium_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    big_test_dict    = {}
    for img_path, key in big_test:
        if not key in big_test_dict.keys():
            big_test_dict[key] = []
        big_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    ####
    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt,    is_validation=True)
    val_small_dataset     = BaseTripletDataset(small_test_dict, opt,  is_validation=True)
    val_medium_dataset    = BaseTripletDataset(medium_test_dict, opt, is_validation=True)
    val_big_dataset       = BaseTripletDataset(big_test_dict, opt,    is_validation=True)

    return {'training':train_dataset, 'testing_set1':val_small_dataset, 'testing_set2':val_medium_dataset, \
            'testing_set3':val_big_dataset, 'evaluation':eval_dataset}




"""=================================================================="""
################## BASIC PYTORCH DATASET USED FOR ALL DATALOADERS ##################################
class BaseTripletDataset(Dataset):
    def __init__(self, image_dict, opt, samples_per_class=8, is_validation=False):
        #Number of files dictating the dataset length
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])
        #Validation flag. If set, no augmentation is performed!
        self.is_validation = is_validation
        #All externally set training parameters.
        self.pars        = opt
        #Dict with all training classes and respective image paths.
        self.image_dict  = image_dict
        #All available training classes.
        self.avail_classes    = sorted(list(self.image_dict.keys()))
        #Convert image dictionary from classname:content to class_idx:content
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))

        #For a training dataset: initialize starting class and a list of classes already
        #visited as well as number of samples already drawn for the current class selection.
        if not self.is_validation:
            self.samples_per_class = samples_per_class
            #Select current class to sample images from up to <samples_per_class>
            self.current_class   = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0


        ##### Set up Augmentation/Basic Preprocessing Function
        #Normalize with ImageNet defaults.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf_list = []
        if not self.is_validation:
            transf_list.extend([transforms.RandomResizedCrop(size=224),
                                transforms.RandomHorizontalFlip(0.5)])
        else:
            transf_list.extend([transforms.Resize(256),
                                transforms.CenterCrop(224)])

        transf_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transf_list)

        # Create additionally list of tuples (class, image_paths) for easier sampling.
        self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        # Init flag to denote first call to the dataset.
        self.is_init = True

    def ensure_3dim(self, img):
        """
        Function to ensure input imags are 3D and in RGB style.

        Input:
            img: img to ensure to be of RGB style (i.e. BSx3xWxH)
        """
        if len(img.size)==2:
            img = img.convert('RGB')
        return img


    def __getitem__(self, idx):
        if self.is_init:
            self.current_class = self.avail_classes[idx%len(self.avail_classes)]
            self.is_init = False

        if not self.is_validation:
            if self.samples_per_class==1:
                #For cases like ProxyNCA, samples do not need to be aggregated for each class.
                return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

            if self.n_samples_drawn==self.samples_per_class:
                #Once enough samples per class have been drawn, we choose another class to draw samples from.
                #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                #previously or one before that.
                counter = copy.deepcopy(self.avail_classes)
                for prev_class in self.classes_visited:
                    if prev_class in counter: counter.remove(prev_class)
                self.current_class   = counter[idx%len(counter)]

                self.classes_visited = self.classes_visited[1:]+[self.current_class]
                self.n_samples_drawn = 0

            class_sample_idx = idx%len(self.image_dict[self.current_class])
            self.n_samples_drawn += 1

            out_img = self.transform(self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx])))
            return self.current_class,out_img
        else:
            return self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

    def __len__(self):
        return self.n_files




"""=================================================================="""
################## BASIC PYTORCH DATASET FOR INTERCLASS/AUXILIARY TASK TRAINING ##################################
class ClusterDataset(Dataset):
    """
    For more detailed explanations of used variables, see the Basic Dataloader above.
    """
    def __init__(self, image_paths, image_labels, samples_per_class, opt):
        #Number of files dictating the dataset length
        self.n_files           = len(image_paths)
        #
        self.samples_per_class = samples_per_class
        #
        self.pars        = opt
        #
        self.image_paths = image_paths
        ### Update training image labels based on precomputed cluster-ids passed thorugh image_labels.
        self.update_labels(image_labels)

        #####
        transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
                       utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]
        self.transform = transforms.Compose(transf_list)


    def update_labels(self, image_labels):
        """
        Function to update training labels for each training sample.

        Inputs:
            image_labels: list, each element at its position correspond to the mined clusterlabel for BaseTripletDataset.image_list
        """
        self.avail_classes = np.unique(image_labels)
        self.counter       = copy.deepcopy(list(self.avail_classes))

        self.image_labels  = image_labels
        self.indexer       = {i:np.where(image_labels==i)[0] for i in self.avail_classes}

        self.current_class   = np.random.randint(len(self.avail_classes))
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

        self.image_list = [[(self.image_paths[x],key) for x in self.indexer[key]] for key in self.indexer.keys()]
        self.image_list = [x for y in self.image_list for x in y]


    def ensure_3dim(self, img):
        """
        Function to ensure input imags are 3D and in RGB style.

        Input:
            img: img to ensure to be of RGB style (i.e. BSx3xWxH)
        """
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.samples_per_class==1:
            return (self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0]))))

        if self.n_samples_drawn==self.samples_per_class:
            #Once enough samples per class have been drawn, we choose another class to draw samples from.
            #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            #previously or one before that.
            self.counter = copy.deepcopy(list(self.avail_classes))
            for prev_class in self.classes_visited:
                if prev_class in self.counter: self.counter.remove(prev_class)
            self.current_class   = np.random.choice(self.counter)
            self.classes_visited = self.classes_visited[1:]+[self.current_class]
            self.n_samples_drawn = 0

        self.n_samples_drawn += 1

        ### Perform random sample switching for a given class. Increase stability during training and avoid overfitting to a bad prior clustering.
        if not np.random.choice(2,p=[self.pars.random_cluster_pick_p, 1-self.pars.random_cluster_pick_p]):
            class_choice = np.random.choice(self.avail_classes)
        else:
            class_choice = self.current_class
        idx = self.indexer[class_choice][idx%len(self.indexer[class_choice])]

        return self.current_class,self.transform(self.ensure_3dim(Image.open(self.image_paths[idx])))


    def __len__(self):
        return self.n_files
