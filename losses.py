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

import random, itertools as it, numpy as np, faiss, random

import torch

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
from PIL import Image



"""================================================================================================="""
def loss_select(loss, opt, to_optim, param_idx):
    """
    Selection function which returns the respective criterion while appending to list of trainable parameters if required.

    Args:
        loss:      str, name of loss function to return.
        opt:       argparse.Namespace, contains all training-specific parameters.
        to_optim:  list of trainable parameters. Is extend if loss function contains those as well.
        param_idx: Index denoting the task for which the loss is set up. 0==Class, 1==Aux.
    Returns:
        criterion (torch.nn.Module inherited), to_optim (optionally appended)
    """
    i = param_idx
    if loss=='triplet':
        loss_params  = {'margin':opt.margin[i], 'sampling_method':opt.sampling[i]}
        criterion    = TripletLoss(**loss_params)
    elif loss=='marginloss':
        loss_params  = {'margin':opt.margin[i], 'nu': opt.nu[i],
                        'beta':opt.beta[i], 'n_classes':opt.all_num_classes[i],
                        'sampling_method':opt.sampling[i]}
        criterion    = MarginLoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.beta_lr[i], 'weight_decay':0}]
    elif loss=='proxynca':
        loss_params  = {'num_proxies':opt.all_num_classes[i], 'embedding_dim':opt.embed_sizes[i]}
        criterion    = ProxyNCALoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.proxy_lr[i]}]
    elif loss=='adversarial':
        loss_params  = {'class_dim':opt.embed_dim_target, 'aux_dim':opt.embed_dim_source,
                        'proj_dim': opt.adv_dim}
        criterion    = AdvLoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.lr, 'weight_decay':1e-6}]
    else:
        raise Exception('Loss {} not available!'.format(loss))

    return criterion, to_optim





"""================================================================================================="""
### Sampler() holds all possible triplet sampling options: random, SemiHardNegative & Distance-Weighted.
class Sampler():
    def __init__(self, method='random'):
        self.method = method
        if method=='semihard':
            self.give = self.semihardsampling
        elif method=='distance':
            self.give = self.distanceweightedsampling
        elif method=='random':
            self.give = self.randomsampling

    def randomsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and randomly
        selects <len(batch)> triplets.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        unique_classes = np.unique(labels)
        indices        = np.arange(len(batch))
        class_dict     = {i:indices[labels==i] for i in unique_classes}

        sampled_triplets = [list(it.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        #NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
        return sampled_triplets

    def semihardsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on semihard sampling introduced in 'Deep Metric Learning via Lifted Structured Feature Embedding'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            neg = labels!=l; pos = labels==l

            anchors.append(i)
            pos[i] = 0
            p      = np.random.choice(np.where(pos)[0])
            positives.append(p)

            #Find negatives that violate tripet constraint semi-negatives
            neg_mask = np.logical_and(neg,d>d[p])
            neg_mask = np.logical_and(neg_mask,d<self.margin+d[p])
            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

    def distanceweightedsampling(self, batch, labels, lower_cutoff=0.5, upper_cutoff=1.4):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on distance sampling introduced in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
            lower_cutoff: float, lower cutoff value for negatives that are too close to anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
            upper_cutoff: float, upper cutoff value for positives that are too far away from the anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)



        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))
            #Sample negatives by distance
            negatives.append(np.random.choice(bs,p=q_d_inv))

        sampled_triplets = [[a,p,n] for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets


    def pdist(self, A, eps = 1e-4):
        """
        Efficient function to compute the distance matrix for a matrix A.

        Args:
            A:   Matrix/Tensor for which the distance matrix is to be computed.
            eps: float, minimal distance/clampling value to ensure no zero values.
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = eps).sqrt()


    def inverse_sphere_distances(self, batch, dist, labels, anchor_label):
        """
        Function to utilise the distances of batch samples to compute their
        probability of occurence, and using the inverse to sample actual negatives to the resp. anchor.

        Args:
            batch:        torch.Tensor(), batch for which the sampling probabilities w.r.t to the anchor are computed. Used only to extract the shape.
            dist:         torch.Tensor(), computed distances between anchor to all batch samples.
            labels:       np.ndarray, labels for each sample for which distances were computed in dist.
            anchor_label: float, anchor label
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        bs,dim       = len(dist),batch.shape[-1]

        #negated log-distribution of distances of unit sphere in dimension <dim>
        log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
        #Set sampling probabilities of positives to zero
        log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

        q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
        #Set sampling probabilities of positives to zero
        q_d_inv[np.where(labels==anchor_label)[0]] = 0

        ### NOTE: Cutting of values with high distances made the results slightly worse.
        # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

        q_d_inv = q_d_inv/q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()




"""================================================================================================="""
### Standard Triplet Loss, finds triplets in Mini-batches.
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1, sampling_method='random'):
        """
        Basic Triplet Loss as proposed in 'FaceNet: A Unified Embedding for Face Recognition and Clustering'
        Args:
            margin:             float, Triplet Margin - Ensures that positives aren't placed arbitrarily close to the anchor.
                                Similarl, negatives should not be placed arbitrarily far away.
            sampling_method:    Method to use for sampling training triplets. Used for the Sampler-class.
        """
        super(TripletLoss, self).__init__()
        self.margin             = margin
        self.sampler            = Sampler(method=sampling_method)

    def triplet_distance(self, anchor, positive, negative):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            triplet loss (torch.Tensor(), batch-averaged)
        """
        #Sample triplets to use for training.
        sampled_triplets = self.sampler.give(batch, labels)
        #Compute triplet loss
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        return torch.mean(loss)




"""================================================================================================="""
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class MarginLoss(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance'):
        """
        Basic Margin Loss as proposed in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            margin:          float, fixed triplet margin (see also TripletLoss).
            nu:              float, regularisation weight for beta. Zero by default (in literature as well).
            beta:            float, initial value for trainable class margins. Set to default literature value.
            n_classes:       int, number of target class. Required because it dictates the number of trainable class margins.
            beta_constant:   bool, set to True if betas should not be trained.
            sampling_method: str, sampling method to use to generate training triplets.
        Returns:
            Nothing!
        """
        super(MarginLoss, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant

        self.beta_val = beta
        self.beta     = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampling_method    = sampling_method
        self.sampler            = Sampler(method=sampling_method)


    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            margin loss (torch.Tensor(), batch-averaged)
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        sampled_triplets = self.sampler.give(batch, labels)

        #Compute distances between anchor-positive and anchor-negative.
        d_ap, d_an = [],[]
        for triplet in sampled_triplets:
            train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

            pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
            neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

        #Group betas together by anchor class in sampled triplets (as each beta belongs to one class).
        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        #Compute actual margin postive and margin negative loss
        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        #Compute normalization constant
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        #Actual Margin Loss
        loss = torch.sum(pos_loss+neg_loss) if pair_count==0. else torch.sum(pos_loss+neg_loss)/pair_count

        #(Optional) Add regularization penalty on betas.
        # if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss




"""================================================================================================="""
### ProxyNCALoss containing trainable class proxies. Works independent of batch size.
class ProxyNCALoss(torch.nn.Module):
    def __init__(self, num_proxies, embedding_dim):
        """
        Basic ProxyNCA Loss as proposed in 'No Fuss Distance Metric Learning using Proxies'.

        Args:
            num_proxies:     int, number of proxies to use to estimate data groups. Usually set to number of classes.
            embedding_dim:   int, Required to generate initial proxies which are the same size as the actual data embeddings.
        Returns:
            Nothing!
        """
        super(ProxyNCALoss, self).__init__()
        self.num_proxies   = num_proxies
        self.embedding_dim = embedding_dim
        self.PROXIES = torch.nn.Parameter(torch.randn(num_proxies, self.embedding_dim) / 8)
        self.all_classes = torch.arange(num_proxies)


    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            proxynca loss (torch.Tensor(), batch-averaged)
        """
        #Normalize batch in case it is not normalized (which should never be the case for ProxyNCA, but still).
        #Same for the PROXIES. Note that the multiplication by 3 seems arbitrary, but helps the actual training.
        batch       = 3*torch.nn.functional.normalize(batch, dim=1)
        PROXIES     = 3*torch.nn.functional.normalize(self.PROXIES, dim=1)
        #Group required proxies
        pos_proxies = torch.stack([PROXIES[pos_label:pos_label+1,:] for pos_label in labels])
        neg_proxies = torch.stack([torch.cat([self.all_classes[:class_label],self.all_classes[class_label+1:]]) for class_label in labels])
        neg_proxies = torch.stack([PROXIES[neg_labels,:] for neg_labels in neg_proxies])
        #Compute Proxy-distances
        dist_to_neg_proxies = torch.sum((batch[:,None,:]-neg_proxies).pow(2),dim=-1)
        dist_to_pos_proxies = torch.sum((batch[:,None,:]-pos_proxies).pow(2),dim=-1)
        #Compute final proxy-based NCA loss
        negative_log_proxy_nca_loss = torch.mean(dist_to_pos_proxies[:,0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))
        return negative_log_proxy_nca_loss




"""================================================================================================="""
### Gradient Reversal Layer
class GradRev(torch.autograd.Function):
    """
    Implements an autograd class to flip gradients during backward pass.
    """
    def forward(self, x):
        """
        Container which applies a simple identity function.

        Input:
            x: any torch tensor input.
        """
        return x.view_as(x)

    def backward(self, grad_output):
        """
        Container to reverse gradient signal during backward pass.

        Input:
            grad_output: any computed gradient.
        """
        return (grad_output * -1.)

### Gradient reverse function
def grad_reverse(x):
    """
    Applies gradient reversal on input.

    Input:
        x: any torch tensor input.
    """
    return GradRev()(x)




"""================================================================================================="""
### Adversarial Loss
class AdvLoss(torch.nn.Module):
    def __init__(self, class_dim, aux_dim, proj_dim=512):
        """
        Adversial Loss Function that uses a projection network to decorrelate two embeddings living in
        DIFFERENT embedding spaces. While the projection network learns to closely project both embeddings,
        the gradient reversal ensures that the embeddings are actually decorrelated.
        """
        super(AdvLoss, self).__init__()
        self.class_dim, self.aux_dim = class_dim, aux_dim
        self.proj_dim = proj_dim

        #Projection network
        self.regressor = torch.nn.Sequential(torch.nn.Linear(self.aux_dim, proj_dim), torch.nn.ReLU(), torch.nn.Linear(proj_dim, self.class_dim)).type(torch.cuda.FloatTensor)


    def forward(self, class_features, aux_features):
        #Apply gradient reversal on input embeddings.
        features = [torch.nn.functional.normalize(grad_reverse(class_features),dim=-1), torch.nn.functional.normalize(grad_reverse(aux_features),dim=-1)]
        #Project one embedding to the space of the other (with normalization), then compute the correlation.
        sim_loss = -1.*torch.mean(torch.mean((features[0]*torch.nn.functional.normalize(self.regressor(features[1]),dim=-1))**2,dim=-1))
        return sim_loss
