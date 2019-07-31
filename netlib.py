# Copyright (C) 2019 Karsten Roth and Biagio Brattoli
#
# This file is part of DeepMetricLearning_with_InterClassCharacteristics
#
# DeepMetricLearning_with_InterClassCharacteristics is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepMetricLearning_with_InterClassCharacteristics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""=================================================================="""
#################### LIBRARIES #################
import torch, os, numpy as np

import torch.nn as nn
import pretrainedmodels as ptm



"""=================================================================================================================================="""
### NETWORK CONTAINER for RESNET50
# Network container to run with MIC using a ResNet50-backend.
class NetworkSuperClass_ResNet50(nn.Module):
    def __init__(self, opt):
        super(NetworkSuperClass_ResNet50, self).__init__()

        self.pars = opt

        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)


        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None


        ### Set Embedding Layer
        in_feat = self.model.last_linear.in_features
        self.out_modes, self.embed_sizes   = opt.tasks, opt.embed_sizes
        self.model.last_linear = nn.ModuleDict({task: torch.nn.Linear(in_feat, self.embed_sizes[i]) for i,task in enumerate(self.out_modes)})


        ### Resid. Blocks broken down for specific targeting. Primarily used for initial clustering which makes use
        ### of features at different levels.
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])



    def forward(self, x, is_init_cluster_generation=False):
        itermasks, out_coll = [],{}

        #Compute First Layer Output
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        if is_init_cluster_generation:
            #If the first clusters before standardization are computed: We use the initial layers with strong
            #average pooling. Using these, we saw much better initial grouping then when using layer combinations or
            #only the last layer.
            x = torch.nn.functional.avg_pool2d(self.model.layer1(x),18,12)
            x = torch.nn.functional.normalize(x.view(x.size(0),-1))
            return x
        else:
            #Run Rest of ResNet
            for layerblock in self.layer_blocks:
                x = layerblock(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0),-1)

            #Store the final conv. layer output, it might be useful.
            out_coll['Last'] = x
            for out_mode in self.out_modes:
                mod_x = self.model.last_linear[out_mode](x)
                out_coll[out_mode] = torch.nn.functional.normalize(mod_x, dim=-1)
            return out_coll
