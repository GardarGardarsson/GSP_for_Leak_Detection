#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

+-------------------------------+
|                               |
|      G N N   M O D E L S      | 
|                               |
+-------------------------------+

Created on Wed Jul 28 15:05:55 2021

@author: gardar
'''


import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

# Note on choosing the K-order of the polynomial for the the ChebConv layers
# 'K' effectively denotes how far the information propagates within a layer
# From G. Hajgató et al. [arXiv:2104.13619]:
# ----------------------------------------------------------------------------
#   From this perspective, the order of the polynomial determined by the 
#   demand that the observed information reaches all the unobserved nodes. 
#   This implies that – as long as the observation points are selected 
#   uniformly randomly in the graph – the polynomial order should be at 
#   least equal to the graph diameter.
# ----------------------------------------------------------------------------

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels, 120, K=240)
        self.conv2 = ChebConv(120, 60, K=120)
        self.conv3 = ChebConv(60, 30, K=20)
        self.conv4 = ChebConv(30, out_channels, K=1, bias=False)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)
        torch.nn.init.xavier_normal_(self.conv4.weight)

    def forward(self, data):
        x, edge_index, edge_weight  = data.x, data.edge_index, data.weight
        x = F.silu(self.conv1(x, edge_index, edge_weight))
        x = F.silu(self.conv2(x, edge_index, edge_weight))
        x = F.silu(self.conv3(x, edge_index, edge_weight))
        x = self.conv4(x, edge_index, edge_weight)
        return torch.sigmoid(x)

class GCN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, \
                 dropout, return_embeds=False):
        
        super(GCN, self).__init__()
        
        # A list of GCN convolutional layers
        self.convs = None
        
        # A list of batch normalisation layers
        self.bns = None
        
        
        '''
        IMPLEMENT PARENT AND CHILDREN CLASS
        PARENT CLASS HAS TRAINING, EVALUATE, SAVE, LOAD MODEL
        CHILDREN CLASS HAS VARIOUS DIFFERENT NETWORK SETUPS I.E.
        CHEBNET / DCRNN ETC ETC ...
        '''
        
        '''
        CHECK OUT THESE BAD BOYS!
        DCRNN
        Diffusion Convolutional Recurrent Neural Network
        https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html
        '''
        
        

