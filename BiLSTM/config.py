# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:53:58 2024

@author: jings
"""

import torch

num_epochs = 10000           
batch_size = 128            
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_DT_config = {
    'input_size': 4,         
    'hidden_size': 128,      
    'num_layers': 3,         
    'output_size': 4,       
    'dropout': 0.2,     
    'learning_rate':0.001
}

loss_fn = 'MSELoss'          