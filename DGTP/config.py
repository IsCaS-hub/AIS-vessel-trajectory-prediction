# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:53:58 2024

@author: Lenovo
"""


import torch

# 通用训练参数
num_epochs = 10000          
batch_size = 128           
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_E_config = {
    'input_size': 4,         
    'hidden_size': 128,     
    'num_layers': 3,      
    'output_size': 4,        
    'dropout': 0.2,         
    'learning_rate':0.001
}

model_E2_config = {
    'input_size': 4,         
    'hidden_size': 128,     
    'num_layers': 3,         
    'output_size': 4,      
    'dropout': 0.2,         
    'learning_rate':0.001
}

model_DT_config = {
    'input_size': 4,        
    'hidden_size': 128,    
    'num_layers': 3,         
    'output_size': 4,      
    'dropout': 0.2,     
    'learning_rate':0.001  
}

model_DD_config = {
    'input_size': 4,        
    'hidden_size': 128,      
    'num_layers': 3,        
    'output_size': 4,      
    'dropout': 0.2,          
    'learning_rate':0.001
}
