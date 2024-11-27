

import torch
import torch.nn as nn
import config
import torch
from torch.nn import Module, LSTM, Linear, GRU
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from zmq import device
import torch.nn.functional as F   
import matplotlib.pyplot as plt
import json



class Net_Encoder(Module):
    def __init__(self, config):
        super(Net_Encoder, self).__init__()
        self.LSTM = LSTM(input_size=config.model_E_config["input_size"], hidden_size=config.model_E_config["hidden_size"],
                         num_layers=config.model_E_config["num_layers"], batch_first=True, dropout=config.model_E_config["dropout"],
                         bidirectional=True)  
        self.linear1 = Linear(in_features=config.model_E_config["hidden_size"], out_features=config.model_E_config["hidden_size"])
        
    def forward(self, x, hidden=None):
        lstm_out, (hidden, cell) = self.LSTM(x, hidden)  
        linear_out = self.linear1(lstm_out)
        return linear_out, (hidden, cell)



class Net_Decoder_T(Module):
    def __init__(self, config):
        super(Net_Decoder_T, self).__init__()
        self.LSTM = LSTM(input_size=config.model_E_config["hidden_size"], hidden_size=config.model_DT_config["hidden_size"],
                         num_layers=config.model_DT_config["num_layers"], batch_first=True, dropout=config.model_DT_config["dropout"],
                         bidirectional=True)  
        self.linear1 = Linear(in_features=config.model_DT_config["hidden_size"], out_features=config.model_DT_config["hidden_size"])
        self.linear2 = Linear(in_features=config.model_DT_config["hidden_size"], out_features=config.model_DT_config["output_size"])
        
    def forward(self, x, hidden=None):
        lstm_out, (hidden, cell) = self.LSTM(x, hidden)  
        linear_out = F.tanh(self.linear1(lstm_out[:,-1,:]))
        linear_out = self.linear2(linear_out)
        return linear_out, (hidden, cell)
