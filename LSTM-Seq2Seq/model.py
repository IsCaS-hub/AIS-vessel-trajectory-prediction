# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:55:31 2024

@author: Lenovo
"""
"""
import torch
import torch.nn as nn
import config

# 第一个 BiGRU，用于目的地预测（输出一个点）
class BiGRUEncoderDecoderDest(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(BiGRUEncoderDecoderDest, self).__init__()
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Encoder 处理输入
        encoder_output, hidden = self.encoder(x)
        # 只取 encoder_output 最后一个时间步的输出
        output = self.fc(encoder_output[:, -1, :])  # 取最后一个时间步的输出
        output = output.unsqueeze(1) # 扩展为（batch_size,1,4）
        return output

# 第二个 BiGRU，用于未来轨迹预测（输出多个时间步）
class BiGRUEncoderDecoderFuture(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(BiGRUEncoderDecoderFuture, self).__init__()
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.decoder = nn.GRU(hidden_size * 2, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Encoder 处理输入
        encoder_output, hidden = self.encoder(x)
        # Decoder：用 encoder 的输出作为 decoder 的输入
        decoder_output, _ = self.decoder(encoder_output, hidden)
        # 预测多个时间步的输出
        output = self.fc(decoder_output)[:,:10,:]  # 取前十个时间步的输出
        return output

# 定义级联 BiGRU 模型
class CascadedBiGRU(nn.Module):
    def __init__(self):
        super(CascadedBiGRU, self).__init__()
        # 第一个阶段用于目的地预测
        self.first_stage = BiGRUEncoderDecoderDest(
            input_size=config.model_dest_config["input_size"],
            hidden_size=config.model_dest_config["hidden_size"],
            output_size=config.model_dest_config["output_size"],  # 目的地预测（4维：LAT, LON, SOG, COG）
            num_layers=config.model_dest_config["num_layers"],
            dropout=config.model_dest_config["dropout"]
        )

        # 第二个阶段用于未来轨迹预测
        self.second_stage = BiGRUEncoderDecoderFuture(
            input_size=config.model_future_config["input_size"],
            hidden_size=config.model_future_config["hidden_size"],
            output_size=config.model_future_config["output_size"],  # 未来轨迹预测（4维：LAT, LON, SOG, COG）
            num_layers=config.model_future_config["num_layers"],
            dropout=config.model_future_config["dropout"]
        )

    def forward(self, x):
        # 第一阶段：目的地预测
        destination_pred = self.first_stage(x)
        # print('destination_pred.shape', destination_pred.shape)  # 输出形状应为 (batch_size, 4)

        # 扩展目的地预测的维度，并与历史轨迹拼接
        # destination_pred = destination_pred.unsqueeze(1)  # 扩展为 (batch_size, 1, 4)
        x_extended = torch.cat((x, destination_pred), dim=1)  # 拼接为 (batch_size, seq_len + 1, 4)
        # print('x_extended.shape', x_extended.shape)

        # 第二阶段：未来轨迹预测
        future_trajectory_pred = self.second_stage(x_extended.detach())
        # print('future_trajectory_pred.shape', future_trajectory_pred.shape)  # 输出应为 (batch_size, 10, 4)

        return destination_pred, future_trajectory_pred
"""

import torch
import torch.nn as nn
import config
import torch
from torch.nn import Module, LSTM, Linear, GRU
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from zmq import device
import torch.nn.functional as F   # 用于 one-hot 编码
import matplotlib.pyplot as plt
import json



class Net_Encoder(Module):
    def __init__(self, config):
        super(Net_Encoder, self).__init__()
        self.LSTM = LSTM(input_size=config.model_E_config["input_size"], hidden_size=config.model_E_config["hidden_size"],
                         num_layers=config.model_E_config["num_layers"], batch_first=True, dropout=config.model_E_config["dropout"],
                         bidirectional=False)  # Replace GRU with LSTM
        self.linear1 = Linear(in_features=config.model_E_config["hidden_size"], out_features=config.model_E_config["hidden_size"])
        
    def forward(self, x, hidden=None):
        lstm_out, (hidden, cell) = self.LSTM(x, hidden)  # LSTM returns hidden and cell states
        linear_out = self.linear1(lstm_out)
        return linear_out, (hidden, cell)



class Net_Decoder_T(Module):
    def __init__(self, config):
        super(Net_Decoder_T, self).__init__()
        self.LSTM = LSTM(input_size=config.model_E_config["hidden_size"], hidden_size=config.model_DT_config["hidden_size"],
                         num_layers=config.model_DT_config["num_layers"], batch_first=True, dropout=config.model_DT_config["dropout"],
                         bidirectional=False)  # Replace GRU with LSTM
        self.linear1 = Linear(in_features=config.model_DT_config["hidden_size"], out_features=config.model_DT_config["hidden_size"])
        self.linear2 = Linear(in_features=config.model_DT_config["hidden_size"], out_features=config.model_DT_config["output_size"])
        
    def forward(self, x, hidden=None):
        lstm_out, (hidden, cell) = self.LSTM(x, hidden)  
        linear_out = F.tanh(self.linear1(lstm_out[:,-1,:]))
        linear_out = self.linear2(linear_out)
        return linear_out, (hidden, cell)
