

import torch
import torch.optim as optim
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt
import config
from config import num_epochs, batch_size, device, model_E_config, model_DT_config, model_DD_config, loss_fn, save_dir
import numpy as np
import model
import torch.nn.functional as F
import math
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def initialize_weights(model):
    for layer in model.children():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LSTM) or isinstance(layer, nn.GRU):
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

def calculate_length_loss(pred_trajectory, true_trajectory):
    pred_length = torch.norm(pred_trajectory[-1, :2] - pred_trajectory[-2, :2])
    true_length = torch.norm(true_trajectory[-1, :2] - true_trajectory[-2, :2])
    return torch.abs(pred_length - true_length)

def calculate_direction_loss(pred_trajectory, true_trajectory):
    pred_directions = pred_trajectory[1:] - pred_trajectory[:-1]
    true_directions = true_trajectory[1:] - true_trajectory[:-1]
    pred_angles = torch.atan2(pred_directions[:, 1], pred_directions[:, 0])
    true_angles = torch.atan2(true_directions[:, 1], true_directions[:, 0])
    return torch.mean(torch.abs(pred_angles - true_angles))

def calculate_similarity_loss(pred_trajectory, true_trajectory):
    distance, _ = fastdtw(pred_trajectory, true_trajectory, dist=euclidean)
    return distance

def train_model(train_loader_dest, train_loader_future, val_loader_dest, val_loader_future):

    Net_Encoder = model.Net_Encoder
    Net_Encoder2 = model.Net_Encoder2
    Net_Decoder_T = model.Net_Decoder_T
    Net_Decoder_D = model.Net_Decoder_D

    model_E = Net_Encoder(config).to(device)
    model_E2 = Net_Encoder2(config).to(device)
    model_DT = Net_Decoder_T(config).to(device)
    model_DD = Net_Decoder_D(config).to(device)
    
    initialize_weights(model_E)
    initialize_weights(model_E2)
    initialize_weights(model_DT)
    initialize_weights(model_DD)

    criterion_t = nn.MSELoss()          
    criterion_d = nn.MSELoss() 

    optimizer_E = torch.optim.Adam(model_E.parameters(), lr=config.model_E_config["learning_rate"])
    optimizer_E2 = torch.optim.Adam(model_E2.parameters(), lr=config.model_E2_config["learning_rate"])
    optimizer_DT = torch.optim.Adam(model_DT.parameters(), lr=config.model_DT_config["learning_rate"])
    optimizer_DD = torch.optim.Adam(model_DD.parameters(), lr=config.model_DD_config["learning_rate"])

    valid_loss_min = float("inf")  
    bad_epoch = 0
    global_step = 0
    train_loss_all_t = []
    train_loss_all_d = []
    train_loss_all = []
    valid_loss_all_t = []
    valid_loss_all_d = []
    valid_loss_all = []

    writer_dest = pd.ExcelWriter('model_E_loss.xlsx', engine='xlsxwriter')
    writer_dest = pd.ExcelWriter('model_DT_loss.xlsx', engine='xlsxwriter')
    writer_future = pd.ExcelWriter('model_DD_loss.xlsx', engine='xlsxwriter')

    loss_T = {'Train Loss': [], 'Validation Loss': []}
    loss_D = {'Train Loss': [], 'Validation Loss': []}

    for epoch in range(num_epochs):
        model_E.train()
        model_E2.train()
        model_DT.train()
        model_DD.train()
        hidden_train_E = None
        hidden_train_E2 = None
        hidden_train_DT = None
        hidden_train_DD = None
        train_loss_t = []
        train_loss_d = []
        train_loss = []       

        for (train_inputs, train_dest_outputs), (train_inputs_future, train_future_outputs) in zip(train_loader_dest, train_loader_future):
            train_inputs = train_inputs.to(device)
            train_outputs = train_dest_outputs.to(device)
            train_inputs_future = train_inputs_future.to(device)
            train_future_outputs = train_future_outputs.to(device)

            for num in range(10): 
                  
                for k in range(2): 
                    optimizer_E.zero_grad()
                    optimizer_DD.zero_grad()
                    pred_E, hidden_train_E = model_E(train_inputs[:,num:num+10,:], hidden_train_E)
                    data_DD = pred_E #[BS, 10, 128]
                    pred_DD, hidden_train_DD = model_DD(data_DD, hidden_train_DD)
                    
                    hidden_train_E = None
                    hidden_train_E2 = None
                    hidden_train_DT = None
                    hidden_train_DD = None
                                    
                    train_dest_outputs_s = torch.squeeze(train_outputs[:,10,:], dim=1)
                    loss_d = criterion_t(pred_DD, train_dest_outputs_s)
                    
                    loss_d.backward()  
                    optimizer_E.step()
                    optimizer_DD.step() 
                
                for kk in range(2):
                    optimizer_E2.zero_grad()
                    optimizer_DT.zero_grad()
                    data_ED = torch.unsqueeze(pred_DD, dim=1) #[BS, 1 , 4]
                    pred_ED, hidden_train_E2 = model_E2(data_ED, hidden_train_E2) #[BS, 1 , 128]                
                    data_DT = torch.cat((data_DD, pred_ED), dim=1)     # #[BS, 11 , 128]            
                    pred_DT, hidden_train_DT = model_DT(data_DT.detach(), hidden_train_DT)
    
                    hidden_train_E = None
                    hidden_train_E2 = None
                    hidden_train_DT = None
                    hidden_train_DD = None
    
                    
                    train_future_outputs_s = torch.squeeze(train_outputs[:,num+11,:], dim=1)
                    loss_t = criterion_t(pred_DT[:,:], train_future_outputs_s[:,:])                
                    
                    total_loss = loss_t 
                    
                    total_loss.backward()  
                    optimizer_E2.step()
                    optimizer_DT.step()                
            
                    train_loss_t.append(total_loss.item())
                    global_step += 1
            
        train_loss_t_cur = np.mean(train_loss_t)
        train_loss_all_t.append(train_loss_t_cur)               

        model_E.eval()  
        model_E2.eval() 
        model_DT.eval()
        model_DD.eval() 
        hidden_valid_E = None
        hidden_valid_E2 = None
        hidden_valid_DT = None
        hidden_valid_DD = None
        valid_loss_t = []
        valid_loss_d = []
        valid_loss = []
        with torch.no_grad():
            for (val_inputs, val_dest_outputs), (val_inputs_future, val_future_outputs) in zip(val_loader_dest, val_loader_future):
                val_inputs = val_inputs.to(device)
                val_outputs = val_dest_outputs.to(device)
                val_inputs_future = val_inputs_future.to(device)
                val_future_outputs = val_future_outputs.to(device)
                
                for num in range(2):
                    pred_E, hidden_valid_E = model_E(val_inputs[:,num:num+10,:], hidden_valid_E)
                    data_DD = pred_E
                    pred_DD, hidden_valid_DD = model_DD(data_DD, hidden_valid_DD)
                    
                    hidden_valid_E = None
                    hidden_valid_E2 = None
                    hidden_valid_DT = None
                    hidden_valid_DD = None
                    
                    data_ED = torch.unsqueeze(pred_DD, dim=1) #[BS, 1 , 4]
                    pred_ED, hidden_valid_E2 = model_E2(data_ED, hidden_valid_E2) #[BS, 1 , 128]                
                    data_DT = torch.cat((data_DD, pred_ED), dim=1)    
                    pred_DT, hidden_valid_DT = model_DT(data_DT, hidden_valid_DT)
    
                    hidden_valid_E = None
                    hidden_valid_E2 = None
                    hidden_valid_DT = None
                    hidden_valid_DD = None              
    
                    val_dest_outputs_s = torch.squeeze(val_outputs[:,10,:], dim=1)
                    loss_d = criterion_t(pred_DD, val_dest_outputs_s)
                
                    val_future_outputs_s = torch.squeeze(val_outputs[:,num+11,:], dim=1)
                    loss_t = criterion_t(pred_DT[:,:], val_future_outputs_s[:,:])                    
                    
                    dest = (val_outputs[:,10,:]).unsqueeze(1)

                    total_loss = loss_t
                    
                    valid_loss_t.append(total_loss.item())
                
            valid_loss_t_cur = np.mean(valid_loss_t)
            valid_loss_all_t.append(valid_loss_t_cur)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss Future: {train_loss_t_cur:.4f}, '
              f'Val Loss Future: {valid_loss_t_cur:.4f}')
                
  
        pd.DataFrame({'Train Loss': [train_loss_t_cur], 'Validation Loss': [valid_loss_t_cur]}).to_excel(writer_future, sheet_name=f'Epoch_{epoch+1}', index=False)

        # loss_D['Train Loss'].append(train_loss_d_cur)
        # loss_D['Validation Loss'].append(valid_loss_d_cur)
        loss_T['Train Loss'].append(train_loss_t_cur)
        loss_T['Validation Loss'].append(valid_loss_t_cur)
  
        if valid_loss_t_cur < valid_loss_min:
            valid_loss_min = valid_loss_t_cur
            bad_epoch = 0
            torch.save(model_E.state_dict(), os.path.join(save_dir, 'model_E.pth'))
            torch.save(model_E2.state_dict(), os.path.join(save_dir, 'model_E2.pth'))
            torch.save(model_DT.state_dict(), os.path.join(save_dir, 'model_DT.pth'))
            torch.save(model_DD.state_dict(), os.path.join(save_dir, 'model_DD.pth'))
            print(f"Validation loss for future model decreased. Saving both 'model_dest.pth' and 'model_future.pth'.")

    writer_dest.close()
    writer_future.close()

    plot_loss_curves(loss_T, 'Future Prediction Model Loss', 'model_future_loss_plot.png')

def plot_loss_curves(loss_data, title, save_path):
    plt.figure()
    plt.plot(loss_data['Train Loss'], label='Train Loss')
    plt.plot(loss_data['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

    

    