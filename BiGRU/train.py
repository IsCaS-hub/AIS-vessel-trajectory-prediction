
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

def train_model(train_loader_dest, train_loader_future, val_loader_dest, val_loader_future):
    Net_Decoder_T = model.Net_Decoder_T
    model_DT = Net_Decoder_T(config).to(device)
    
    initialize_weights(model_DT)
    criterion_t = nn.MSELoss()          
    criterion_d = nn.MSELoss() 

    optimizer_DT = torch.optim.Adam(model_DT.parameters(), lr=config.model_DT_config["learning_rate"])

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
        model_DT.train()
        hidden_train_DT = None
        train_loss_t = []
        train_loss_d = []
        train_loss = []       

        for (train_inputs, train_dest_outputs), (train_inputs_future, train_future_outputs) in zip(train_loader_dest, train_loader_future):
            train_inputs = train_inputs.to(device)
            train_outputs = train_dest_outputs.to(device)
            train_inputs_future = train_inputs_future.to(device)
            train_future_outputs = train_future_outputs.to(device)

            for num in range(2): 
                optimizer_DT.zero_grad()            
                pred_DT, hidden_train_DT = model_DT(train_inputs[:,num:num+10,:], hidden_train_DT)
                hidden_train_DT = None
           
                train_future_outputs_s = torch.squeeze(train_outputs[:,num+11,:], dim=1)
                loss_t = criterion_t(pred_DT, train_future_outputs_s)            
                
                total_loss = loss_t 

                total_loss.backward()  
                optimizer_DT.step()                
            
            train_loss_t.append(total_loss.item())
            global_step += 1
            
        train_loss_t_cur = np.mean(train_loss_t)
        train_loss_all_t.append(train_loss_t_cur)               

        model_DT.eval()
        hidden_valid_DT = None
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
                    pred_DT, hidden_valid_DT = model_DT(val_inputs[:,num:num+10,:], hidden_valid_DT)                
    
                    hidden_valid_DT = None              
    
                    val_future_outputs_s = torch.squeeze(val_outputs[:,num+11,:], dim=1)
                    loss_t = criterion_t(pred_DT, val_future_outputs_s)            
                    
                    valid_loss_t.append(total_loss.item())
                
            valid_loss_t_cur = np.mean(valid_loss_t)
            valid_loss_all_t.append(valid_loss_t_cur)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss Future: {train_loss_t_cur:.4f}, '
              f'Val Loss Future: {valid_loss_t_cur:.4f}')
                
        pd.DataFrame({'Train Loss': [train_loss_t_cur], 'Validation Loss': [valid_loss_t_cur]}).to_excel(writer_future, sheet_name=f'Epoch_{epoch+1}', index=False)

        loss_T['Train Loss'].append(train_loss_t_cur)
        loss_T['Validation Loss'].append(valid_loss_t_cur)

        
        if valid_loss_t_cur < valid_loss_min:
            valid_loss_min = valid_loss_t_cur
            bad_epoch = 0
            torch.save(model_DT.state_dict(), os.path.join(save_dir, 'model_DT.pth'))
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

    

    