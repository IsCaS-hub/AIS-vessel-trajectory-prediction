# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:12:52 2024

@author: Lenovo
"""

import torch
from data_loader import load_data  
from train import train_model
from config import num_epochs, batch_size, device, model_E_config, model_DT_config, model_DD_config, loss_fn, save_dir
import model

def main():
    
    folder_path = 'D:/JINGwork/software_institution/Boat_trajctory_destination/Jing_new/Compared/data/'  
    train_loader_dest, train_loader_future, val_loader_dest, val_loader_future = load_data(folder_path, batch_size=batch_size)

    train_model(train_loader_dest, train_loader_future, val_loader_dest, val_loader_future)

if __name__ == "__main__":
    main()