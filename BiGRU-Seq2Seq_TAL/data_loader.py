
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle

scaler_dest = StandardScaler()
scaler_future = StandardScaler()

def load_csv_data(folder_path):

    train_inputs_dest, train_dest_outputs, train_inputs_future, train_future_outputs = [], [], [], []
    val_inputs_dest, val_dest_outputs, val_inputs_future, val_future_outputs = [], [], [], []  

    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    np.random.shuffle(all_files)  

    split_idx = int(len(all_files) * 0.7)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    for csv_file in train_files:
        try:
            inputs_dest, dest_outputs, inputs_future, future_outputs = process_csv(csv_file)
            train_inputs_dest.append(inputs_dest)
            train_dest_outputs.append(dest_outputs)
            train_inputs_future.append(inputs_future)
            train_future_outputs.append(future_outputs)
        except ValueError as e:
            print(f"Error processing {csv_file}: {e}")

    for csv_file in val_files:
        try:
            inputs_dest, dest_outputs, inputs_future, future_outputs = process_csv(csv_file)
            val_inputs_dest.append(inputs_dest)
            val_dest_outputs.append(dest_outputs)
            val_inputs_future.append(inputs_future)
            val_future_outputs.append(future_outputs)
        except ValueError as e:
            print(f"Error processing {csv_file}: {e}")

    train_inputs_dest = np.array(train_inputs_dest)  
    train_dest_outputs = np.array(train_dest_outputs)  
    train_inputs_future = np.array(train_inputs_future)  
    train_future_outputs = np.array(train_future_outputs)  

    val_inputs_dest = np.array(val_inputs_dest)  
    val_dest_outputs = np.array(val_dest_outputs)  
    val_inputs_future = np.array(val_inputs_future) 
    val_future_outputs = np.array(val_future_outputs) 

    train_inputs_dest = scaler_dest.fit_transform(train_inputs_dest.reshape(-1, 4)).reshape(-1, 21, 4)
    train_inputs_future = scaler_future.fit_transform(train_inputs_future.reshape(-1, 4)).reshape(-1, 21, 4)

    train_dest_outputs = scaler_dest.fit_transform(train_dest_outputs.reshape(-1, 4)).reshape(-1, 21, 4)
    train_future_outputs = scaler_future.fit_transform(train_future_outputs.reshape(-1, 4)).reshape(-1, 21, 4)

    val_inputs_dest = scaler_dest.transform(val_inputs_dest.reshape(-1, 4)).reshape(-1, 21, 4)
    val_inputs_future = scaler_future.transform(val_inputs_future.reshape(-1, 4)).reshape(-1, 21, 4)
    
    val_dest_outputs = scaler_dest.transform(val_dest_outputs.reshape(-1, 4)).reshape(-1, 21, 4)
    val_future_outputs = scaler_future.transform(val_future_outputs.reshape(-1, 4)).reshape(-1, 21, 4)

    return (train_inputs_dest, train_dest_outputs, train_inputs_future, train_future_outputs,
            val_inputs_dest, val_dest_outputs, val_inputs_future, val_future_outputs)


def process_csv(csv_file):
    df = pd.read_csv(csv_file)
    inputs_dest = df[['LAT', 'LON', 'SOG', 'COG']].iloc[:].values 
    dest_output = df[['LAT', 'LON', 'SOG', 'COG']].iloc[:].values
    inputs_future = df[['LAT', 'LON', 'SOG', 'COG']].iloc[:].values 
    future_output = df[['LAT', 'LON', 'SOG', 'COG']].iloc[:].values 

    return inputs_dest, dest_output, inputs_future, future_output

def save_scalers():
    with open('scaler_dest.pkl', 'wb') as f:
        pickle.dump(scaler_dest, f)
    with open('scaler_future.pkl', 'wb') as f:
        pickle.dump(scaler_future, f)

def load_data(folder_path, batch_size=32):

    train_inputs_dest, train_dest_outputs, train_inputs_future, train_future_outputs, \
    val_inputs_dest, val_dest_outputs, val_inputs_future, val_future_outputs = load_csv_data(folder_path)


    train_inputs_dest_tensor = torch.tensor(train_inputs_dest, dtype=torch.float32)
    train_dest_outputs_tensor = torch.tensor(train_dest_outputs, dtype=torch.float32)
    train_inputs_future_tensor = torch.tensor(train_inputs_future, dtype=torch.float32)
    train_future_outputs_tensor = torch.tensor(train_future_outputs, dtype=torch.float32)

    val_inputs_dest_tensor = torch.tensor(val_inputs_dest, dtype=torch.float32)
    val_dest_outputs_tensor = torch.tensor(val_dest_outputs, dtype=torch.float32)
    val_inputs_future_tensor = torch.tensor(val_inputs_future, dtype=torch.float32)
    val_future_outputs_tensor = torch.tensor(val_future_outputs, dtype=torch.float32)

    print(f"train_inputs_dest_tensor shape: {train_inputs_dest_tensor.shape}")
    print(f"train_dest_outputs_tensor shape: {train_dest_outputs_tensor.shape}")
    print(f"train_inputs_future_tensor shape: {train_inputs_future_tensor.shape}")
    print(f"train_future_outputs_tensor shape: {train_future_outputs_tensor.shape}")
    
    print(f"val_inputs_dest_tensor shape: {val_inputs_dest_tensor.shape}")
    print(f"val_dest_outputs_tensor shape: {val_dest_outputs_tensor.shape}")
    print(f"val_inputs_future_tensor shape: {val_inputs_future_tensor.shape}")
    print(f"val_future_outputs_tensor shape: {val_future_outputs_tensor.shape}")


    train_dataset_dest = TensorDataset(train_inputs_dest_tensor, train_dest_outputs_tensor)
    train_dataset_future = TensorDataset(train_inputs_future_tensor, train_future_outputs_tensor)

    val_dataset_dest = TensorDataset(val_inputs_dest_tensor, val_dest_outputs_tensor)
    val_dataset_future = TensorDataset(val_inputs_future_tensor, val_future_outputs_tensor)


    train_loader_dest = DataLoader(train_dataset_dest, batch_size=batch_size, shuffle=True)
    train_loader_future = DataLoader(train_dataset_future, batch_size=batch_size, shuffle=True)

    val_loader_dest = DataLoader(val_dataset_dest, batch_size=batch_size, shuffle=False)
    val_loader_future = DataLoader(val_dataset_future, batch_size=batch_size, shuffle=False)
    
    save_scalers()

    return (train_loader_dest, train_loader_future, val_loader_dest, val_loader_future)