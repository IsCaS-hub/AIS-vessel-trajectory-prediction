import torch 
import pandas as pd
import os
import matplotlib.pyplot as plt
from model import Net_Decoder_T
import config
from config import num_epochs, batch_size, device, model_E_config, model_DT_config, model_DD_config, loss_fn, save_dir, test_data_folder 
import numpy as np
import joblib

# Load model function
def load_model(config):  
    model_DT = Net_Decoder_T(config).to(device) 
    model_DT.load_state_dict(torch.load(config.save_dir + 'model_DT.pth'))  
    model_DT.eval()
    return model_DT

# Load single test data file
def load_single_test_data(csv_file):
    df = pd.read_csv(csv_file)
    data = df[['LAT', 'LON', 'SOG', 'COG']].values
    return data

# Load scaler
def load_scalers():
    scaler_dest = joblib.load('./scaler_dest.pkl')  
    scaler_future = joblib.load('./scaler_future.pkl')  
    return scaler_dest, scaler_future

# Prediction function for a single CSV file
def predict_for_single_csv(csv_file, model_DT, scaler_dest, scaler_future):
    data = load_single_test_data(csv_file)
    
    if data.shape[0] < 10:  # Ensure there are at least 10 time steps
        print(f"Skipping file {csv_file}: insufficient data (less than 10 time steps).")
        return None
    
    test_input = torch.tensor(data[:], dtype=torch.float32).to(device).unsqueeze(0) 
    data_Y_d = torch.tensor(data[:], dtype=torch.float32).to(device).unsqueeze(0)
    data_Y_t = torch.tensor(data[:], dtype=torch.float32).to(device).unsqueeze(0)
        
    test_input = scaler_dest.transform(test_input.cpu().numpy().reshape(-1, 4)).reshape(1, 21, 4)
    test_input = torch.tensor(test_input, dtype=torch.float32).to(device)

    criterion_t = torch.nn.HuberLoss()          
    criterion_d = torch.nn.HuberLoss() 
    hidden_predict_DT = None
    future_pred = []
    rmse_per_time_step = []  # 保存每个时间步的误差
    with torch.no_grad():
        data_input = test_input[:,:10,:]
        for num in range(10):
            pred_DT, hidden_predict_DT = model_DT(data_input, hidden_predict_DT)  
            loss_t = criterion_t(pred_DT, data_Y_t[:,num+11,:])
            hidden_predict_DT = None
    
            future_pred_point = pred_DT.squeeze(0).cpu().numpy().tolist()
            shifted_data_input = torch.zeros_like(data_input)
            shifted_data_input[:, :-1, :] = data_input[:, 1:, :] 
            shifted_data_input[:,-1,:] = pred_DT
            data_input = shifted_data_input
                        
            future_pred_point = scaler_future.inverse_transform(np.array(future_pred_point)).tolist()
            future_pred.append(future_pred_point)

            # 计算每个时间步的 RMSE
            rmse_t_lat = np.sqrt(np.mean((data_Y_t[:, num + 11, 1].cpu().numpy() - future_pred_point[1]) ** 2))
            rmse_t_lon = np.sqrt(np.mean((data_Y_t[:, num + 11, 0].cpu().numpy() - future_pred_point[0]) ** 2))
            rmse_per_time_step.append((rmse_t_lat, rmse_t_lon))

    future_pred = np.array(future_pred)
    return {
        'file': csv_file,
        'input_trajectory': data[:],
        'groundtruth_destination': data[:],
        'future_trajectory_groundtruth': data[:],
        'future_trajectory_pred': future_pred,
        'rmse_per_time_step': rmse_per_time_step
    }

# Main function to predict and calculate RMSE
def predict_and_calculate_rmse():
    model_DT = load_model(config)
    scaler_dest, scaler_future = load_scalers()
    csv_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith('.csv')]
    
    rmse_time_step_data = {i: {'lat_rmse': [], 'lon_rmse': []} for i in range(10)}

    for csv_file in csv_files:
        prediction = predict_for_single_csv(csv_file, model_DT, scaler_dest, scaler_future)
        if prediction is not None:
            rmse_per_time_step = prediction['rmse_per_time_step']
            for num, (lat_rmse, lon_rmse) in enumerate(rmse_per_time_step):
                rmse_time_step_data[num]['lat_rmse'].append(lat_rmse)
                rmse_time_step_data[num]['lon_rmse'].append(lon_rmse)
    
    # 计算每个时间步的均值和标准差并保存到 Excel
    rmse_summary = {
        'Time Step': [],
        'Latitude RMSE Mean': [],
        'Latitude RMSE Std': [],
        'Longitude RMSE Mean': [],
        'Longitude RMSE Std': []
    }
    for num in range(10):
        lat_rmse_values = rmse_time_step_data[num]['lat_rmse']
        lon_rmse_values = rmse_time_step_data[num]['lon_rmse']
        
        rmse_summary['Time Step'].append(num)
        rmse_summary['Latitude RMSE Mean'].append(np.mean(lat_rmse_values))
        rmse_summary['Latitude RMSE Std'].append(np.std(lat_rmse_values))
        rmse_summary['Longitude RMSE Mean'].append(np.mean(lon_rmse_values))
        rmse_summary['Longitude RMSE Std'].append(np.std(lon_rmse_values))
    
    rmse_df = pd.DataFrame(rmse_summary)
    rmse_df.to_excel('rmse_per_time_step.xlsx', index=False)
    print("RMSE data saved to rmse_per_time_step.xlsx")

if __name__ == "__main__":
    predict_and_calculate_rmse()
