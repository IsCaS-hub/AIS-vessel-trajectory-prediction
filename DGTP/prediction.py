
from sklearn.metrics import mean_squared_error
import torch 
import pandas as pd
import os
import matplotlib.pyplot as plt
from model import Net_Encoder, Net_Encoder2, Net_Decoder_T, Net_Decoder_D
import config
from config import num_epochs, batch_size, device, model_E_config, model_DT_config, model_DD_config, loss_fn, save_dir, test_data_folder 
import numpy as np
import joblib  


def load_model(config):  

    model_E  = Net_Encoder(config).to(device)
    model_E2  = Net_Encoder2(config).to(device)
    model_DT = Net_Decoder_T(config).to(device)
    model_DD = Net_Decoder_D(config).to(device)
    model_E.load_state_dict(torch.load(config.save_dir + 'model_E.pth'))  
    model_E2.load_state_dict(torch.load(config.save_dir + 'model_E2.pth')) 
    model_DT.load_state_dict(torch.load(config.save_dir + 'model_DT.pth'))  
    model_DD.load_state_dict(torch.load(config.save_dir + 'model_DD.pth')) 
    
    model_E.eval()
    model_E2.eval()
    model_DT.eval()
    model_DD.eval()
    return model_E, model_E2, model_DT, model_DD

def load_single_test_data(csv_file):
    df = pd.read_csv(csv_file)
    data = df[['LAT', 'LON', 'SOG', 'COG']].values
    return data


def load_scalers():
    scaler_dest = joblib.load('./scaler_dest.pkl')  
    scaler_future = joblib.load('./scaler_future.pkl')  
    return scaler_dest, scaler_future

def predict_for_single_csv(csv_file, model_E, model_E2, model_DT, model_DD, scaler_dest, scaler_future):
    data = load_single_test_data(csv_file)
    
    if data.shape[0] < 10:  
        print(f"Skipping file {csv_file}: insufficient data (less than 10 time steps).")
        return None
    
    test_input = torch.tensor(data[:], dtype=torch.float32).to(device).unsqueeze(0)  
    data_Y_d = torch.tensor(data[:], dtype=torch.float32).to(device).unsqueeze(0)  
    data_Y_t = torch.tensor(data[:], dtype=torch.float32).to(device).unsqueeze(0)  
        

    test_input = scaler_dest.transform(test_input.cpu().numpy().reshape(-1, 4)).reshape(1, 21, 4)
    test_input = torch.tensor(test_input, dtype=torch.float32).to(device)

    criterion_t = torch.nn.HuberLoss()          
    criterion_d = torch.nn.HuberLoss() 
    hidden_predict_E = None
    hidden_predict_E2 = None
    hidden_predict_DT = None
    hidden_predict_DD = None
    
    future_pred = []
    dest_pred = []

    with torch.no_grad():
        data_input = test_input[:,:10,:]
        for num in range(10):
            pred_E, hidden_predict_E = model_E(data_input, hidden_predict_E)
            data_DD = pred_E
            pred_DD, hidden_predict_DD = model_DD(data_DD, hidden_predict_DD)            
            
            hidden_predict_E = None
            hidden_predict_E2 = None
            hidden_predict_DT = None
            hidden_predict_DD = None
            
            data_ED = torch.unsqueeze(pred_DD, dim=1) #[BS, 1 , 4]
            pred_ED, hidden_predict_E2 = model_E2(data_ED, hidden_predict_E2) #[BS, 1 , 128]                
            data_DT = torch.cat((data_DD, pred_ED), dim=1)    
            pred_DT, hidden_predict_DT = model_DT(data_DT, hidden_predict_DT)        
            
            hidden_predict_E =  None
            hidden_predict_DT = None
    
            future_pred_point = pred_DT.squeeze(0).cpu().numpy().tolist()
            dest_pred_point = pred_DD.squeeze(0).cpu().numpy().tolist()
            
            shifted_data_input = torch.zeros_like(data_input)
            shifted_data_input[:, :-1, :] = data_input[:, 1:, :] 
            shifted_data_input[:,-1,:] = pred_DT
            data_input = shifted_data_input
                        
            future_pred_point = scaler_future.inverse_transform(np.array(future_pred_point)).tolist()
            future_pred.append(future_pred_point)
            dest_pred_point = scaler_future.inverse_transform(np.array(dest_pred_point)).tolist()
            dest_pred.append(dest_pred_point)

        future_pred = np.array(future_pred)

    return {
        'file': csv_file,
        'input_trajectory': data[:],  
        'groundtruth_destination': data[:],  
        'future_trajectory_groundtruth': data[:], 
        'destination_pred': dest_pred,
        'future_trajectory_pred': future_pred,
    }

def plot_prediction(prediction, output_folder='prediction_plot_losschange'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = os.path.basename(prediction['file'])
    save_path = os.path.join(output_folder, f'plot_{file_name.replace(".csv", ".png")}')

    input_trajectory = prediction['input_trajectory']
    groundtruth_destination = prediction['groundtruth_destination']
    future_trajectory_groundtruth = prediction['future_trajectory_groundtruth']
    destination_pred = prediction['destination_pred']
    future_trajectory_pred = prediction['future_trajectory_pred']
    
    
    input_lat = input_trajectory[:10, 1]
    input_lon = input_trajectory[:10, 0]
    
    
    future_lat = future_trajectory_groundtruth[-10:,1]
    future_lon = future_trajectory_groundtruth[-10:,0]
    
    
    destination_lat = groundtruth_destination[10,1]
    destination_lon = groundtruth_destination[10,0]
    
    
    future_pred_lat = np.array(future_trajectory_pred)[:, 1]
    future_pred_lon = np.array(future_trajectory_pred)[:, 0]
    
    pred_destination_lat = np.mean(np.array(destination_pred)[:,1])
    pred_destination_lon = np.mean(np.array(destination_pred)[:,0])

    plt.figure(figsize=(8, 6))


    plt.scatter(input_lat, input_lon, color='green', marker='o', s=70, label='Input Trajectory')
    plt.plot(input_lat, input_lon, color='green')
    
    plt.scatter(destination_lat, destination_lon, color='blue', marker='^', s=70, label='Destination Groundtruth')
    
    plt.scatter(pred_destination_lat, pred_destination_lon, color='red', marker='^', s=70, label='Predicted Destination')


    plt.scatter(future_lat, future_lon, color='blue', marker='o', s=70, label='Future Groundtruth Trajectory')
    plt.plot(future_lat, future_lon, color='blue')

    plt.scatter(future_pred_lat, future_pred_lon, color='red', marker='o', s=70, label='Predicted Future Trajectory')
    plt.plot(future_pred_lat, future_pred_lon, color='red')

    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def predict_and_plot():
    model_E, model_E2, model_DT, model_DD = load_model(config)

    scaler_dest, scaler_future = load_scalers()
    csv_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        prediction = predict_for_single_csv(csv_file, model_E, model_E2, model_DT, model_DD, scaler_dest, scaler_future)        
        
        if prediction is not None:
            plot_prediction(prediction)


if __name__ == "__main__":
    predict_and_plot()