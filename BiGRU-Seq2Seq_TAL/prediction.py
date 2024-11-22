
import torch 
import pandas as pd
import os
import matplotlib.pyplot as plt
from model import Net_Encoder, Net_Decoder_T
import config
from config import num_epochs, batch_size, device, model_E_config, model_DT_config, loss_fn, save_dir, test_data_folder 
import numpy as np
import joblib 


def load_model(config):  
    model_E  = Net_Encoder(config).to(device)
    model_DT = Net_Decoder_T(config).to(device)
    model_E.load_state_dict(torch.load(config.save_dir + 'model_E.pth'))  
    model_DT.load_state_dict(torch.load(config.save_dir + 'model_DT.pth'))  
    
    model_E.eval()
    model_DT.eval()
    return model_E, model_DT

def load_single_test_data(csv_file):
    df = pd.read_csv(csv_file)
    data = df[['LAT', 'LON', 'SOG', 'COG']].values
    return data

def load_scalers():
    scaler_dest = joblib.load('./scaler_dest.pkl')  
    scaler_future = joblib.load('./scaler_future.pkl')  
    return scaler_dest, scaler_future

def predict_for_single_csv(csv_file, model_E, model_DT, scaler_dest, scaler_future):
    data = load_single_test_data(csv_file)
    
    if data.shape[0] < 10:
        print(f"Skipping file {csv_file}: insufficient data (less than 10 time steps).")
        return None
    
    test_input = torch.tensor(data[:], dtype=torch.float32).to(device).unsqueeze(0)  
    data_Y_d = torch.tensor(data[:], dtype=torch.float32).to(device).unsqueeze(0)  
    data_Y_t = torch.tensor(data[:], dtype=torch.float32).to(device).unsqueeze(0)  
        
    test_input = scaler_dest.transform(test_input.cpu().numpy().reshape(-1, 4)).reshape(1, 21, 4)
    test_input = torch.tensor(test_input, dtype=torch.float32).to(device)

    criterion_t = torch.nn.MSELoss()     
    criterion_d = torch.nn.MSELoss()   
    hidden_predict_E = None
    hidden_predict_DT = None
    future_pred = []
    with torch.no_grad():
        data_input = test_input[:,:10,:]
        for num in range(10):
            pred_E, hidden_predict_E = model_E(data_input, hidden_predict_E)       # 编码器编码
            data_DT = pred_E
            pred_DT, hidden_predict_DT = model_DT(data_DT, hidden_predict_DT)   # 目的地解码器解码
                
            loss_t = criterion_t(pred_DT, data_Y_t[:,num+11,:])

            hidden_predict_E =  None
            hidden_predict_DT = None

            future_pred_point = pred_DT.squeeze(0).cpu().numpy().tolist()
            
            shifted_data_input = torch.zeros_like(data_input)
            shifted_data_input[:, :-1, :] = data_input[:, 1:, :] #将data_input前移
            shifted_data_input[:,-1,:] = pred_DT
            data_input = shifted_data_input
                        
            future_pred_point = scaler_future.inverse_transform(np.array(future_pred_point)).tolist()
            future_pred.append(future_pred_point)

        future_pred = np.array(future_pred)

    return {
        'file': csv_file,
        'input_trajectory': data[:],  
        'groundtruth_destination': data[:],  
        'future_trajectory_groundtruth': data[:], 
        'future_trajectory_pred': future_pred
    }

# Plot predictions
def plot_prediction(prediction, output_folder='prediction_plot'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = os.path.basename(prediction['file'])
    save_path = os.path.join(output_folder, f'plot_{file_name.replace(".csv", ".png")}')

    input_trajectory = prediction['input_trajectory']
    groundtruth_destination = prediction['groundtruth_destination']
    future_trajectory_pred = prediction['future_trajectory_pred']
    

    input_lat = input_trajectory[:10, 1]
    input_lon = input_trajectory[:10, 0]
    

    destination_lat = groundtruth_destination[-10:, 1]
    destination_lon = groundtruth_destination[-10:, 0]
    

    future_pred_lat = np.array(future_trajectory_pred)[:, 1]
    future_pred_lon = np.array(future_trajectory_pred)[:, 0]

    input_data = pd.DataFrame({
        'lat': input_lat,
        'lon': input_lon
    })
    input_data.to_csv(os.path.join(output_folder, f'input_trajectory_{file_name}'), index=False)
    destination_data = pd.DataFrame({
        'lat': destination_lat,
        'lon': destination_lon
    })
    destination_data.to_csv(os.path.join(output_folder, f'destination_trajectory_{file_name}'), index=False)
    future_pred_data = pd.DataFrame({
        'lat': future_pred_lat,
        'lon': future_pred_lon
    })
    future_pred_data.to_csv(os.path.join(output_folder, f'predicted_trajectory_{file_name}'), index=False)

    plt.figure(figsize=(8, 6))


    plt.scatter(input_lat, input_lon, color='green', marker='o', s=70, label='Input Trajectory')
    plt.plot(input_lat, input_lon, color='green')

    plt.scatter(destination_lat, destination_lon, color='blue', marker='o', s=70, label='Future Groundtruth Trajectory')
    plt.plot(destination_lat, destination_lon, color='blue')

    plt.scatter(future_pred_lat, future_pred_lon, color='red', marker='o', s=70, label='Predicted Future Trajectory')
    plt.plot(future_pred_lat, future_pred_lon, color='red')

    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.savefig(save_path)
    plt.close()



def predict_and_plot():
    model_E, model_DT = load_model(config)

    scaler_dest, scaler_future = load_scalers()

    csv_files = [os.path.join(test_data_folder, f) for f in os.listdir(test_data_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        prediction = predict_for_single_csv(csv_file, model_E, model_DT, scaler_dest, scaler_future)
        if prediction is not None:
            plot_prediction(prediction)
    

if __name__ == "__main__":
    predict_and_plot()
