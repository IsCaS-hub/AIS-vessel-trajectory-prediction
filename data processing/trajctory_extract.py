import pandas as pd
import os

interpolated_file_path = 'D:/JINGwork/software_institution/Boat_trajctory_destination/Jing_new/AIS-points_interpolation/interpolated_trajectory.csv'
output_base_path = 'D:/JINGwork/software_institution/Boat_trajctory_destination/Jing_new/trajctory_extraction'

if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

df = pd.read_csv(interpolated_file_path)

df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
grouped = df.sort_values('BaseDateTime').groupby('MMSI')

for mmsi, group in grouped:
    start_index = 0
    subtrack_number = 0
    for i in range(1, len(group)):
        time_diff = (group.iloc[i]['BaseDateTime'] - group.iloc[i - 1]['BaseDateTime']).total_seconds() / 60
        
        if time_diff > 30:
            sub_df = group.iloc[start_index:i]
            subtrack_filename = f'{mmsi}_subtrack_{subtrack_number}.csv'
            sub_df.to_csv(os.path.join(output_base_path, subtrack_filename), index=False)
            subtrack_number += 1
            start_index = i

    sub_df = group.iloc[start_index:]
    subtrack_filename = f'{mmsi}_subtrack_{subtrack_number}.csv'
    sub_df.to_csv(os.path.join(output_base_path, subtrack_filename), index=False)


