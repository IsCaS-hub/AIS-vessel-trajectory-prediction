# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:51:10 2024

@author: jings
"""

import pandas as pd
import os
from datetime import timedelta
import numpy as np
from scipy.interpolate import CubicSpline

data_folder_path = 'D:/JINGwork/software_institution/Boat_trajctory_destination/Jing_new/AIS-1-clean'
interpolated_file_path = 'D:/JINGwork/software_institution/Boat_trajctory_destination/Jing_new/AIS-points_interpolation/interpolated_trajectory.csv'

all_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.csv')]
all_data = [pd.read_csv(file, low_memory=False) for file in all_files]  

df = pd.concat(all_data, ignore_index=True)
df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

grouped = df.sort_values('BaseDateTime').groupby('MMSI')
interp_df_list = []

for mmsi, group in grouped:
    group = group.sort_values(by='BaseDateTime')

    for i in range(1, len(group)):
        time_diff = (group.iloc[i]['BaseDateTime'] - group.iloc[i - 1]['BaseDateTime']).total_seconds() / 60
        if time_diff < 1:
            interp_df_list.append(group.iloc[i - 1].to_dict())
        elif 1 <= time_diff <= 30:
            start_time = group.iloc[i - 1]['BaseDateTime']
            end_time = group.iloc[i]['BaseDateTime']
            time_range = pd.date_range(start=start_time, end=end_time, freq='T')

            start_timestamp = start_time.timestamp()
            end_timestamp = end_time.timestamp()

            lat_spline = CubicSpline([start_timestamp, end_timestamp], [group.iloc[i - 1]['LAT'], group.iloc[i]['LAT']])
            lon_spline = CubicSpline([start_timestamp, end_timestamp], [group.iloc[i - 1]['LON'], group.iloc[i]['LON']])

            interp_lat = lat_spline(time_range.map(pd.Timestamp.timestamp))
            interp_lon = lon_spline(time_range.map(pd.Timestamp.timestamp))

            for j, t in enumerate(time_range):
                interp_df_list.append({
                    'MMSI': mmsi,
                    'BaseDateTime': t,
                    'LAT': interp_lat[j],
                    'LON': interp_lon[j],
                    'SOG': np.interp(t.timestamp(), [start_timestamp, end_timestamp], [group.iloc[i - 1]['SOG'], group.iloc[i]['SOG']]),
                    'COG': np.interp(t.timestamp(), [start_timestamp, end_timestamp], [group.iloc[i - 1]['COG'], group.iloc[i]['COG']]),
                    'Length': group.iloc[i - 1]['Length'],
                    'Width': group.iloc[i - 1]['Width']
                })

        else:
            interp_df_list.append(group.iloc[i - 1].to_dict())

    interp_df_list.append(group.iloc[-1].to_dict())
interp_df = pd.DataFrame(interp_df_list)
interp_df.to_csv(interpolated_file_path, index=False)
