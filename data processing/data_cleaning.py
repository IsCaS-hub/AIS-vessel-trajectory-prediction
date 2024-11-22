# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 08:59:24 2024

@author: jings
"""

import os
import pandas as pd


input_folder = 'D:/JINGwork/software_institution/Boat_trajctory_destination/Jing_new/AIS-1'
output_folder = 'D:/JINGwork/software_institution/Boat_trajctory_destination/Jing_new/AIS-1-clean'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)

        df = pd.read_csv(file_path)
        df.sort_values(by=['MMSI', 'BaseDateTime'], inplace=True)
        df = df[df['MMSI'].apply(lambda x: len(str(x)) == 9)]

        df = df[(df['Length'] >= 3) & (df['Width'] >= 2)]

        df = df[(df['LON'] >= -180.0) & (df['LON'] <= 180.0)]
        df = df[(df['LAT'] >= -90.0) & (df['LAT'] <= 90.0)]
        df = df[(df['SOG'] >= 0) & (df['SOG'] <= 51.2)]
        df = df[(df['COG'] >= -204.7) & (df['COG'] <= 204.8)]

        new_filename = 'clean_' + filename
        output_path = os.path.join(output_folder, new_filename)
        df.to_csv(output_path, index=False)
