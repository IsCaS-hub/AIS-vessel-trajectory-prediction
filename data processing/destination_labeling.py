# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:57:04 2024

@author: jings
"""

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.stats import norm

input_folder = 'E:/Jing_Project/Boat trajectory and destination prediction/Jing_new/merged_trajectories'
output_folder = 'E:/Jing_Project/Boat trajectory and destination prediction/Jing_new/trajectories_with_destination'
os.makedirs(output_folder, exist_ok=True)

input_folder2 = 'E:/Jing_Project/Boat trajectory and destination prediction/Jing_new/trajctory_extraction'
all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

all_files2 = [os.path.join(input_folder2, f) for f in os.listdir(input_folder2) if f.endswith('.csv')]

sampling_interval = 1

start_end_points = []
all_trajectories = []
all_sampled_trajectories_points = []

for file in all_files:
    df = pd.read_csv(file)
    if 'LAT' in df.columns and 'LON' in df.columns and 'SOG' in df.columns and 'COG' in df.columns:
        start_point = df.iloc[0][['LAT', 'LON', 'SOG', 'COG']].values
        end_point = df.iloc[-1][['LAT', 'LON', 'SOG', 'COG']].values
        start_end_points.append(np.append(start_point, [1]))  
        start_end_points.append(np.append(end_point, [0]))    
        
        sampled_points = df.iloc[1:-1:sampling_interval][['LAT', 'LON', 'SOG', 'COG']].values
        all_sampled_trajectories_points.append(np.vstack([start_point, sampled_points, end_point]))
        
        all_trajectories.append(df[['LAT', 'LON', 'SOG', 'COG']].values)

start_end_points = np.array(start_end_points)
all_sampled_trajectories_points = np.vstack(all_sampled_trajectories_points)

dbscan_all = DBSCAN(eps=0.002, min_samples=2)
clusters_all = dbscan_all.fit_predict(all_sampled_trajectories_points[:, :2])

cluster_centers_all = []
for cluster_label in np.unique(clusters_all):
    if cluster_label == -1:
        continue
    cluster_points = all_sampled_trajectories_points[clusters_all == cluster_label]
    center = cluster_points[:, :2].mean(axis=0)
    cluster_centers_all.append(center)

slow_points_centers = []  
through_points_centers = []  

for center in cluster_centers_all:
    center = np.array(center)
    if center.ndim == 0:
        center = np.array([center, center])  

    distances = np.sqrt(np.sum((all_sampled_trajectories_points[:, :2] - center) ** 2, axis=1).astype('float'))
    cluster_points = all_sampled_trajectories_points[distances <= 0.01]
    sog_values = cluster_points[:, 2]
    cog_values = cluster_points[:, 3]

    slow_points_ratio = np.sum(sog_values <= 0.1) / len(sog_values)

    if slow_points_ratio > 0:
        slow_points_centers.append(center)
    else:
        sog_greater_than_02_ratio = np.sum(sog_values > 0.1) / len(sog_values)

        if sog_greater_than_02_ratio > 0.1:
            cog_mean = np.mean(cog_values)
            cog_diffs = np.abs(cog_values - cog_mean)

            cog_diff_std = np.std(cog_values)
            cog_in_range = np.sum(cog_diffs <= (cog_diff_std * 1.5)) / len(cog_values)

            if cog_in_range < 0.88:
                through_points_centers.append(center)

def merge_centers(cluster_centers, threshold=0.01):
    merged_centers = []
    distance_matrix = cdist(cluster_centers, cluster_centers)
    merged = np.zeros(len(cluster_centers), dtype=bool)

    for i in range(len(cluster_centers)):
        if merged[i]:
            continue
        close_centers = np.where((distance_matrix[i] < threshold) & (distance_matrix[i] > 0))[0]
        if len(close_centers) > 0:
            merged_centers.append(np.mean([cluster_centers[i]] + [cluster_centers[j] for j in close_centers], axis=0))
            merged[i] = True
            merged[close_centers] = True
        else:
            merged_centers.append(cluster_centers[i])
    
    return merged_centers

merged_centers_startend = merge_centers(start_end_points[:, :2])
merged_centers_slow = merge_centers(slow_points_centers)
merged_centers_through = merge_centers(through_points_centers)

plt.figure(figsize=(10, 6), dpi=100)

for trajectory in all_trajectories:
    plt.plot(trajectory[:, 1], trajectory[:, 0], alpha=0.2)

for center in merged_centers_startend:
    plt.scatter(center[1], center[0], color='blue',label='Public Destination' if 'Public Destination' not in plt.gca().get_legend_handles_labels()[1] else "")

for center in merged_centers_slow:
    plt.scatter(center[1], center[0], color='red',label='Public Stay Point' if 'Public Stay Point' not in plt.gca().get_legend_handles_labels()[1] else "")

for center in merged_centers_through:
    plt.scatter(center[1], center[0], color='green',label='Public Transit Point' if 'Public Transit Point' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.legend()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajectory Clusters and Centers')

plt.show()

combined_destination_centers = np.vstack([merged_centers_startend, merged_centers_slow, merged_centers_through])

Dist = 0.02  

for file in all_files2:
    df = pd.read_csv(file)

    if 'LAT' in df.columns and 'LON' in df.columns:
        trajectory = df[['LAT', 'LON']].values
        
        df['Dest_LAT'] = np.nan
        df['Dest_LON'] = np.nan

        for i in range(len(trajectory)):
            current_point = trajectory[i]

            if i < 9:  
                df.at[i, 'Dest_LAT'] = current_point[0]
                df.at[i, 'Dest_LON'] = current_point[1]
                continue
            else:
                previous_point = trajectory[i - 9]

            segment_direction = current_point - previous_point  # 修正后的行进方向向量

            point_distances = np.sqrt(np.sum((combined_destination_centers - current_point) ** 2, axis=1).astype(np.float32))

            direction_vectors = combined_destination_centers - current_point

            valid_destinations = combined_destination_centers[np.dot(direction_vectors, segment_direction) > 0]
            valid_distances = point_distances[np.dot(direction_vectors, segment_direction) > 0]

            if len(valid_destinations) > 0:
                min_distance_idx = np.argmin(valid_distances)
                if valid_distances[min_distance_idx] < Dist:
                    df.at[i, 'Dest_LAT'] = valid_destinations[min_distance_idx, 0]
                    df.at[i, 'Dest_LON'] = valid_destinations[min_distance_idx, 1]
                else:
                    df.at[i, 'Dest_LAT'] = current_point[0]
                    df.at[i, 'Dest_LON'] = current_point[1]
            else:
                df.at[i, 'Dest_LAT'] = current_point[0]
                df.at[i, 'Dest_LON'] = current_point[1]

        output_file = os.path.join(output_folder, os.path.basename(file))
        df.to_csv(output_file, index=False)

        output_file = os.path.join(output_folder, os.path.basename(file))
        df.to_csv(output_file, index=False)
        
        
