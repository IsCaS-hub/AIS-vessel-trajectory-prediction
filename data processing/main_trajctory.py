# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:33:05 2024

@author: jings
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

output_base_path = 'E:/Jing_Project/Boat trajectory and destination prediction/Jing_new/trajctory_extraction_clean_filter'
save_path = 'E:/Jing_Project/Boat trajectory and destination prediction/Jing_new/merged_trajectories'

if not os.path.exists(save_path):
    os.makedirs(save_path)

all_files = [os.path.join(output_base_path, f) for f in os.listdir(output_base_path) if f.endswith('.csv')]

all_trajectories = []
trajectory_files = []

for file in all_files:
    sub_df = pd.read_csv(file)
    if 'LAT' in sub_df.columns and 'LON' in sub_df.columns:
        all_trajectories.append(sub_df)
        trajectory_files.append(file)

def simplify_trajectory(trajectory, step=20):
    return trajectory.iloc[::step]

def compute_dtw_similarity(trajectory1, trajectory2):
    distance, _ = fastdtw(trajectory1[['LAT', 'LON']].values, trajectory2[['LAT', 'LON']].values, dist=euclidean)
    return distance

def compute_trajectory_length(trajectory):
    coords = trajectory[['LAT', 'LON']].values
    return np.sum([euclidean(coords[i], coords[i+1]) for i in range(len(coords)-1)])

def merge_similar_trajectories(trajectories, trajectory_files, threshold):
    n = len(trajectories)
    used = np.zeros(n)  
    unique_trajectories = []

    for i in range(n):
        if used[i]:
            continue

        current_cluster = [trajectories[i]]
        current_files = [trajectory_files[i]]
        used[i] = 1

        simplified_trajectory_i = simplify_trajectory(trajectories[i]) 
        for j in range(i + 1, n):
            if not used[j]:
                simplified_trajectory_j = simplify_trajectory(trajectories[j])  
                similarity = compute_dtw_similarity(simplified_trajectory_i, simplified_trajectory_j)

                if similarity < threshold:
                    current_cluster.append(trajectories[j])
                    current_files.append(trajectory_files[j])
                    used[j] = 1  

        longest_trajectory_idx = np.argmax([compute_trajectory_length(traj) for traj in current_cluster])
        longest_trajectory = current_cluster[longest_trajectory_idx]
        longest_trajectory_file = current_files[longest_trajectory_idx]

        save_trajectory(longest_trajectory, longest_trajectory_file)

        unique_trajectories.append((longest_trajectory, longest_trajectory_file))
    
    return unique_trajectories

def save_trajectory(trajectory, file_name):
    base_name = os.path.basename(file_name)
    save_file_path = os.path.join(save_path, base_name)
    trajectory.to_csv(save_file_path, index=False)

threshold = 0.1  

main_trajectories = merge_similar_trajectories(all_trajectories, trajectory_files, threshold)

plt.figure(figsize=(10, 6), dpi=100)

for trajectory, _ in main_trajectories:
    plt.plot(trajectory['LON'], trajectory['LAT'], alpha=0.7)

plt.title('Main Trajectories Visualization (After Similarity Merging)')
plt.xlim((-65.644, -65.221))
plt.ylim((18.069, 18.363))
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()
