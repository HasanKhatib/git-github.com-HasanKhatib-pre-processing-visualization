import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("ex2_data.csv")

# generate timestamp from Date and Time columns
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S:%f')

# calculate time in seconds from the start of the recording
df['time_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

# Define acceleration thresholds to remove near-zero movement
# since the data is huge, I will use a threshold of 1.9 m/s²
ax_threshold = 1.9
ay_threshold = 1.9
az_threshold = 1.9

# apply the threshold to filter out stationary data
mask = (np.abs(df['Ax']) > ax_threshold) | (np.abs(df['Ay']) > ay_threshold) | (np.abs(df['Az']) > az_threshold)
df_filtered = df[mask]

# Plot the filtered data
fig, ax1 = plt.subplots(figsize=(10, 5))

# First y-axis for acceleration (Ax, Ay, Az)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Acceleration (m/s²)", color='tab:blue')
ax1.plot(df_filtered['time_seconds'], df_filtered['Ax'], label='Ax', color='b', linestyle='-')
ax1.plot(df_filtered['time_seconds'], df_filtered['Ay'], label='Ay', color='c', linestyle='--')
ax1.plot(df_filtered['time_seconds'], df_filtered['Az'], label='Az', color='m', linestyle='-.')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create second y-axis for gyroscope (Gx, Gy, Gz)
ax2 = ax1.twinx()
ax2.set_ylabel("Gyroscope (°/s)", color='tab:red')
ax2.plot(df_filtered['time_seconds'], df_filtered['Gx'], label='Gx', color='r', linestyle='-')
ax2.plot(df_filtered['time_seconds'], df_filtered['Gy'], label='Gy', color='orange', linestyle='--')
ax2.plot(df_filtered['time_seconds'], df_filtered['Gz'], label='Gz', color='brown', linestyle='-.')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Add grid, legend, and title
ax1.grid(True)
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
plt.title("Filtered Motion Data (Stationary Data Removed)")
plt.show()