import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ex1_data.csv")

df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')

plt.figure(figsize=(10, 5))
plt.plot(df['time'], df['temperature'], color='orange', label='Temperature')

# Customizations
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Variation in Malmö")
plt.legend(loc="upper right")
plt.grid(True)
plt.xticks(rotation=45)

plt.show()