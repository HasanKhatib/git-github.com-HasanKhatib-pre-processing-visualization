import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ex3_data.csv")

# Data preprocessing
df["noted_date"] = pd.to_datetime(df["noted_date"], format="%d-%m-%Y %H:%M")

df["out/in"] = df["out/in"].replace({"In": 1, "Out": 0})

df["date"] = df["noted_date"].dt.date
df["time"] = df["noted_date"].dt.time

# filter data for a specific date (08-12-2018)
df_filtered = df[df["date"] == pd.to_datetime("08-12-2018").date()].copy()

print(df_filtered.head())

# filter data for a specific week (02-12-2018 to 08-12-2018)
df_week = df[(df["date"] >= pd.to_datetime("02-12-2018").date()) & 
             (df["date"] <= pd.to_datetime("08-12-2018").date())]

indoor = df_week[df_week["out/in"] == 1]
outdoor = df_week[df_week["out/in"] == 0]

# plot indoor and outdoor temperature
plt.figure(figsize=(10, 5))
plt.plot(indoor["noted_date"], indoor["temp"], label="Indoor Temperature", color="blue", linestyle="-")
plt.plot(outdoor["noted_date"], outdoor["temp"], label="Outdoor Temperature", color="red", linestyle="--")

# Labels and title
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Indoor vs. Outdoor Temperature (02-12-2018 to 08-12-2018)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

plt.show()