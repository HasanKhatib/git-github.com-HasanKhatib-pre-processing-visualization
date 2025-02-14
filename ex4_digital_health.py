import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("ex4_data.csv")

# Display dataset info
print(df.info())
print(df.describe())

### PART 1: Transform "calories" to be Normally Distributed ###

# Apply different transformations
df["calories_log"] = np.log1p(df["calories"])  # Log transformation
df["calories_sqrt"] = np.sqrt(df["calories"])  # Square root transformation
df["calories_cbrt"] = np.cbrt(df["calories"])  # Cube root transformation

# Plot histograms for each transformation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(df["calories_log"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Log Transform")

sns.histplot(df["calories_sqrt"], bins=30, kde=True, ax=axes[1])
axes[1].set_title("Square Root Transform")

sns.histplot(df["calories_cbrt"], bins=30, kde=True, ax=axes[2])
axes[2].set_title("Cube Root Transform")

plt.show()

# Select the best transformation based on visual inspection
df["calories_transformed"] = df["calories_cbrt"]  # Assuming cube root is best


### PART 2: Keep One Sample per Participant, then Visualize "Age", "Height", and "Weight" ###
df_unique = df.drop_duplicates(subset=["X1"], keep="first").copy()

print(f"Unique participants: {df_unique.shape[0]}")

# Plot Age, Height, and Weight
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].plot(df_unique["X1"], df_unique["age"], color="red", label="Age")
axes[1].plot(df_unique["X1"], df_unique["height"], color="blue", label="Height")
axes[2].plot(df_unique["X1"], df_unique["weight"], color="green", label="Weight")

axes[0].set_title("Age Distribution")
axes[1].set_title("Height Distribution")
axes[2].set_title("Weight Distribution")

for ax in axes:
    ax.legend(loc="upper right")
    ax.grid(True)

axes[2].set_xlabel("Participant ID")

plt.show()


### PART 3: Visualize "Steps", "Heart Rate", and "Calories" for First 3 Participants ###
# !IMPORTANT: for the x-axis, we will use the index of the dataframe.
#               A different graph would have been more appropriate than the line plot as the data is not continuous

df_three = df[df["X1"].isin([1, 2, 3])]

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Steps
for pid, color in zip([1, 2, 3], ["blue", "green", "red"]):
    subset = df_three[df_three["X1"] == pid]
    axes[0].plot(subset.index, subset["steps"], label=f"Participant {pid}", color=color)
axes[0].set_title("Steps")
axes[0].legend(loc="upper right")
axes[0].grid(True)

# Heart Rate
for pid, color in zip([1, 2, 3], ["blue", "green", "red"]):
    subset = df_three[df_three["X1"] == pid]
    axes[1].plot(subset.index, subset["hear_rate"], label=f"Participant {pid}", color=color)
axes[1].set_title("Heart Rate")
axes[1].legend(loc="upper right")
axes[1].grid(True)

# Calories
for pid, color in zip([1, 2, 3], ["blue", "green", "red"]):
    subset = df_three[df_three["X1"] == pid]
    axes[2].plot(subset.index, subset["calories"], label=f"Participant {pid}", color=color)
axes[2].set_title("Calories")
axes[2].legend(loc="upper right")
axes[2].grid(True)

plt.show()


### PART 4: Normalize and Standardize Selected Columns ###
# Example for the normalization: if we have a column with values [10, 20, 30], after normalization we will have [0, 0.5, 1]
# Example for the standardization: if we have a column with values [10, 20, 30], after standardization we will have [-1, 0, 1]
print(f"\n\nPart 4: Normalize and Standardize Selected Columns")

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

df[["age_norm", "height_norm", "weight_norm"]] = minmax_scaler.fit_transform(df[["age", "height", "weight"]])
df[["steps_std", "heartrate_std"]] = standard_scaler.fit_transform(df[["steps", "hear_rate"]])

print(df.head())


### PART 5: Split Data into Train (70%), Validation (15%), and Test (15%) ###
print(f"\n\nPart 5: Split Data into Train, Validation, and Test Sets")
# df_train will be 70% of the original dataset
# df_val and df_test will be 15% each of the original dataset
df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# Display dataset sizes
print(f"Training Set: {df_train.shape[0]} rows")
print(f"Validation Set: {df_val.shape[0]} rows")
print(f"Test Set: {df_test.shape[0]} rows")