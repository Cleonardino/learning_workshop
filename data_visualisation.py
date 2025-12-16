import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = "./data/"
OUTPUT_PATH = "./outputs"
Path(OUTPUT_PATH).mkdir(exist_ok=True)




def time_step_to_minutes(time_step):
	return int(pd.Timestamp(time_step).timestamp() / 60)




# Load and preprocess data
train_data = pd.read_csv(Path(DATA_PATH) / "X_train.csv")
train_data = train_data.drop(columns="Unnamed: 9", errors="ignore")


train_data["time_step"] = train_data["time_step"].apply(time_step_to_minutes)


columns = [
"visibility",
"temperature",
"humidity",
"humidex",
"windchill",
"wind",
"pressure",
]


for column in columns:
	train_data[column] = train_data[column].interpolate()
	train_data[column] = train_data[column].bfill()


# Sort by time just in case
train_data = train_data.sort_values("time_step")


# Create visualization
fig, axes = plt.subplots(len(columns), 1, figsize=(14, 18), sharex=True)


for ax, column in zip(axes, columns):
	ax.plot(train_data["time_step"], train_data[column])
	ax.set_ylabel(column)
	ax.grid(True, alpha=0.3)


axes[-1].set_xlabel("Time (minutes since epoch)")
fig.suptitle("Training Data Time-Series Visualization", fontsize=16)


plt.tight_layout(rect=[0, 0, 1, 0.96])


# Save figure
output_file = Path(OUTPUT_PATH) / "train_data_visualization.png"
plt.savefig(output_file, dpi=300)
plt.close()


print(f"Visualization saved to {output_file}")