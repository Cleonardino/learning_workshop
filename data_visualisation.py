import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from data_preparation import columns, train_data

DATA_PATH = "./data/"
OUTPUT_PATH = "./outputs"
Path(OUTPUT_PATH).mkdir(exist_ok=True)

def time_step_to_minutes(time_step):
	return int(pd.Timestamp(time_step).timestamp() / 60)


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



# Create histograms
fig, axes = plt.subplots(len(columns), 1, figsize=(12, 18))


for ax, column in zip(axes, columns):
	ax.hist(train_data[column].dropna(), bins=50)
	ax.set_title(f"Distribution of {column}")
	ax.set_ylabel("Frequency")
	ax.grid(True, alpha=0.3)


axes[-1].set_xlabel("Value")
fig.suptitle("Training Data Feature Distributions", fontsize=16)


plt.tight_layout(rect=[0, 0, 1, 0.96])


# Save figure
output_file = Path(OUTPUT_PATH) / "train_data_histograms.png"
plt.savefig(output_file, dpi=300)
plt.close()


print(f"Histogram visualization saved to {output_file}")