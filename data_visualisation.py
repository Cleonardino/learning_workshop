import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from data_preparation import columns, import_datasets, prepare_data, prepare_label

DATA_PATH = "./data/"
OUTPUT_PATH = "./visualisation_outputs"
Path(OUTPUT_PATH).mkdir(exist_ok=True)

train_data, train_labels, test_data = import_datasets()
train_data = prepare_data(train_data)
train_labels = prepare_label(train_labels)
test_data = prepare_data(test_data)
all_data_nan = 0
only_one_nan = 0
clean_line_count = 0
for i in range(train_data["time_step"].count()):
	if np.isnan(train_data["consumption"][i]) and np.isnan(train_labels["washing_machine"][i]):
		all_data_nan += 1
	elif np.isnan(train_data["consumption"][i]) or np.isnan(train_labels["washing_machine"][i]):
		only_one_nan += 1
	else:
		clean_line_count += 1
print(f"Count both null lines : {all_data_nan}")
print(f"Count only label or data nan : {only_one_nan}")
print(f"Count no nan lines : {clean_line_count}")

# Create visualization
fig, axes = plt.subplots(len(columns), 1, figsize=(14, 18), sharex=True)

for ax, column in zip(axes, columns):
	ax.plot(train_data["minutes_since_Epoch"], train_data[column])
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

import seaborn as sns

# Compute correlation matrix
corr_matrix = train_data[columns].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Matrix", fontsize=16)

# Save figure
output_file = Path(OUTPUT_PATH) / "train_data_correlation_matrix.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Correlation matrix saved to {output_file}")