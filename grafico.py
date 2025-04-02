from matplotlib import ticker
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pandas.tseries import frequencies

# Define constant
# rt60 = 0.47
rt60 = 0.8
w = 2.2 / rt60

# Load data from CSV file (ensure it has 'i', 'j', 'k', 'F', and 'A' columns)
csv_file = "data.csv"  # Update with your actual file name

# Check if file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(
        f"Error: The file '{csv_file}' was not found. Please check the file path and try again."
    )

df = pd.read_csv(csv_file)

# Check if required columns exist
required_columns = {"i", "j", "k", "F", "A"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Create bar labels by concatenating 'i', 'j', and 'k'
labels = df["i"].astype(str) + "-" + df["j"].astype(str) + "-" + df["k"].astype(str)

# Convert data types
df["Frequency"] = pd.to_numeric(df["F"], errors="coerce")
df["Amplitude"] = pd.to_numeric(df["A"], errors="coerce")
df["Amplitude"] *= 100

df = df.dropna()  # Remove any rows with invalid data
df = df.sort_values(by="Frequency")  # Ensure frequency is sorted

# Sum overlapping bars
# First, create an array of all frequencies to account for overlaps
base_frequencies = np.unique(df["Frequency"])
frequencies = base_frequencies
widths = frequencies * 0
for i, f in enumerate(widths):
    widths[i] = w
sums = []
sum_widths = []
sum_amps = []

# Sum amplitudes for overlapping frequency points
for i, f in enumerate(base_frequencies):
    if i == base_frequencies.size - 1:
        break
    p1 = f + w / 2
    p2 = base_frequencies[i + 1] - w / 2
    if p1 > p2:
        if p1 - p2 < 1:
            continue
        sums += [(p1 + p2) / 2]
        sum_amps += [df["Amplitude"][i] + df["Amplitude"][i + 1]]
        sum_widths += [p1 - p2]


# Plot
plt.figure(figsize=(12, 6))
colors = np.random.rand(len(df))  # Generate random colors for each data point

# Overlaps
plt.bar(
    sums,
    sum_amps,
    # color=plt.cm.viridis(np.linspace(0, 1, len(frequencies))),
    color=(0.9, 0.4, 0, 0.5),
    width=sum_widths,
    align="center",
)
# Center Black Bars for overlaps
# for i, j in enumerate(sum_widths):
#     sum_widths[i] *= 0.1
#
# plt.bar(
#     sums,
#     sum_amps,
#     color=(0, 0, 0),
#     width=sum_widths,
#     align="center",
# )


# Orange bars and Black center bars
plt.bar(
    frequencies,
    df["Amplitude"],
    # color=plt.cm.viridis(np.linspace(0, 1, len(frequencies))),
    color=(0.7, 0.3, 0, 0.5),
    width=widths,
    align="center",
)
for i, j in enumerate(widths):
    widths[i] *= 0.05
plt.bar(
    frequencies,
    df["Amplitude"],
    color=(0, 0, 0),
    width=widths,
    align="center",
)

plt.xscale("log")
plt.title("Modi Stazionari")
plt.xlabel("Frequenza (Hz)")
plt.ylabel("Ampiezza (%)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.gca().xaxis.set_minor_formatter(
    ticker.FuncFormatter(lambda x, _: "{:.0f}".format(x))
)
plt.show()
