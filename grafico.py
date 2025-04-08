# Costanti ------------------------------------------------------------------

# RT60 Attuale = 0.8
# RT60 AMROC = 0.23
# RT60 "Ideale" = 0.47
RT60 = 0.8

SUM_DEPTH = 2
W = 2.2 / RT60

# ----------------------------------------------------------------------------

from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from numpy._core.fromnumeric import size
from numpy.typing import ArrayLike
import pandas as pd
from pandas.tseries import frequencies

import os
import numpy as np

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
    widths[i] = W


# Sum amplitudes for overlapping frequency points
def calculate_overlaps(values=(ArrayLike, ArrayLike, ArrayLike)):
    sums = []
    sum_widths = []
    sum_amps = []

    for i, f in enumerate(values[0]):
        if i == size(values[0]) - 1:
            break
        p1 = f + W / 2
        p2 = values[0][i + 1] - W / 2
        if p1 > p2:
            if p1 - p2 < 1:
                continue
            sums += [(p1 + p2) / 2]
            sum_amps += [values[1][i] + values[1][i + 1]]
            sum_widths += [p1 - p2]

    return (sums, sum_amps, sum_widths)


def draw_bars(values=(ArrayLike, ArrayLike, ArrayLike), draw_center=False):
    plt.bar(
        values[0],
        values[1],
        # color=plt.cm.viridis(np.linspace(0, 1, len(frequencies))),
        color=(0.7, 0.3, 0, 0.5),
        width=W,
        align="center",
    )

    if not draw_center:
        return
    for i, _ in enumerate(widths):
        values[2][i] *= 0.05
    plt.bar(
        values[0],
        values[1],
        color=(0, 0, 0),
        width=values[2],
        align="center",
    )


# Plot
plt.figure(figsize=(12, 6))
colors = np.random.rand(len(df))  # Generate random colors for each data point

base = (frequencies, df["Amplitude"], widths)

# Overlaps
overlaps = [calculate_overlaps(base)]
draw_bars(values=overlaps[0])
for i in range(1, SUM_DEPTH):
    overlaps += [calculate_overlaps(overlaps[i - 1])]
    draw_bars(values=overlaps[i])

# Main Freqs
draw_bars(values=base, draw_center=True)


plt.xscale("log")
plt.yscale("log" if SUM_DEPTH != 1 else "linear")

plt.title(f"Modi Stazionari (RT60={RT60}, Depth={SUM_DEPTH})")
plt.xlabel("Frequenza (Hz)")
plt.ylabel("Ampiezza (%)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

""" Possible semitone fix

def hz_to_semitones(x):
    return 12 * np.log2(x / 440)

plt.gca().xaxis.set_minor_formatter(
    ticker.FuncFormatter(lambda x, _: f"{hz_to_semitones(x):.1f}".format(x))
)
formatter = FuncFormatter(lambda x, pos: f"{hz_to_semitones(x, pos):.1f}")
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(formatter)
"""

plt.tight_layout()
plt.show()
