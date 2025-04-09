# Costanti ------------------------------------------------------------------

# RT60 Attuale = 0.8
# RT60 AMROC = 0.23
RT60 = 0.26

SUM_DEPTH = 20
W = 2.2 / RT60

STARTING_COLOR = (0.3, 0.4, 0, 0.5)

Y_AS_LOG = False
SHOW_DB = True
SHOW_LINES_DB = True
SHOW_PERCENT = True
SHOW_LINES_PERCENT = False

# ----------------------------------------------------------------------------

import matplotlib.pyplot as plt

from numpy._core.fromnumeric import size
from numpy.typing import ArrayLike
import pandas as pd
from pandas.tseries import frequencies
import math

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
amplitudes = list(map(lambda x: x * 100, df["Amplitude"]))

df = df.dropna()  # Remove any rows with invalid data

# Sum overlapping bars
# First, create an array of all frequencies to account for overlaps
base_frequencies = np.unique(df["Frequency"])
frequencies = base_frequencies
widths = [W] * frequencies.size


# Sum amplitudes for overlapping frequency points
# Frequency, Amplutude, Width, Color
def calculate_overlaps(values=(ArrayLike, ArrayLike, ArrayLike, ArrayLike)):
    sum_freqs = []
    sum_amps = []
    sum_widths = []
    sum_colors = []

    for i, f1 in enumerate(values[0]):
        if i == size(values[0]) - 1:
            break

        f2 = values[0][i + 1]
        w1 = values[2][i] / 2
        w2 = values[2][i + 1] / 2
        p1 = f1 + w1
        p2 = f2 - w2

        if p1 > p2:
            # Skip bands that are too short
            if p1 - p2 < f1 * 0.01:
                continue

            sum_freqs += [(p1 + p2) / 2]
            sum_amps += [values[1][i] + values[1][i + 1]]
            sum_widths += [p1 - p2]
            sum_colors += [
                tuple(
                    map(
                        lambda a, b: max(min(a + b, 1), 0),
                        values[3][i],
                        (0, -0.03, 0.1, 0),
                    )
                )
            ]

    return (sum_freqs, sum_amps, sum_widths, sum_colors)


# Frequency, Amplutude, Width, Color
def draw_bars(values=(ArrayLike, ArrayLike, ArrayLike, ArrayLike), draw_center=False):
    plt.bar(
        values[0],
        values[1],
        color=values[3],
        width=values[2],
        align="center",
    )

    if not draw_center:
        return
    plt.bar(
        values[0],
        values[1],
        color=(0, 0, 0),
        width=list(map(lambda x: x * 0.03, values[2])),
        align="center",
    )


def percent_to_db(x):
    return (
        x[0],
        list(map(lambda y: 20 * math.log10(y), x[1])),
        x[2],
        tuple(map(lambda _: (0.9, 0, 0, 1), x[3])),
    )


# Plot
plt.figure(figsize=(12, 6))
colors = np.random.rand(len(df))  # Generate random colors for each data point

base = (frequencies, amplitudes, widths, [STARTING_COLOR] * frequencies.size)

# Overlaps
overlaps = [calculate_overlaps(base)]
for i in range(0, SUM_DEPTH):
    new_overlap = calculate_overlaps(overlaps[i])
    if new_overlap == ([], [], [], []):
        print(f"Out of overlaps at iteration {i}")
        break
    # new_overlap = sorted(new_overlap, key=lambda a: a[0])
    overlaps += [new_overlap]

# Scaled to dB
overlaps_db = [base] + overlaps
overlaps_db = list(map(percent_to_db, overlaps_db))

if SHOW_PERCENT:
    for i, o in enumerate(overlaps):
        draw_bars(values=o)

    # Main Freqs
    draw_bars(values=base, draw_center=SHOW_LINES_PERCENT)

if SHOW_DB:
    for i, o in enumerate(reversed(overlaps_db)):
        draw_center = i == len(overlaps_db) - 1 and SHOW_LINES_DB
        draw_bars(values=o, draw_center=draw_center)

plt.xscale("log")
if Y_AS_LOG:
    plt.yscale("log")

plt.title(f"Modi Stazionari (RT60={RT60}, Depth={SUM_DEPTH})")
plt.xlabel("Frequenza (Hz)")
plt.ylabel("Ampiezza")
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
