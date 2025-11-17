import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Set font and style parameters
ps = 12
matplotlib.rcParams.update(
    {
        "font.size": ps,
        "axes.titlesize": ps,
        "axes.labelsize": ps,
        "xtick.labelsize": ps,
        "ytick.labelsize": ps,
        "font.family": "sans",
        "lines.linewidth": 0.5,
    }
)

# Load data
df = pd.read_pickle("compiled_results.pkl")
print(df.head())

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5), dpi=80)

# Plot histogram
n, bins, patches = ax.hist(
    df["combined_gof"], bins=25, edgecolor="black", color="skyblue"
)

# Labels and title
ax.set_xlabel(r"$gof_\mathrm{combined}$")
ax.set_ylabel("Frequency")
# ax.set_title('Histogram of Combined GOF Values')

# Add grid for better readability
ax.grid(True, linestyle="--", alpha=0.5)

# Tight layout
fig.tight_layout()

# Save figure as SVG and PDF
fn = "figures/gof_combined_histogram.pdf"
fig.savefig(fn)
print(f"Done writing {fn}")

full_path = os.path.abspath(fn)
print(f"full path: {full_path}")
