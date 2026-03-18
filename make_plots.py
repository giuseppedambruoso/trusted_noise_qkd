import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# Publication-Quality Matplotlib Settings
# ==========================================
plt.rcParams.update(
    {
        "font.family": "serif",  # Serif fonts are standard for academic papers
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.2,  # Thicker axis borders
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "in",  # Ticks point inwards (classic journal style)
        "ytick.direction": "in",
        "xtick.top": True,  # Show ticks on the top axis
        "ytick.right": True,  # Show ticks on the right axis
        "figure.dpi": 300,  # High resolution
    }
)

# 1. Load the merged data
df = pd.read_csv("merged_results.csv")

# ==========================================
# PLOT 1: Delta R vs Alpha
# ==========================================
df["R_diff"] = df["R_star"] - df["R_0"]
deltar_df = df.groupby("alpha")["R_diff"].max().reset_index()
deltar_df.rename(columns={"R_diff": "deltar"}, inplace=True)
deltar_df = deltar_df.sort_values(by="alpha")

fig1, ax1 = plt.subplots(figsize=(8, 5))

# Using a black line with unfilled/white-filled markers for a clean, high-contrast look
ax1.plot(
    deltar_df["alpha"],
    deltar_df["deltar"],
    marker="o",
    linestyle="-",
    color="black",
    linewidth=1.5,
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=1.2,
)

ax1.set_xlabel(r"Alpha ($\alpha$)")
ax1.set_ylabel(r"$\Delta R = \max_p (R^* - R_0)$")
# Many papers omit titles and use captions instead, but keeping it clean if you need it
ax1.set_title(r"$\Delta R$ vs. Alpha", pad=15)

# Scientific plots typically avoid dense grids; rely on inward ticks instead.
ax1.grid(False)

output_filename_1 = "deltar_vs_alpha_pub.png"
plt.savefig(output_filename_1, bbox_inches="tight")
plt.close(fig1)
print(f"Plot 1 successfully saved as {output_filename_1}")


# ==========================================
# PLOT 2: Contour Plot (q* vs alpha and p)
# ==========================================
q_col = "Q_star"

fig2, ax2 = plt.subplots(figsize=(8, 5))

# 1. Filled contour (the colors)
contour_filled = ax2.tricontourf(
    df["alpha"], df["p"], df[q_col], levels=20, cmap="viridis", alpha=0.9
)

# 2. Line contours overlaid (adds crisp boundaries to the color shifts)
contour_lines = ax2.tricontour(
    df["alpha"],
    df["p"],
    df[q_col],
    levels=20,
    colors="black",
    linewidths=0.4,
    alpha=0.6,
)

# Add colorbar and format its border/label
cbar = plt.colorbar(contour_filled, ax=ax2)
cbar.outline.set_linewidth(1.2)
cbar.ax.tick_params(direction="in", width=1.2)
# Rotate the label so it reads naturally top-to-bottom
cbar.set_label(r"$q^*$", rotation=270, labelpad=20, fontsize=14)

ax2.set_xlabel(r"Alpha ($\alpha$)")
ax2.set_ylabel(r"$p$")
ax2.set_title(r"Contour Plot: $q^*$ vs $\alpha$ and $p$", pad=15)

output_filename_2 = "contour_q_star_pub.png"
plt.savefig(output_filename_2, bbox_inches="tight")
plt.close(fig2)
print(f"Plot 2 successfully saved as {output_filename_2}")
