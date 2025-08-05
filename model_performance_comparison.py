import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for better looking plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Model names and colors
models = ["MLP", "GNN", "Invariant", "Attention"]
colors = {
    "MLP": "#FF6B6B",
    "GNN": "#4ECDC4",
    "Invariant": "#9B59B6",
    "Attention": "#F39C12",
    "Train": "#2ECC71",
    "Test": "#3498DB",
}

# Training and testing results for each model
# Format: [Avg. Reward, Avg. Reward/Episode, Avg. Episode Length, Avg. Captured]
train_results = {
    "MLP": [0.0187, 1.875, 100.0, 1.19],
    "GNN": [0.9408, 30.43, 32.34, 3.36],
    "Invariant": [0.8103, 29.52, 36.44, 3.31],
    "Attention": [1.0524, 32.83, 31.19, 3.58],
}
test_results = {
    "MLP": [-0.0875, -8.75, 100.0, 0.13],
    "GNN": [0.5181, 24.29, 46.89, 2.89],
    "Invariant": [0.1935, 12.42, 64.17, 1.88],
    "Attention": [0.2743, 15.87, 57.87, 2.16],
}

metrics = ["Avg. Reward", "Avg. Reward/Episode", "Avg. Episode Length", "Avg. Captured"]
metric_units = ["", "", "steps", "hosts"]

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Main comparison chart with key insights in first row
gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1])

# Add Key Insights panel in the first row, third column
ax_insights = fig.add_subplot(gs[0, 2])
ax_insights.axis('off')

# Create 3 sections side by side in a single row
# Column 1: Key Insights (left)
ax_insights.text(0.02, 0.95, 
    "🎯 Key Insights:\n"
    "• GNN wins with\n"
    "  best test perf.\n"
    "  (2.89 hosts)\n"
    "• 86% retention",
    transform=ax_insights.transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# Column 2: Efficiency Ranking (center)
ax_insights.text(0.35, 0.95, 
    "⚡ Efficiency:\n"
    "• Attention: 31\n"
    "• GNN: 32\n"
    "• Invariant: 36\n"
    "• MLP: 100",
    transform=ax_insights.transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

# Column 3: Parameter Efficiency (right)
ax_insights.text(0.68, 0.95, 
    "🏗️ Parameters:\n"
    "• Invariant: 13K\n"
    "• Attention: 21K\n"
    "• GNN: 62K\n"
    "• MLP: 181K",
    transform=ax_insights.transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

# Individual metric comparisons
for i, metric in enumerate(metrics):
    if i < 2:
        ax = fig.add_subplot(gs[0, i])
    else:
        ax = fig.add_subplot(gs[1, i - 2])

    x = np.arange(len(models))
    width = 0.35

    train_vals = [train_results[model][i] for model in models]
    test_vals = [test_results[model][i] for model in models]

    bars1 = ax.bar(
        x - width / 2,
        train_vals,
        width,
        label="Training",
        color=colors["Train"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        test_vals,
        width,
        label="Testing",
        color=colors["Test"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel(f"{metric} ({metric_units[i]})", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_title(f"{metric}", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Highlight best performer
    if metric == "Avg. Captured":
        ax.add_patch(
            Rectangle(
                (0.5, -0.5),
                1,
                max(test_vals) + 1,
                fill=False,
                edgecolor="gold",
                linewidth=3,
                linestyle="--",
            )
        )

# Performance retention chart
ax_retention = fig.add_subplot(gs[1, 2])
retention_data = {
    "MLP": [-467, -467, 100, 11],  # Retention percentages
    "GNN": [55, 80, 69, 86],
    "Invariant": [24, 42, 57, 57],
    "Attention": [26, 48, 54, 60],
}

x = np.arange(len(metrics))
width = 0.35

for i, model in enumerate(models):
    bars = ax_retention.bar(
        x + i * width,
        retention_data[model],
        width,
        label=model,
        color=colors[model],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax_retention.annotate(
            f"{height}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

ax_retention.set_ylabel("Performance Retention (%)", fontweight="bold")
ax_retention.set_xlabel("Metrics", fontweight="bold")
ax_retention.set_title(
    "Generalization Analysis", fontsize=12, fontweight="bold", pad=20
)
ax_retention.set_xticks(x + width / 2)
ax_retention.set_xticklabels(["Reward", "Reward/Ep", "Length", "Captured"], rotation=45)
ax_retention.legend()
ax_retention.grid(axis="y", linestyle="--", alpha=0.3)
ax_retention.axhline(y=0, color="red", linestyle="-", alpha=0.5)

# Summary statistics table
ax_summary = fig.add_subplot(gs[2, :])
ax_summary.axis("off")

summary_data = [
    [
        "Architecture",
        "Parameters",
        "Train Hosts",
        "Test Hosts",
        "Retention",
        "Efficiency",
        "Rank",
    ],
    ["GNN", "62,221", "3.36", "2.89", "86%", "32 steps", "🥇 1st"],
    ["Attention", "21,515", "3.58", "2.16", "60%", "31 steps", "🥈 2nd"],
    ["Invariant", "13,067", "3.31", "1.88", "57%", "36 steps", "🥉 3rd"],
    ["MLP", "180,789", "1.19", "0.13", "11%", "100 steps", "4th"],
]

table = ax_summary.table(
    cellText=summary_data[1:],
    colLabels=summary_data[0],
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the table
for i in range(len(summary_data[0])):
    table[(0, i)].set_facecolor("#34495E")
    table[(0, i)].set_text_props(weight="bold", color="white")

for i in range(1, len(summary_data)):
    for j in range(len(summary_data[0])):
        if i == 1:  # GNN row (1st place - winner)
            table[(i, j)].set_facecolor("#E8F8F5")
        elif i == 2:  # Attention row (2nd place)
            table[(i, j)].set_facecolor("#FFF2CC")
        elif i == 3:  # Invariant row (3rd place)
            table[(i, j)].set_facecolor("#F4E6FF")
        else:  # MLP row (4th place)
            table[(i, j)].set_facecolor("#FADBD8")

plt.suptitle(
    "🌟 NASimEmu Architecture Performance Comparison",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# Key insights now moved to first row - no bottom text needed

plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.15)
plt.savefig(
    "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.show()

print("📊 Enhanced performance comparison chart generated!")
print("📈 Features: Value labels, retention analysis, summary table, and insights")
print("🎨 Saved as high-resolution PNG: model_performance_comparison.png")
