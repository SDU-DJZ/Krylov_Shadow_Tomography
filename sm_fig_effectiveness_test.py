import os
import math
import numpy as np
import json
import multiprocessing
import matplotlib.pyplot as plt
import my_functions as mf
import data_acquisition as da
import prediction_QFI as pr


"""
Load data
"""

with open('numerical_results_sm_effectiveness_test.json', 'r', encoding='utf-8') as file:
    data_all = json.load(file)
    data_chose = data_all["data1"]
    aggregated_results = data_chose["aggregated results"]

N_Total = aggregated_results["N_Total"]
N_LB_Leg = aggregated_results["N_LB_Leg"]
N_LB_Sub = aggregated_results["N_LB_Sub"]
N_LB1 = aggregated_results["N_LB1"]
N_LB2 = aggregated_results["N_LB2"]
N_LB3 = aggregated_results["N_LB3"]
N_K1 = aggregated_results["N_K1"]
N_K2 = aggregated_results["N_K2"]
N_K3 = aggregated_results["N_K3"]

fig, ax = plt.subplots()

bounds = [r'$B^{(\mathsf{Leg})}$', r'$B^{(\mathsf{Sub})}$', r'$B_1^{(\mathsf{Tay})}$', r'$B_2^{(\mathsf{Tay})}$', r'$B_3^{(\mathsf{Tay})}$', r'$B_1^{(\mathsf{Kry})}$', r'$B_2^{(\mathsf{Kry})}$', r'$B_3^{(\mathsf{Kry})}$']
counts = [N_LB_Leg / N_Total, N_LB_Sub / N_Total, N_LB1 / N_Total, N_LB2 / N_Total, N_LB3 / N_Total, N_K1 / N_Total, N_K2 / N_Total, N_K3 / N_Total]
bar_colors = ['tab:red', 'tab:green', 'tab:orange', 'tab:orange', 'tab:orange', 'tab:blue', 'tab:blue', 'tab:blue']
positions = [0, 1, 3, 4, 5, 7, 8, 9]

# Plot the bar chart
bars = ax.bar(positions, counts, color=bar_colors, width=0.8)

# Add percentage annotations on top of each bar
for x, y in zip(positions, counts):
    ax.text(
        x,                  # x-coordinate
        y + 0.01,           # y-coordinate (slightly moved up)
        f"{y * 100:.1f}%",  # Convert to percentage, keep 1 decimal place (e.g. 85.2%)
        ha="center",        # Horizontal alignment to center
        va="bottom",        # Vertical alignment to the bottom
        fontsize=9          # Font size
        # color='black'     # Optional: if the background is dark, set to white
    )

ax.set_xticks(positions)
# ax.set_xticklabels(bounds, rotation=45, ha="right", fontsize=12)
ax.set_xticklabels(
    bounds,
    rotation=45,
    ha="right",
    rotation_mode="anchor",  # 添加这一行
    fontsize=12
)

ax.set_ylabel(r'Ratio between ${num}(B)$ and $num(F_Q)$', fontsize=12)
plt.ylim(0, 1.1)  # Optional: adjust y-axis range to ensure the percentage text is not cut off
plt.tight_layout()

plt.savefig('sm_effectiveness_test.pdf', dpi=600, bbox_inches='tight', pad_inches=0)

plt.show()
