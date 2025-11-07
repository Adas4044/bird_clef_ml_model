#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

csv_path = Path('../outputs/model_comparison.csv')
if not csv_path.exists():
    csv_path = Path('outputs/model_comparison.csv')
df = pd.read_csv(csv_path)

print("Current metrics:")
print(df)
print()

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(df))
width = 0.2

metrics_to_plot = ['precision', 'recall', 'f1']
colors = ['#3498db', '#2ecc71', '#e74c3c']

for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    offset = width * (i - 1)
    ax.bar(x + offset, df[metric], width, label=metric.capitalize(), color=color)

ax.set_ylabel('Score')
ax.set_title('Model Comparison - Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(df['model'])
ax.legend()
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    offset = width * (i - 1)
    for j, v in enumerate(df[metric]):
        ax.text(j + offset, v + 0.01, f'{v:.3f}',
               ha='center', va='bottom', fontsize=9)

plt.tight_layout()

output_path = Path('../outputs/model_comparison.png')
if not output_path.parent.exists():
    output_path = Path('outputs/model_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"New plot saved to {output_path}")
print("\nValues in the plot:")
for idx, row in df.iterrows():
    print(f"{row['model']}:")
    print(f"  Precision: {row['precision']:.3f}")
    print(f"  Recall: {row['recall']:.3f}")
    print(f"  F1: {row['f1']:.3f}")
