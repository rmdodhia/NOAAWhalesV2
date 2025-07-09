import os
import pandas as pd
from glob import glob

# Set the base directory
base_dir = "DataInput"

# Find all species directories
species_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Collect all label dataframes
dfs = []
for species in species_dirs:
    label_path = os.path.join(base_dir, species, "LabelsOverlap400ms", f"{species}_labels.csv")
    if os.path.exists(label_path):
        df = pd.read_csv(label_path)
        df['species'] = species  # Add species column
        dfs.append(df)

# Combine all dataframes
if dfs:
    all_labels = pd.concat(dfs, ignore_index=True)
    # Count by species and location
    if 'location' in all_labels.columns and 'label' in all_labels.columns:
        counts = all_labels.groupby(['species', 'location', 'label']).size().reset_index(name='count')
        print(counts)
    else:
        print("Required columns 'location' or 'label' not found in the CSV files.")
else:
    print("No label files found.")

import matplotlib.pyplot as plt

if dfs and 'location' in all_labels.columns and 'label' in all_labels.columns:
    # Pivot for plotting
    pivot_counts = counts.pivot_table(index='species', columns='label', values='count', aggfunc='sum', fill_value=0)
    # Stacked barchart in percentages
    percent = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100
    percent.plot(kind='bar', stacked=True)
    plt.ylabel('Percentage')
    plt.title('Label Distribution by Species (Percentages)')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig("label_distribution_by_species (%).png")

    # Side-by-side barchart with total numbers
    pivot_counts.plot(kind='bar')
    plt.ylabel('Count')
    plt.title('Label Distribution by Species (Counts)')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig("label_distribution_by_species.png")
