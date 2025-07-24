import os
import pandas as pd
from glob import glob
from datetime import datetime

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
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    plt.title(f'Label Distribution by Species (Percentages) - {now}')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig("label_distribution_by_species (%).png")

    # Side-by-side barchart with total numbers
    pivot_counts.plot(kind='bar')
    plt.ylabel('Count')
    plt.title(f'Label Distribution by Species (Counts) - {now}')
    plt.legend(title='Label')
    plt.tight_layout()
    plt.savefig("label_distribution_by_species.png")

# --- SUGGESTED SPLIT LOGIC ---
# Pivot to get positive counts per (species, location)
pos_counts = counts[counts['label'] == 1].pivot(index='location', columns='species', values='count').fillna(0).astype(int)

# Print table for review
print("\nPositive (label=1) counts per location/species:")
print(pos_counts)

# Set minimum positive count threshold per species per split
MIN_POS = 1000  # You can adjust this value

# Prepare for assignment
locations = pos_counts.index.tolist()
species = pos_counts.columns.tolist()

# Initialize splits and their species counts
def empty_split():
    return {s: 0 for s in species}
splits = {'train': [], 'validation': [], 'test': []}
split_counts = {'train': empty_split(), 'validation': empty_split(), 'test': empty_split()}

# Sort locations by total positives only (ignore min_species)
pos_counts['total'] = pos_counts.sum(axis=1)
pos_counts_sorted = pos_counts.sort_values(['total'], ascending=False)

# Greedy assignment: assign each location to the split where it helps the most underrepresented species
for loc in pos_counts_sorted.index:
    loc_counts = pos_counts.loc[loc]
    # For each split, compute how much this location helps the least-represented species
    best_split = None
    best_gain = -1
    for split in splits:
        gain = sum(
            max(0, min(MIN_POS - split_counts[split][s], loc_counts[s]))
            for s in species
        )
        if gain > best_gain:
            best_gain = gain
            best_split = split
    splits[best_split].append(loc)
    for s in species:
        split_counts[best_split][s] += loc_counts[s]

# Print suggested splits and their species coverage
print(f"\nMinimum positive samples per species per split: {MIN_POS}")
print("\nSuggested location splits (with positive counts per species):")
for split, locs in splits.items():
    print(f"{split}: {locs}")
    split_df = pos_counts.loc[locs]
    sums = split_df.sum()
    print(sums)
    for s in species:
        if sums[s] < MIN_POS:
            print(f"  WARNING: {split} has only {sums[s]} positives for {s} (less than {MIN_POS})")

# Print summary table: positive samples per species per split
summary = pd.DataFrame({split: [split_counts[split][s] for s in species] for split in splits}).T
summary.columns = species
print("\nSummary: Positive samples per species per split (stage):")
print(summary)

# Optionally, you can adjust the splits above to better balance or to meet your own criteria.
# You can now use these splits to assign locations in your main pipeline.
