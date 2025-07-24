import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the CSV file
datafile = 'Assignations/Run_test_PtGraham_215D_SWCorner_val_AL16_NM1_213D_Iniskin.csv'
df = pd.read_csv(datafile)

# Ensure multilabel is int (if not already)
df['multilabel'] = df['multilabel'].astype(int)

# Convert multilabel to binarylabel: 0 = no whale, 1 = whale present
# (multilabel 1,2,3 -> 1; 0 -> 0)
df['binarylabel'] = df['multilabel'].apply(lambda x: 0 if x == 0 else 1)

# Group and count: for each assigned, species, binarylabel, count occurrences
count_df = df.groupby(['assigned', 'species', 'binarylabel']).size().reset_index(name='count')
print(count_df)

# Plot: assigned on x, count on y, hue=species, stacked by binarylabel (0/1)
plt.figure(figsize=(12, 6))
# We want: for each assigned/species, a bar split by binarylabel (0/1)
# So we pivot to have binarylabel as columns, then plot stacked bars for each species/assigned
pivot = count_df.pivot_table(index=['assigned', 'species'], columns='binarylabel', values='count', fill_value=0)

# Calculate total counts for each assigned/species group for percentage annotation
totals = pivot.sum(axis=1)

assigned_order = ['train', 'validation', 'test'] if set(df['assigned']) == {'train','validation','test'} else sorted(df['assigned'].unique())
species_order = sorted(df['species'].unique())
bar_width = 0.2  # Make bars thinner
bar_gap = 0.15   # Increase gap between groups
num_species = len(species_order)

fig, ax = plt.subplots(figsize=(14, 7))
bar_positions = []
for i, assigned in enumerate(assigned_order):
    group_start = i * (num_species * bar_width + bar_gap)
    for j, species in enumerate(species_order):
        pos = group_start + j * bar_width
        bar_positions.append(pos)
        idx = (assigned, species)
        row = pivot.loc[idx] if idx in pivot.index else pd.Series({0:0, 1:0})
        total = totals[idx] if idx in totals.index else 0
        # Plot absent (0)
        b0 = ax.bar(pos, row.get(0, 0), width=bar_width, color=sns.color_palette()[j], alpha=0.5, label=f'{species} (absent)' if i==0 else "")
        # Plot present (1)
        b1 = ax.bar(pos, row.get(1, 0), width=bar_width, bottom=row.get(0, 0), color=sns.color_palette()[j], alpha=1.0, label=f'{species} (present)' if i==0 else "")
        # Annotate percentages
        if total > 0:
            pct0 = 100 * row.get(0, 0) / total
            pct1 = 100 * row.get(1, 0) / total
            if row.get(0, 0) > 0:
                ax.text(pos, row.get(0, 0)/2, f'{pct0:.1f}%', ha='center', va='center', fontsize=9, color='black')
            if row.get(1, 0) > 0:
                ax.text(pos, row.get(0, 0) + row.get(1, 0)/2, f'{pct1:.1f}%', ha='center', va='center', fontsize=9, color='white')

# Set x-ticks and labels at the center of each group
group_centers = [i * (num_species * bar_width + bar_gap) + (num_species-1)*bar_width/2 for i in range(len(assigned_order))]
ax.set_xticks(group_centers)
ax.set_xticklabels(assigned_order)
ax.set_xlabel('Assigned')
ax.set_ylabel('Count')
ax.set_title('Label Counts by Assigned and Species (Stacked by Whale Presence)')
# Custom legend: one for each species, with (present) and (absent) in legend
handles = []
for j, species in enumerate(species_order):
    handles.append(
        plt.Rectangle((0,0),1,1, color=sns.color_palette()[j], alpha=1.0, label=f'{species} (present)')
    )
    handles.append(
        plt.Rectangle((0,0),1,1, color=sns.color_palette()[j], alpha=0.5, label=f'{species} (absent)')
    )
ax.legend(handles=handles, title='Species (whale presence)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
os.makedirs('Diagnostics', exist_ok=True)
plt.savefig('Diagnostics/Assigned_Distribution_Stacked.png')
plt.close()
print('Plot saved to Diagnostics/Assigned_Distribution_Stacked.png')


