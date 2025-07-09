import os
import csv
from collections import Counter

# Path to the Assignations folder and the file to load
assignations_folder = "Assignations"
filename = "Run_test_AL16_BS4_201D_val_Iniskin_215D.csv"  # Change this to your actual file name
filepath = os.path.join(assignations_folder, filename)

split_counts = Counter()

with open(filepath, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        split = row['assigned']  # 'train', 'validation', or 'test'
        split_counts[split] += 1

print("Sample counts by split:")
for split, count in split_counts.items():
    print(f"  {split}: {count}")