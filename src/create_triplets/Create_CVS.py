import pandas as pd
import random
from itertools import combinations

"""Creates triplets from an Excel file containing IDs and their subgroup information. These are then saved to a CSV file."""

# Load the Excel file
#Change the path to your Excel file (Die vereinfachte Version: Stempelliste_bueschel_Neuses_einfach.xlsx)
df = pd.read_excel("info.xlsx")

# Create a dictionary mapping subgroup to list of IDs
grouped = df.groupby("Stempeluntergruppe Av")["Dédalo ID"].apply(list)

# Flatten to list of all IDs and their group
all_data = df[["Dédalo ID", "Stempeluntergruppe Av"]].values.tolist()

triplets = []

# Loop through each group
for group_name, ids in grouped.items():
    
    # All combinations of anchor-positive pairs in the same group
    ap_pairs = list(combinations(ids, 2))

    # Get negatives: all IDs not in the current group
    negative_ids = [id_ for id_, g in all_data if g != group_name]
    
    for anchor, positive in ap_pairs:
        if not negative_ids:
            continue
        negative = random.choice(negative_ids)
        triplets.append((
            f"Anchor_{anchor}_a.jpg",
            f"Positive_{positive}_a.jpg",
            f"Negative_{negative}_a.jpg"
        ))

# Optional: save to CSV
triplet_df = pd.DataFrame(triplets, columns=["Anchor", "Positive", "Negative"])
triplet_df.to_csv("triplets.csv", index=False)

print(f"Generated {len(triplets)} triplets.")
