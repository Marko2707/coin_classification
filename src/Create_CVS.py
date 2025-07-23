""" Generate triplets from an Excel file for coin classification into new csv files "triplets_obv.csv" and "triplets_rev.csv"."""
import pandas as pd
import random
from itertools import combinations

#Loads the Excel file given in the course (The simple version) 
df = pd.read_excel("info.xlsx")

""" Creates new csv with triplets based on the Dédalo ID and Stempeluntergruppe. 
--> Returns us a cross product of all combinations
 of Anchor and Positive images for each Stempeluntergruppe, with a random Negative image from another Stempeluntergruppe. """
def generate_triplets(df, subgroup_column, suffix, output_filename):
    # Group by the specified subgroup column
    grouped = df.groupby(subgroup_column)["Dédalo ID"].apply(list)

    # Create a list of all Dédalo IDs and their corresponding subgroup
    all_data = df[["Dédalo ID", subgroup_column]].values.tolist()

    triplets = []

    # For each group, create triplets
    for group_name, ids in grouped.items():
        ap_pairs = list(combinations(ids, 2))  # All anchor positive pairs 
        negative_ids = [id_ for id_, g in all_data if g != group_name]  # Another id from a different cluster (Group)
        
        # Create triplets by pairing each anchor with each positive and a random negative
        for anchor, positive in ap_pairs:
            if not negative_ids:
                continue
            negative = random.choice(negative_ids)
            triplets.append((
                f"Anchor_{anchor}_{suffix}.jpg",
                f"Positive_{positive}_{suffix}.jpg",
                f"Negative_{negative}_{suffix}.jpg"
            ))

    # Save as CSV for later use
    triplet_df = pd.DataFrame(triplets, columns=["Anchor", "Positive", "Negative"])
    triplet_df.to_csv(output_filename, index=False)
    print(f"{output_filename}: {len(triplets)} Triplets erstellt.")

# Create triplets for Obv (obverse) --> "triplets_obv.csv"
generate_triplets(df, "Stempeluntergruppe Av", "a", "triplets_obv.csv")

# Create triplets for Rev (reverse) --> "triplets_rev.csv"
generate_triplets(df, "Stempeluntergruppe Rv", "r", "triplets_rev.csv")
