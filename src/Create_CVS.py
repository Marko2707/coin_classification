import pandas as pd
import random
from itertools import combinations

"""Creates triplets from an Excel file based on subgroup information for obverse and reverse sides."""

# Lade die Excel-Datei (vereinfachte Version)
df = pd.read_excel("info.xlsx")

def generate_triplets(df, subgroup_column, suffix, output_filename):
    # Gruppiere nach der gegebenen Stempeluntergruppe
    grouped = df.groupby(subgroup_column)["Dédalo ID"].apply(list)

    # Erstelle eine Liste aller (ID, Gruppe)-Paare
    all_data = df[["Dédalo ID", subgroup_column]].values.tolist()

    triplets = []

    # Für jede Gruppe
    for group_name, ids in grouped.items():
        ap_pairs = list(combinations(ids, 2))  # Alle Anchor-Positive-Kombinationen
        negative_ids = [id_ for id_, g in all_data if g != group_name]  # IDs außerhalb der Gruppe

        for anchor, positive in ap_pairs:
            if not negative_ids:
                continue
            negative = random.choice(negative_ids)
            triplets.append((
                f"Anchor_{anchor}_{suffix}.jpg",
                f"Positive_{positive}_{suffix}.jpg",
                f"Negative_{negative}_{suffix}.jpg"
            ))

    # Speichern als CSV
    triplet_df = pd.DataFrame(triplets, columns=["Anchor", "Positive", "Negative"])
    triplet_df.to_csv(output_filename, index=False)
    print(f"{output_filename}: {len(triplets)} Triplets erstellt.")

# Erzeuge Triplets für Av (obverse)
generate_triplets(df, "Stempeluntergruppe Av", "a", "triplets_obv.csv")

# Erzeuge Triplets für Rv (reverse)
generate_triplets(df, "Stempeluntergruppe Rv", "r", "triplets_rev.csv")
