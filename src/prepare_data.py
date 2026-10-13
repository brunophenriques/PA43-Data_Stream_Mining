import pandas as pd

from features import (
    add_target,
    add_representation_a,
    add_representation_b,
    save_dataset
)


# Load raw data
faro = pd.read_csv("data/raw/faro_2021_2025.csv")
angra = pd.read_csv("data/raw/angra_2021_2025.csv")

# Parse timestamps
faro["time"] = pd.to_datetime(faro["time"])
angra["time"] = pd.to_datetime(angra["time"])

# Add target
faro = add_target(faro)
angra = add_target(angra)

# Build representations
faro_a = add_representation_a(faro)
faro_b = add_representation_b(faro)

angra_a = add_representation_a(angra)
angra_b = add_representation_b(angra)

# Save processed datasets
save_dataset(faro_a, "data/processed/faro_representation_a.csv")
save_dataset(faro_b, "data/processed/faro_representation_b.csv")

save_dataset(angra_a, "data/processed/angra_representation_a.csv")
save_dataset(angra_b, "data/processed/angra_representation_b.csv")
print("Processed datasets saved successfully.")