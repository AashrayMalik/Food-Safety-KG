import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

_SRC = Path(__file__).resolve().parents[1]

# Load CSV
INPUT_CSV = _SRC / "data" / "FoodItems" / "Milk" / "processed" / "chunks" / "milk_dairy_chunks.csv"
OUTPUT_CSV = _SRC / "data" / "FoodItems" / "Milk" / "processed" / "chunks" / "milk_validation_set.csv"

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input chunks CSV not found: {INPUT_CSV}. "
        "Run chunk_fooditem_sources.py first to generate it."
    )

df = pd.read_csv(INPUT_CSV)

# Column containing semicolon-separated labels
LABEL_COL = "lifecycle_stage_hint"

# Convert strings to lists
label_lists = df[LABEL_COL].fillna("").str.split(";")

# Multi-hot encode labels
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_lists)

# Dummy feature matrix (only used to track row indices)
X = np.arange(len(df)).reshape(-1, 1)

# Sample 25% of rows
X_remaining, Y_remaining, X_sample, Y_sample = iterative_train_test_split(
    X,
    Y,
    test_size=0.25
)

# Retrieve sampled rows
sampled_df = df.iloc[X_sample.flatten()]

# Save
sampled_df.to_csv(OUTPUT_CSV, index=False)

print(f"Original rows: {len(df)}")
print(f"Sampled rows: {len(sampled_df)}")