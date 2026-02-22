# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
path = kagglehub.dataset_download("rrkcoder/swiggy-restaurants-dataset")

import os
import pandas as pd
print("Dataset path:", path)
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            csv_path = os.path.join(root, file)
            print(f"\n--- {file} ---")
            df = pd.read_csv(csv_path, nrows=5)
            print("Columns:", df.columns.tolist())
            print(df.head())

print("First 5 records:", df.head())