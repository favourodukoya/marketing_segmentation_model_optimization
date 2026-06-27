import pandas as pd

# 1. Load the CSV file
file_path = r"C:\Users\3ples\Desktop\__NXU__\module6_ANN\data\teleconnect.csv"
df = pd.read_csv(file_path)

# 2. Loop through each column to find missing values
for col in df.columns:
    # Count total missing values in the column
    missing_count = df[col].isna().sum()

    # Get the row numbers (index) where the values are missing
    # Note: df.index starts at 0. If you want 1-indexed row numbers, use: df[df[col].isna()].index + 1
    missing_rows = df[df[col].isna()].index.tolist()

    # 3. Print the results in your requested format
    print(f"{col}: missing-{missing_count}")
    print(f"{col}: {missing_rows}")
    print()  # Adds an empty line between columns for readability
