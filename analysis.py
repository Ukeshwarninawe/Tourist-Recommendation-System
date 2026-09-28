import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Numerical columns
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# Categorical columns
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Display categorical columns
print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns.tolist())

# Histogram
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved as cleaned_dataset.csv")
print("Final Shape:", df.shape)