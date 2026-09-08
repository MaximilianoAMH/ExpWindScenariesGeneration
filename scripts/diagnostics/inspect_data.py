from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/wind_dataset.csv")

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumnas:")
print(df.columns.tolist())

print("\nTipos:")
print(df.dtypes)

print("\nPrimeras filas:")
print(df.head())

print("\nÚltimas filas:")
print(df.tail())

print("\nValores nulos:")
print(df.isna().sum())

# DATE a datetime
df["DATE"] = pd.to_datetime(df["DATE"])

print("\nRango temporal:")
print("Inicio:", df["DATE"].min())
print("Fin:", df["DATE"].max())

print("\nEstadísticas WIND:")
print(df["WIND"].describe())