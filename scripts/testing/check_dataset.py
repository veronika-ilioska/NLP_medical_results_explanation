import os

import pandas as pd

df = pd.read_csv(os.getenv("LABS_GENERATION"))

print(df.columns.tolist())
print(df[["lab_name", "input_text", "target_text"]].head())
print("Rows:", len(df))
#nemame target_text
#prvo probuvam na 20 rows za da vidam dali se ok podatocite

df = pd.read_csv(os.getenv("SILVER_RULE"))

print(df.columns.tolist())
print(df[["lab_name", "input_text", "target_text"]].head())
print("Rows:", len(df))