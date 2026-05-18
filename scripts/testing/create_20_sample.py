import os

import pandas as pd

input_path = os.getenv("LABS_GENERATION")
output_path = os.getenv("LABS_TESTING")

df = pd.read_csv(input_path)

sample = df.head(20)

sample.to_csv(output_path, index=False)

print("Created:", output_path)
print(sample[["lab_name", "VALUE", "VALUEUOM", "FLAG", "input_text"]])