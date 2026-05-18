import pandas as pd
import os

input_path = os.getenv("LABS_SAMPLE")
output_path = os.getenv("LABS_GENERATION")


if not os.path.exists(input_path):
    print("File not found:")
    print(input_path)
    exit()


df = pd.read_csv(input_path)

print("CSV loaded successfully.")
print("Number of rows:", len(df))
print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


def create_input(row):
    return f"""Patient sex: {row.get('GENDER', 'unknown')}
Admission type: {row.get('ADMISSION_TYPE', 'unknown')}
Admission diagnosis: {row.get('DIAGNOSIS', 'unknown')}

Laboratory test: {row.get('lab_name', 'unknown')}
Fluid: {row.get('fluid', 'unknown')}
Category: {row.get('category', 'unknown')}
Measured value: {row.get('VALUE', 'unknown')} {row.get('VALUEUOM', '')}
Abnormal flag: {row.get('FLAG', 'not available')}

Task: Generate a short medical explanation of this laboratory result."""


df["input_text"] = df.apply(create_input, axis=1)
df["target_text"] = ""

df.to_csv(output_path, index=False)

print("\nDone.")
print("New file created here:")
print(output_path)