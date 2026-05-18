import os

import pandas as pd

input_path = os.getenv("LABS_GENERATION")
output_path = os.getenv("SILVER_RULE")

df = pd.read_csv(input_path)

def generate_description(row):
    lab_name = row.get("lab_name", "laboratory test")
    value = row.get("VALUE", "unknown")
    unit = row.get("VALUEUOM", "")
    flag = row.get("FLAG", "")
    diagnosis = row.get("DIAGNOSIS", "the patient's clinical condition")

    if pd.isna(flag) or str(flag).strip() == "":
        flag_text = "The result is not marked as abnormal in the available data."
    else:
        flag_text = f"The result is marked as {flag}."

    return (
        f"The patient's {lab_name} result is {value} {unit}. "
        f"{flag_text} "
        f"This finding should be interpreted together with the patient's clinical context, "
        f"including the admission diagnosis of {diagnosis}."
    )

df["target_text"] = df.apply(generate_description, axis=1)

df.to_csv(output_path, index=False)

print("Done.")
print("Created:", output_path)
print(df[["lab_name", "VALUE", "VALUEUOM", "FLAG", "target_text"]].head())
#ne e silver standard ni treba LLM model za generiranje na target_text, ova e samo rule-based pristap baziran na dostapnite podatoci