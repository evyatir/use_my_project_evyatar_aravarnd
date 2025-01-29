"""i create correlation data fram template with the important data (after feature engineering - no sap flow & no dendrometer"""


import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Set the width to a large value

# Define the column names
columns = [
    "T2", "T4", "T5", "T7", "T9", "T12", "T14", "T15",
    "D irr", "E irr", "50% water", "100% water",
    "D_50", "D_100", "E_50", "E_100",
    "sensor b", "sensor a", "row number"
]

# Create an empty DataFrame with the specified columns
df = pd.DataFrame(columns=columns)

# Data from the image
data = [
    {"row number": 1, "sensor b": "tdr_salt_80", "sensor a": "tdr_salt_40"},
    {"row number": 2, "sensor b": "tdr_water_40", "sensor a": "tdr_salt_40"},
    {"row number": 3, "sensor b": "tdr_water_80", "sensor a": "tdr_salt_40"},
    {"row number": 4, "sensor b": "tensiometer_40", "sensor a": "tdr_salt_40"},
    {"row number": 5, "sensor b": "tensiometer_80", "sensor a": "tdr_salt_40"},
    {"row number": 6, "sensor b": "frond_growth_rate", "sensor a": "tdr_salt_40"},
    {"row number": 7, "sensor b": "tdr_water_40", "sensor a": "tdr_salt_80"},
    {"row number": 8, "sensor b": "tdr_water_80", "sensor a": "tdr_salt_80"},
    {"row number": 9, "sensor b": "tensiometer_40", "sensor a": "tdr_salt_80"},
    {"row number": 10, "sensor b": "tensiometer_80", "sensor a": "tdr_salt_80"},
    {"row number": 11, "sensor b": "frond_growth_rate", "sensor a": "tdr_salt_80"},
    {"row number": 12, "sensor b": "tdr_water_80", "sensor a": "tdr_water_40"},
    {"row number": 13, "sensor b": "tensiometer_40", "sensor a": "tdr_water_40"},
    {"row number": 14, "sensor b": "tensiometer_80", "sensor a": "tdr_water_40"},
    {"row number": 15, "sensor b": "frond_growth_rate", "sensor a": "tdr_water_40"},
    {"row number": 16, "sensor b": "tensiometer_40", "sensor a": "tdr_water_80"},
    {"row number": 17, "sensor b": "tensiometer_80", "sensor a": "tdr_water_80"},
    {"row number": 18, "sensor b": "frond_growth_rate", "sensor a": "tdr_water_80"},
    {"row number": 19, "sensor b": "tensiometer_80", "sensor a": "tensiometer_40"},
    {"row number": 20, "sensor b": "frond_growth_rate", "sensor a": "tensiometer_40"},
    {"row number": 21, "sensor b": "frond_growth_rate", "sensor a": "tensiometer_80"},

]

# Convert data to a DataFrame and concatenate with the existing DataFrame
new_data = pd.DataFrame(data)
df = pd.concat([df, new_data], ignore_index=True)

# Display the DataFrame
print(df)

# Optionally save the DataFrame to a CSV file
output_path = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/corr_df_template.csv"
df.to_csv(output_path, index=False)
print(f"DataFrame with rows saved to: {output_path}")

