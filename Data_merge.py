from jupyter_core.migrate import regex
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV


""" CREATE A TOTAL DATA TABLE WITH ALL THE DATA AND ADD RELEVANT COLUMNS"""


# Function to read sensor data and organize it
def process_sensor_data(file_path, sensor_type, start_date=None, normalize=False):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
    if start_date:
        data = data[data['date'] >= start_date]
    long_data = data.melt(
        id_vars=['date'],
        var_name='tree',
        value_name=sensor_type
    )
    long_data = long_data.sort_values(by=['date', 'tree']).reset_index(drop=True)

    # Normalize the sensor_type column if requested
    if normalize:
        max_value = long_data[sensor_type].max()
        if max_value > 0:  # Avoid division by zero
            long_data[sensor_type] = long_data[sensor_type] / max_value

    return long_data


# Paths to sensor data files
file_path_tensiometer_40 = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/tensiometer 40.csv"
file_path_tensiometer_80 = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/tensiometer 80.csv"
file_path_frond = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/frond data table.csv"
file_path_dendrometer = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/dendrometer data table.csv"
file_path_tdr_water_40 = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/tdr water 40.csv"
file_path_tdr_water_80 = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/tdr water 80.csv"
file_path_tdr_salt_40 = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/tdr salt 40.csv"
file_path_tdr_salt_80 = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/tdr salt 80.csv"
file_path_sap_flow = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/sap flow table data.csv"
file_path_eto_vpd = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/eto vpd data table.csv"
irrigation_data_path = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/daily irrigation data.csv"
output_path = r"/Users/evyataryatir/Desktop/STARSHIP ARAVA RND/3rd_try_python project/clean_data/final_table_with_all_sensors_and_irrigation.csv"

# Process sensor data - normalize numeric values
tensiometer_40_data = process_sensor_data(file_path_tensiometer_40, 'tensiometer_40', normalize=True)
tensiometer_80_data = process_sensor_data(file_path_tensiometer_80, 'tensiometer_80', normalize=True)
start_date = pd.Timestamp('2024-03-01')  # Start date for frond data
frond_data = process_sensor_data(file_path_frond, 'frond', start_date=start_date)
dendrometer_data = process_sensor_data(file_path_dendrometer, 'dendrometer', normalize=True)
tdr_water_40_data = process_sensor_data(file_path_tdr_water_40, 'tdr_water_40', normalize=True)
tdr_water_80_data = process_sensor_data(file_path_tdr_water_80, 'tdr_water_80', normalize=True)
tdr_salt_40_data = process_sensor_data(file_path_tdr_salt_40, 'tdr_salt_40', normalize=True)
tdr_salt_80_data = process_sensor_data(file_path_tdr_salt_80, 'tdr_salt_80', normalize=True)
sap_flow_data = process_sensor_data(file_path_sap_flow, 'sap_flow', normalize=True)

# Read eto and vpd data
eto_vpd_data = pd.read_csv(file_path_eto_vpd)
eto_vpd_data['date'] = pd.to_datetime(eto_vpd_data['dt'], format='%d/%m/%Y')
eto_vpd_data = eto_vpd_data.drop(columns=['dt'])

# Combine all data
all_data = pd.merge(tensiometer_40_data, tensiometer_80_data, on=['date', 'tree'], how='outer')
all_data = pd.merge(all_data, frond_data, on=['date', 'tree'], how='outer')
all_data = pd.merge(all_data, dendrometer_data, on=['date', 'tree'], how='outer')
all_data = pd.merge(all_data, tdr_water_40_data, on=['date', 'tree'], how='outer')
all_data = pd.merge(all_data, tdr_water_80_data, on=['date', 'tree'], how='outer')
all_data = pd.merge(all_data, tdr_salt_40_data, on=['date', 'tree'], how='outer')
all_data = pd.merge(all_data, tdr_salt_80_data, on=['date', 'tree'], how='outer')
all_data = pd.merge(all_data, sap_flow_data, on=['date', 'tree'], how='outer')
all_data = pd.merge(all_data, eto_vpd_data, on='date', how='left')

# Add 'frond growth rate' column
all_data['frond_growth_rate'] = all_data.groupby('tree')['frond'].diff()

# Read irrigation data
irrigation_data = pd.read_csv(irrigation_data_path)
irrigation_data['date'] = pd.to_datetime(irrigation_data['date'], format='%d/%m/%Y')

# Add irrigation columns based on tree group
all_data['irrigation'] = None
for index, row in irrigation_data.iterrows():
    all_data.loc[(all_data['tree'].str.contains('50', na=False)) & (all_data['date'] == row['date']), 'irrigation'] = \
    row['50% (L)']
    all_data.loc[(all_data['tree'].str.contains('100', na=False)) & (all_data['date'] == row['date']), 'irrigation'] = \
    row['100% (L)']

# Add irrigation calculations
kc = 1.25
all_data['eto (mm/day)'] = pd.to_numeric(all_data['eto (mm/day)'], errors='coerce')
all_data['irrigation'] = pd.to_numeric(all_data['irrigation'], errors='coerce')
all_data['irrigation calc (kc=1)'] = all_data['irrigation'] / (64 * all_data['eto (mm/day)'])
all_data['adjusted_irrigation'] = all_data['irrigation'] / (64 * all_data['eto (mm/day)'] * kc)

all_data['D'] = all_data['tree'].apply(lambda x: int('D' in x))  # 1 for 'D' in tree, 0 otherwise
all_data['E'] = all_data['tree'].apply(lambda x: int('E' in x))  # 1 for 'E' in tree, 0 otherwise
all_data['100'] = all_data['tree'].apply(lambda x: int('100' in x))  # 1 for '100' in tree, 0 otherwise
all_data['50'] = all_data['tree'].apply(lambda x: int('50' in x))  # 1 for '50' in tree, 0 otherwise

# Drop the 'tree' column
all_data = all_data.drop(columns=['tree'])
all_data = all_data.drop(columns=['sap_flow'])
all_data = all_data.drop(columns=['dendrometer'])
all_data = all_data.drop(columns=['adjusted_irrigation'])
all_data = all_data.drop(columns=['irrigation calc (kc=1)'])
all_data = all_data.drop(columns=['frond'])

# Remove rows where 'frond_growth_rate' is negative
all_data = all_data[all_data['frond_growth_rate'] >= 0]
all_data = all_data[all_data['irrigation'] != 0]

# Sort rows by date
all_data = all_data.sort_values(by='date').reset_index(drop=True)
# Drop rows with any NaN values
all_data = all_data.dropna()

# Save final table
all_data.to_csv(output_path, index=False)

print(
    f"Final table with all sensors, irrigation, eto/vpd, 'frond growth rate', and irrigation calculations saved to: {output_path}")
print(all_data.sample(5))

"""MODEL"""
def preprocess_data(data, relevant_columns):
    """
    Filters out rows with missing values for relevant columns.

    Args:
        data (pd.DataFrame): The full dataset.
        relevant_columns (list): Columns to check for missing values.

    Returns:
        pd.DataFrame: Filtered dataset.
    """
    return data.dropna(subset=relevant_columns)


def split_data(data, feature_columns, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset.
        feature_columns (list): Feature column names.
        target_column (str): Target column name.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    X = data[feature_columns]
    Y = data[target_column]
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)


def evaluate_model(name, y_true, y_pred):
    """
    Evaluates the model's performance and prints metrics.

    Args:
        name (str): Model name.
        y_true (pd.Series): Actual target values.
        y_pred (pd.Series): Predicted target values.
    """
    print(f"--- {name} ---")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.2f}")
    print(f"R-squared (R2): {r2_score(y_true, y_pred):.2f}")


# def train_and_evaluate(models, x_train, x_test, y_train, y_test):
#     """
#     Trains and evaluates multiple models.
#
#     Args:
#         models (list): List of (name, model) tuples.
# net        x_train (pd.DataFrame): Training feature set.
#         x_test (pd.DataFrame): Testing feature set.
#         y_train (pd.Series): Training target values.
#         y_test (pd.Series): Testing target values.
#     """
#     for name, model in models:
#         print(f"Training {name}...")
#         model.fit(x_train, y_train)
#         y_pred = model.predict(x_test)
#         evaluate_model(name, y_test, y_pred)

def train_and_evaluate(models, x_train, x_test, y_train, y_test):
    """
    Trains and evaluates multiple models.

    Args:
        models (list): List of (name, model) tuples.
        x_train (pd.DataFrame): Training feature set.
        x_test (pd.DataFrame): Testing feature set.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Testing target values.
    """
    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        evaluate_model(name, y_test, y_pred)

        # Print feature importance if available
        if hasattr(model, "feature_importances_"):
            print(f"\n--- Feature Importance for {name} ---")
            importance = model.feature_importances_
            for feature, imp in zip(x_train.columns, importance):
                print(f"Feature: {feature}, Importance: {imp * 100:.2f}%")

        # Print coefficients for linear models
        elif hasattr(model, "coef_"):
            print(f"\n--- Coefficients for {name} ---")
            coefficients = model.coef_
            for feature, coef in zip(x_train.columns, coefficients):
                print(f"Feature: {feature}, Coefficient: {coef:.4f}")

# Filter out rows with missing values
#"eto (mm/day)","tensiometer_80", "tdr_water_80",
relevant_columns = [
    "irrigation", "frond_growth_rate", "vpd (kPa)","tdr_salt_80"
]
filtered_data = preprocess_data(all_data, relevant_columns)


# Define feature and target columns
target_column = "frond_growth_rate"
feature_columns = [col for col in relevant_columns if col != target_column]

# Split the data
x_train, x_test, y_train, y_test = split_data(filtered_data, feature_columns, target_column)

# Define models
models = [
    ("Random Forest", RandomForestRegressor(random_state=42, n_estimators=100)),
    ("Lasso Regression (CV)", LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5, random_state=42)),
    ("XGBoost", XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, enable_categorical=True)),
]

# Train and evaluate models
train_and_evaluate(models, x_train, x_test, y_train, y_test)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Extract relevant columns for clustering
clustering_data = filtered_data[["frond_growth_rate", "eto (mm/day)"]].dropna()

# Standardize the data for better clustering
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Elbow Method to Determine Optimal K
wcss = []  # Within-Cluster-Sum of Squares
k_range = range(1, 16)  # Test k from 1 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_data_scaled)
    wcss.append(kmeans.inertia_)


# Choose an optimal k based on the elbow point (e.g., visually determined or use a heuristic)
optimal_k = 4  # Example: Assume 3 clusters after observing the elbow plot

# Perform K-Means Clustering with Optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clustering_data["Cluster"] = kmeans.fit_predict(clustering_data_scaled)

# Add cluster labels back to the original data
filtered_data = filtered_data.copy()
filtered_data["Cluster"] = kmeans.predict(scaler.transform(filtered_data[["frond_growth_rate", "eto (mm/day)"]].dropna()))

# Save the clustered data to a new CSV
clustered_output_path = r"C:\Users\shenh\Desktop\data table\clustered_data.csv"
filtered_data.to_csv(clustered_output_path, index=False)

print(f"Clustered data saved to: {clustered_output_path}")

##########################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'date' is in datetime format
filtered_data['eto (mm/day)'] = pd.to_datetime(filtered_data['eto (mm/day)'])

# Create combinations of "D" and "E" with "50" and "100"
filtered_data['Group'] = (
    filtered_data['D'].map({1: "D", 0: ""}) +
    filtered_data['E'].map({1: "E", 0: ""}) +
    " and " +
    filtered_data['50'].map({1: "50", 0: ""}) +
    filtered_data['100'].map({1: "100", 0: ""})
)

# Filter out empty combinations
filtered_data = filtered_data[filtered_data['Group'].str.strip() != "and"]

##################################################


import joblib
#save the models after training
joblib.dump(RandomForestRegressor(random_state=42, n_estimators=100).fit(x_train, y_train), 'random_forest_model.pkl')
joblib.dump(LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5, random_state=42).fit(x_train, y_train), 'lasso_model.pkl')
joblib.dump(XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1).fit(x_train, y_train), 'xgb_model.pkl')

## main
import numpy as np
import pandas as pd
import joblib

def predict_frond_growth(tdr_salt_80, vpd, irrigation):
    """
    Predicts frond growth rate based on user inputs using pre-trained models.

    Args:
        tdr_salt_80 (float): Value for TDR Salt 80 sensor.
        vpd (float): Vapor Pressure Deficit (VPD) value.
        irrigation (float): Amount of irrigation (in liters).

    Returns:
        dict: Predictions from different models.
    """

    # Check inputs
    if not all(isinstance(val, (int, float)) for val in [tdr_salt_80, vpd, irrigation]):
        raise ValueError("All inputs must be numeric.")

    # Load pre-trained models
    rf_model = joblib.load("random_forest_model.pkl")
    lasso_model = joblib.load("lasso_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")

    # Load training data (used to compute mean and std for normalization)
    data_path = r"C:\Users\shenh\Desktop\data table\final_table_with_all_sensors_and_irrigation.csv"
    data = pd.read_csv(data_path)
    feature_columns = ["irrigation", "vpd (kPa)", "tdr_salt_80"]

    # Calculate mean and std for each feature
    means = data[feature_columns].mean()
    stds = data[feature_columns].std()

    # Normalize user input
    user_input = pd.DataFrame([[irrigation, vpd, tdr_salt_80]], columns=feature_columns)
    normalized_input = (user_input - means) / stds

    # Remove rows with NaN values
    normalized_input = normalized_input.dropna()

    # Check if there are still any missing values after cleaning
    if normalized_input.empty:
        raise ValueError("All input data contains missing values after cleaning. Please check your input.")

    # Generate predictions
    predictions = {
        "Random Forest": rf_model.predict(normalized_input)[0],
        "Lasso Regression": lasso_model.predict(normalized_input)[0],
        "XGBoost": xgb_model.predict(normalized_input.to_numpy())[0]
    }

    return predictions




##########################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'date' is in datetime format
filtered_data['eto (mm/day)'] = pd.to_datetime(filtered_data['eto (mm/day)'])

# Create combinations of "D" and "E" with "50" and "100"
filtered_data['Group'] = (
    filtered_data['D'].map({1: "D", 0: ""}) +
    filtered_data['E'].map({1: "E", 0: ""}) +
    " and " +
    filtered_data['50'].map({1: "50", 0: ""}) +
    filtered_data['100'].map({1: "100", 0: ""})
)

# Filter out empty combinations
filtered_data = filtered_data[filtered_data['Group'].str.strip() != "and"]

# Plot the data
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=filtered_data,
    x="eto (mm/day)",
    y="frond_growth_rate",
    hue="Group",
    palette="Set2",
    s=100,
    alpha=0.7
)

plt.title("Frond Growth Rate Over Time by Combined Groups")
plt.xlabel("eto (mm/day)")
plt.ylabel("Frond Growth Rate")
plt.legend(title="Group")
plt.grid()
plt.show()

##################################################

# Plot the data with 'tdr_salt_40' on the x-axis
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=filtered_data,
    x="tdr_salt_40",  # Change x-axis to 'tdr_salt_40'
    y="frond_growth_rate",
    hue="Group",
    palette="Set2",
    s=100,
    alpha=0.7
)

plt.title("Frond Growth Rate vs TDR Salt 40 by Combined Groups")
plt.xlabel("TDR Salt 40")
plt.ylabel("Frond Growth Rate")
plt.legend(title="Group")
plt.ylim(4,8)
plt.grid()
plt.show()

###################################################

# Plot the data with 'tdr_salt_40' on the x-axis
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=filtered_data,
    x="tdr_water_40",  # Change x-axis to 'tdr_salt_40'
    y="frond_growth_rate",
    hue="Group",
    palette="Set2",
    s=100,
    alpha=0.7
)

plt.title("Frond Growth Rate vs TDR Water 40 by Combined Groups")
plt.xlabel("TDR Water 40")
plt.ylabel("Frond Growth Rate")
plt.legend(title="Group")
plt.ylim(4,8)
plt.grid()
plt.show()
##########################################################
import matplotlib.ticker as ticker

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=filtered_data,
    x="frond_growth_rate",
    y="eto (mm/day)",
    hue="Group",
    palette="Set2",
    s=100,
    alpha=0.7
)

plt.title("Frond Growth Rate vs eto (mm/day) Rate by Combined Groups")
plt.xlabel("Frond Growth Rate")
plt.ylabel("eto (mm/day)")
plt.legend(title="Group")

# Set the y-axis scale and ticks
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))  # Adjust interval as needed
plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # Add minor ticks
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))  # Format labels as float with 2 decimals

plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure 'date' is in datetime format
filtered_data['date'] = pd.to_datetime(filtered_data['date'])

# Assign groups based on 'D', 'E', '50', and '100'
filtered_data['Group'] = (
    filtered_data['D'].map({1: "D", 0: ""}) +
    filtered_data['E'].map({1: "E", 0: ""}) +
    " and " +
    filtered_data['50'].map({1: "50", 0: ""}) +
    filtered_data['100'].map({1: "100", 0: ""})
)

# Filter out empty groups
filtered_data = filtered_data[filtered_data['Group'].str.strip() != "and"]

# Plot the data with Seaborn
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=filtered_data,
    x="date",
    y="frond_growth_rate",
    hue="Group",
    palette="Set1",
    marker="o"
)

# Add title and labels
plt.title("Frond Growth Rate Over Time by Groups")
plt.xlabel("Date")
plt.ylabel("Frond Growth Rate")

# Customize the legend and grid
plt.legend(title="Group")
plt.xticks(rotation=45)
plt.grid(linestyle="--", linewidth=0.5)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Shift data for "D and 100" and "D and 50" groups by adding 0.15 to their frond growth rate
filtered_data.loc[filtered_data['Group'] == "D and 100", 'frond_growth_rate'] += 0.15
filtered_data.loc[filtered_data['Group'] == "D and 50", 'frond_growth_rate'] += 0.15

# Re-plot the graph after the shift
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=filtered_data,
    x="date",
    y="frond_growth_rate",
    hue="Group",
    palette="Set1",
    marker="o"
)

# Add title and axis labels
plt.title("Frond Growth Rate Over Time by Groups (Adjusted)", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Frond Growth Rate", fontsize=14)

# Customize legend and grid
plt.legend(title="Group", fontsize=12)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(linestyle="--", linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

