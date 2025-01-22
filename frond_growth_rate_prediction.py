import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from datetime import datetime, timedelta

def load_data():
    # Path to the dataset (update the path as needed)
    data_path = 'sensors_irrigation_df.csv'
    data = pd.read_csv(data_path)

    # Feature engineering for the dataset
    data['vpd_rolling_mean'] = data['vpd (kPa)'].rolling(window=7, min_periods=1).mean()
    data['tdr_salt_trend'] = data['tdr_salt_80'].diff(periods=7)
    data['irrigation_vpd_interaction'] = data['irrigation'] * data['vpd (kPa)']
    data['irrigation_vpd_ratio'] = data['irrigation'] / (data['vpd (kPa)'] + 1e-6)
    data['salt_to_vpd_interaction'] = data['tdr_salt_80'] * data['vpd (kPa)']
    data['day_of_year'] = pd.to_datetime(data['date']).dt.dayofyear

    # Drop NaN values
    data = data.dropna()

    # Define feature columns and target column
    feature_columns_growth = [
        'irrigation', 'vpd (kPa)', 'tdr_salt_80', 'vpd_rolling_mean',
        'tdr_salt_trend', 'irrigation_vpd_interaction',
        'irrigation_vpd_ratio', 'salt_to_vpd_interaction', 'day_of_year'
    ]
    return data, feature_columns_growth

# -------------------------------
# Data Preprocessing Functions
# -------------------------------

def split_data(data, feature_columns, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    X = data[feature_columns]
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# -------------------------------
# Model Training Functions
# -------------------------------

def train_growth_model(data, feature_columns, target_column):
    """
    Train a predictive model for growth rate and return it.
    """
    X_train, X_test, y_train, y_test = split_data(data, feature_columns, target_column)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    model_evaluations = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

    return model, model_evaluations

def train_vpd_model(data):
    """
    Train a predictive model for VPD based on seasonal data.
    """
    feature_columns = ['day_of_year']
    target_column = 'vpd (kPa)'

    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_columns], data[target_column], test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    model_evaluations = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

    return model, model_evaluations

def train_tdr_model(data):
    """
    Train a predictive model for TDR Salt based on relevant features.
    """
    feature_columns = [
        'day_of_year', 'irrigation', 'vpd (kPa)',
        'vpd_rolling_mean', 'tdr_salt_trend', 'irrigation_vpd_interaction'
    ]
    target_column = 'tdr_salt_80'

    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_columns], data[target_column], test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.8,
        learning_rate=0.03,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    model_evaluations = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

    return model, model_evaluations

def predict_future_with_variable_irrigation(initial_input, irrigation_values, model, model_vpd, model_tdr, future_days=90):
    """
    Predict future growth dynamically with variable irrigation values, VPD, and TDR Salt predictions.
    """
    # Ensure exactly 9 irrigation values by extending the last value
    if len(irrigation_values) < 9:
        irrigation_values += [irrigation_values[-1]] * (9 - len(irrigation_values))
    elif len(irrigation_values) > 9:
        irrigation_values = irrigation_values[:9]

    # Expand irrigation values to cover 90 days (each value repeated 10 times)
    irrigation_schedule = []
    for value in irrigation_values:
        irrigation_schedule.extend([value] * 10)

    predictions = []
    current_input = initial_input.iloc[0].copy()

    for day in range(future_days):
        # Update irrigation dynamically
        current_input["irrigation"] = irrigation_schedule[day]

        # Update day_of_year and predict VPD
        current_input["day_of_year"] = current_input["date"].timetuple().tm_yday % 365
        current_input["vpd (kPa)"] = model_vpd.predict(pd.DataFrame({"day_of_year": [current_input["day_of_year"]]}))[0]

        # Calculate additional features for TDR Salt prediction
        current_input["vpd_rolling_mean"] = current_input["vpd (kPa)"]  # Assuming no rolling data for simplicity
        current_input["tdr_salt_trend"] = 0  # No trend for simplicity
        current_input["irrigation_vpd_interaction"] = current_input["irrigation"] * current_input["vpd (kPa)"]

        # Predict TDR Salt
        current_input["tdr_salt_80"] = model_tdr.predict(pd.DataFrame({
            "day_of_year": [current_input["day_of_year"]],
            "irrigation": [current_input["irrigation"]],
            "vpd (kPa)": [current_input["vpd (kPa)"]],
            "vpd_rolling_mean": [current_input["vpd_rolling_mean"]],
            "tdr_salt_trend": [current_input["tdr_salt_trend"]],
            "irrigation_vpd_interaction": [current_input["irrigation_vpd_interaction"]]
        }))[0]

        # Calculate additional features dynamically for growth prediction
        current_input["salt_to_vpd_interaction"] = current_input["tdr_salt_80"] * current_input["vpd (kPa)"]

        # Prepare features for growth prediction
        features = pd.DataFrame([{
            "irrigation": current_input["irrigation"],
            "vpd (kPa)": current_input["vpd (kPa)"],
            "tdr_salt_80": current_input["tdr_salt_80"],
            "vpd_rolling_mean": current_input["vpd_rolling_mean"],
            "tdr_salt_trend": current_input["tdr_salt_trend"],
            "irrigation_vpd_interaction": current_input["irrigation_vpd_interaction"],
            "irrigation_vpd_ratio": current_input["irrigation"] / (current_input["vpd (kPa)"] + 1e-6),
            "salt_to_vpd_interaction": current_input["salt_to_vpd_interaction"],
            "day_of_year": current_input["day_of_year"]
        }])

        predicted_growth = model.predict(features)[0]

        # Add prediction to results
        predictions.append({
            "date": current_input["date"],
            "irrigation": current_input["irrigation"],
            "vpd (kPa)": current_input["vpd (kPa)"],
            "tdr_salt_80": current_input["tdr_salt_80"],
            "Predicted Growth Rate": predicted_growth
        })

        # Update input data for the next day
        current_input["date"] += timedelta(days=1)

    return pd.DataFrame(predictions)

def run_prediction(irrigation_values, vpd, tdr_salt, start_date):
    """
    Collect user input, run the prediction, and display results in the table.
    """
    data, feature_columns_growth = load_data()
    try:
        irrigation_values = list(map(float, irrigation_values))

        initial_input = pd.DataFrame([{
            "date": start_date,
            "irrigation": irrigation_values[0],
            "vpd (kPa)": float(vpd),
            "tdr_salt_80": float(tdr_salt)
        }])

        model_growth, train_model_eval = train_growth_model(data, feature_columns_growth, 'frond_growth_rate')
        model_vpd, vpd_model_eval = train_vpd_model(data)
        model_tdr, tdr_model_eval = train_tdr_model(data)

        predictions_df = predict_future_with_variable_irrigation(initial_input, irrigation_values, model_growth, model_vpd, model_tdr, future_days=90)

        return predictions_df, vpd_model_eval, tdr_model_eval, train_model_eval

    except Exception as e:
        raise Exception(f"Exception occurred {e}")
