import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from frond_growth_rate_prediction import run_prediction
from sklearn.preprocessing import MinMaxScaler

# Load data
def load_data():
    file_path = 'sensors_irrigation_df.csv'  # Replace with your file path
    sensors_irrigation_df = pd.read_csv(file_path)
    sensors_irrigation_df['date'] = pd.to_datetime(sensors_irrigation_df['date'])  # Ensure date is datetime type
    sensors_irrigation_df['month'] = sensors_irrigation_df['date'].dt.month  # Extract month for analysis
    sensors_irrigation_df = sensors_irrigation_df[sensors_irrigation_df['month'].between(3, 10)]  # Filter months to March-October
    return sensors_irrigation_df

data = load_data()

# User-friendly field names
field_name_mapping = {
    "tensiometer_40": "Tensiometer at 40cm",
    "tensiometer_80": "Tensiometer at 80cm",
    "tdr_water_40": "TDR Water at 40cm",
    "tdr_water_80": "TDR Water at 80cm",
    "tdr_salt_40": "TDR Salt at 40cm",
    "tdr_salt_80": "TDR Salt at 80cm",
    "eto (mm/day)": "Evapotranspiration (mm/day)",
    "vpd (kPa)": "Vapor Pressure Deficit (kPa)",
    "frond_growth_rate": "Frond Growth Rate",
    "irrigation": "Irrigation Amount",
    "D": "Dropper Type D",
    "E": "Dropper Type E",
    "100": "100% Water Supplied",
    "50": "50% Water Supplied",
    "date": "Date"
}

# Define metric groups for Seasonal Trends
metric_groups = {
    "Tensiometer": ["tensiometer_40", "tensiometer_80"],
    "TDR Water": ["tdr_water_40", "tdr_water_80"],
    "TDR Salt": ["tdr_salt_40", "tdr_salt_80"]
}

# Reverse mapping for user-friendly names
reverse_mapping = {v: k for k, v in field_name_mapping.items()}

# Sidebar Navigation
st.sidebar.title("Navigation")
visualization_type = st.sidebar.radio("Select Page:", options=[
    "Welcome Page", "Seasonal Trends", "Heatmap (Correlations)", "Tree Health Visualization", "Combination Comparisons",
    "Correlation to Frond Growth Rate", "Run Prediction"
])

# Filters
st.sidebar.header("Filters")
selected_month = st.sidebar.multiselect(
    "Select Month(s):",
    options=range(3, 11),
    format_func=lambda x: datetime(1900, x, 1).strftime('%B'),
    default=range(3, 11)
)
selected_season = st.sidebar.selectbox("Select Season:", options=["None", "Spring", "Summer", "Autumn"])
season_months = {"Spring": [3, 4, 5], "Summer": [6, 7, 8], "Autumn": [9, 10]}
if selected_season != "None":
    selected_month = season_months[selected_season]

selected_dropper = st.sidebar.multiselect("Select Dropper Type:", options=['D', 'E'], default=['D', 'E'])
selected_percentage = st.sidebar.multiselect("Select Water Percentage:", options=['100', '50'], default=['100', '50'])

# Apply filters
filtered_data = data.copy()
filtered_data = filtered_data[filtered_data['month'].isin(selected_month)]
filtered_data = filtered_data[(filtered_data['D'].isin([1 if d == 'D' else 0 for d in selected_dropper])) | (filtered_data['E'].isin([1 if e == 'E' else 0 for e in selected_dropper]))]
filtered_data = filtered_data[(filtered_data['100'].isin([1 if p == '100' else 0 for p in selected_percentage])) | (filtered_data['50'].isin([1 if p == '50' else 0 for p in selected_percentage]))]

if visualization_type == "Welcome Page":
    st.title("Welcome to the Interactive Irrigation Data App")

    col1, col2 = st.columns([2, 1])  # Adjust the ratio to control column widths

    with col1:
        st.markdown(
            """
            ### Overview
            This app provides interactive visualizations and predictions for irrigation and frond growth rate data. 
            Use the sidebar to navigate through various sections, explore the data, and generate insights.

            ### Features:
            - **Seasonal Trends**: Visualize how key metrics change over time.
            - **Heatmaps**: Explore correlations between different parameters.
            - **Tree Health Visualizations**: Gain insights into tree health based on irrigation and other metrics.
            - **Predictions**: Predict frond growth rates using advanced models.

            ### Instructions:
            1. Use the filters in the sidebar to customize the data view.
            2. Navigate to a visualization type or run predictions using the top navigation menu.
            3. Analyze the insights and save any visualizations if needed.

            Enjoy exploring your data!
            """
        )

    with col2:
        st.image("welcome_image.png", width=250)  # Adjust the width as needed

# Visualization logic
elif visualization_type == "Seasonal Trends":
    st.header("Seasonal Trends")

    # Create two columns for side-by-side dropdowns
    col1, col2 = st.columns(2)

    with col1:
        time_scale = st.selectbox("Select Time Scale:", options=["Month", "Week", "Day", "All Data Points"], index=0)

    with col2:
        # Updated metric options to list all metrics individually
        metric_options = [
            "Tensiometer at 40cm", "Tensiometer at 80cm",
            "TDR Water at 40cm", "TDR Water at 80cm",
            "TDR Salt at 40cm", "TDR Salt at 80cm",
            "Evapotranspiration (mm/day)", "Vapor Pressure Deficit (kPa)",
            "Frond Growth Rate", "Irrigation Amount"
        ]
        selected_metrics = st.multiselect(
            "Select Metrics:",
            options=metric_options,
            default=[metric_options[0]]  # Default to the first metric for usability
        )

    # Reverse mapping for the selected metrics
    metrics = [reverse_mapping[selected_metric] for selected_metric in selected_metrics]
    scaled_df = filtered_data.copy()
    scaler = MinMaxScaler()
    for m in range(len(metrics)):  # normalize field for graph
        if filtered_data[metrics[m]].max() > 1:
            scaled_df["scaled " + metrics[m]] = scaler.fit_transform(filtered_data[metrics[m]].values.reshape(-1, 1))
            metrics[m] = "scaled " + metrics[m]

    # Group by the selected timescale
    if time_scale == "Month":
        trend_data = scaled_df.groupby('month')[metrics].mean().reset_index()
        x_axis = 'month'
        x_labels = [datetime(1900, m, 1).strftime('%B') for m in range(3, 11)]
    elif time_scale == "Week":
        scaled_df['week'] = scaled_df['date'].dt.isocalendar().week
        trend_data = scaled_df.groupby('week')[metrics].mean().reset_index()
        x_axis = 'week'
        x_labels = None
    elif time_scale == "Day":
        trend_data = scaled_df.groupby('date')[metrics].mean().reset_index()
        x_axis = 'date'
        x_labels = None
    else:  # All Data Points
        trend_data = scaled_df
        x_axis = 'date'
        x_labels = None

    # Create the line plot for multiple metrics
    fig = px.line(
        trend_data,
        x=x_axis,
        y=metrics,
        title=f'Seasonal Trends ({time_scale})',
        labels={x_axis: time_scale, **{metric: field_name_mapping.get(metric, metric) for metric in metrics}},
        template="plotly_white"
    )
    if x_labels:
        fig.update_xaxes(tickmode='array', tickvals=list(range(3, 11)), ticktext=x_labels)
    st.plotly_chart(fig)

elif visualization_type == "Heatmap (Correlations)":
    st.header("Heatmap (Correlations)")

    # List of columns available for selection
    available_columns = [
        "tensiometer_40", "tensiometer_80", "tdr_water_40", "tdr_water_80",
        "tdr_salt_40", "tdr_salt_80", "eto (mm/day)", "vpd (kPa)", "frond_growth_rate", "irrigation"
    ]
    column_labels = [field_name_mapping[col] for col in available_columns]

    # Multiselect dropdown for column selection
    selected_columns = st.multiselect(
        "Select Columns for Correlation Heatmap:",
        options=column_labels,
        default=column_labels  # Preselect all columns by default
    )

    # Map selected labels back to column names
    selected_columns_mapped = [reverse_mapping[label] for label in selected_columns]

    # Filter the data to the selected columns
    correlation_data = filtered_data[selected_columns_mapped]

    # Compute the correlation matrix
    corr = correlation_data.corr()
    corr.columns = [field_name_mapping.get(c, c) for c in corr.columns]
    corr.index = [field_name_mapping.get(c, c) for c in corr.index]

    # Create the heatmap
    fig = px.imshow(
        corr,
        title='Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        labels={'color': 'Correlation'},
        template="plotly_white",
        width=1000,
        height=800
    )

    # Display the heatmap
    st.plotly_chart(fig)


elif visualization_type == "Tree Health Visualization":
    st.header("Tree Health Visualization")
    tree_metric = st.selectbox("Select Metric for Tree Health:", options=[
        "Tensiometer at 40cm", "Tensiometer at 80cm",
        "TDR Water at 40cm", "TDR Water at 80cm",
        "TDR Salt at 40cm", "TDR Salt at 80cm",
        "Evapotranspiration (mm/day)", "Vapor Pressure Deficit (kPa)",
        "Irrigation Amount"
    ])
    metric_key = reverse_mapping.get(tree_metric, "frond_growth_rate")
    growth_key = reverse_mapping.get("frond_growth_rate", 0)
    sensor_data = filtered_data.groupby('month')[metric_key].mean().reset_index()
    health_data = filtered_data.groupby('month')["frond_growth_rate"].mean().reset_index()
    tmp_df = pd.merge(sensor_data, health_data, on='month')

    fig = px.scatter(
        tmp_df,
        x='month',
        y=metric_key,
        size="frond_growth_rate",
        color="frond_growth_rate",
        title=f'Tree Frond growth rate based on - {tree_metric}',
        labels={'month': 'Month', metric_key: tree_metric, "frond_growth_rate": "Frond Growth Rate"},
        template="plotly_white"
    )
    fig.update_xaxes(tickmode='array', tickvals=list(range(3, 11)),
                     ticktext=[datetime(1900, m, 1).strftime('%B') for m in range(3, 11)])
    st.plotly_chart(fig)

elif visualization_type == "Combination Comparisons":
    st.header("Combination Comparisons")

    # radio selection for plot type (box plot, scatter plot, trend line)
    plot_type = st.radio(
        "Select Plot Type:",
        ["Box Plot", "Scatter Plot", "Trend Line"],
        captions=[
            "Distributions of combinations",
            "Relationships between two fields",
            "Trends of sensors of time",
        ],
        horizontal=True
    )

    # Define possible combinations including all 8 groups
    def create_combinations(row):
        combinations = []
        if row['D'] == 1:
            combinations.append("Only D")
        if row['E'] == 1:
            combinations.append("Only E")
        if row['50'] == 1:
            combinations.append("Only 50")
        if row['100'] == 1:
            combinations.append("Only 100")
        if row['D'] == 1 and row['100'] == 1:
            combinations.append("100% & D")
        if row['E'] == 1 and row['100'] == 1:
            combinations.append("100% & E")
        if row['D'] == 1 and row['50'] == 1:
            combinations.append("50% & D")
        if row['E'] == 1 and row['50'] == 1:
            combinations.append("50% & E")
        return combinations

    # Create a unified "Combination" column as a list of combinations
    filtered_data['Combination'] = filtered_data.apply(create_combinations, axis=1)

    # Expand the data so each row corresponds to a single combination
    expanded_data = filtered_data.explode('Combination')

    # Available combinations for selection
    combinations_map = [
        "100% & D", "100% & E", "50% & D", "50% & E",
        "Only D", "Only E", "Only 50", "Only 100"
    ]

    # Select multiple combinations for scatter plot
    selected_combinations = st.multiselect(
        "Select Combinations:",
        options=combinations_map,
        default=combinations_map[:4]
    )

    # Filter data based on selected combinations
    filtered_combinations = expanded_data[expanded_data['Combination'].isin(selected_combinations)]

    # Box plot
    if plot_type == "Box Plot":
        y_axis = st.selectbox("Select Sensor Measurement For Combination:", options=[field_name_mapping[k] for k in [
            "frond_growth_rate", "tensiometer_40", "tensiometer_80", "tdr_water_40", "tdr_water_80"]])
        y_axis = reverse_mapping[y_axis]

        fig = px.box(
            filtered_combinations,
            x='Combination',
            y=y_axis,
            color='Combination',
            title=f'Comparison of {field_name_mapping[y_axis]} Across Combinations',
            labels={'Combination': 'Combination', y_axis: field_name_mapping[y_axis]},
            template="plotly_white"
        )
        st.plotly_chart(fig)

    # Scatter plot
    elif plot_type == "Scatter Plot":
        # Allow user to select X and Y axes for scatter plot
        x_axis_options = [
            'date', 'tensiometer_40', 'tensiometer_80', 'tdr_water_40',
            'tdr_water_80', 'frond_growth_rate', 'eto (mm/day)', 'vpd (kPa)', 'irrigation', "tdr_salt_40", "tdr_salt_80"
        ]
        x_axis = st.selectbox("Select X-Axis Measurement:", options=[field_name_mapping[k] for k in x_axis_options], index=0)
        x_axis_options.remove(reverse_mapping[x_axis])  # have it so the user cant choose the same field for x and y axis
        y_axis = st.selectbox("Select Y-Axis Measurement:", options=[field_name_mapping[k] for k in x_axis_options], index=1)

        # Normalize the date column for color mapping
        filtered_combinations['date_numeric'] = (
                filtered_combinations['date'] - filtered_combinations['date'].min()).dt.days

        # Plot the scatter chart with colors for categories and time progression
        fig = px.scatter(
            filtered_combinations,
            x=reverse_mapping[x_axis],
            y=reverse_mapping[y_axis],
            color='Combination',  # Use color to differentiate categories
            title=f'Scatter Plot: {x_axis} vs {y_axis}',
            labels={
                reverse_mapping[x_axis]: x_axis,
                reverse_mapping[y_axis]: y_axis,
                'date_numeric': 'Date (Oldest to Newest)',
                'Combination': 'Legend (Colors)'
            },
            color_discrete_sequence=px.colors.qualitative.Set1,  # Use a predefined color palette
            template="plotly_white"
        )

        # Update the layout to position legends
        fig.update_layout(
            legend=dict(
                yanchor="top",  # Anchor at the top
                y=1,  # Position at the top
                xanchor="right",  # Align to the left
                bgcolor="gray",  # Optional: Set a white background
                bordercolor="black",  # Optional: Add a black border
                x=5  # Slight left margin
            )
        )

        st.plotly_chart(fig)

    # Trend line plot
    elif plot_type == "Trend Line":
        y_axis = st.selectbox("Select Y-Axis:", options=[field_name_mapping[k] for k in [
            "frond_growth_rate", "tensiometer_40", "tensiometer_80", "tdr_water_40", "tdr_water_80"]])
        y_axis = reverse_mapping[y_axis]

        # Create a trend line for each combination group over time
        fig = px.line(
            filtered_combinations,
            x='date',
            y=y_axis,
            color='Combination',
            title=f'Trend Line of {field_name_mapping[y_axis]} Over Time',
            labels={'date': 'Date', y_axis: field_name_mapping[y_axis]},
            template="plotly_white"
        )
        fig.update_traces(opacity=0.6)
        st.plotly_chart(fig)

elif visualization_type == "Correlation to Frond Growth Rate":
    st.header("Correlation to Frond Growth Rate")

    # List of columns available for selection
    available_columns = [
        "tensiometer_40", "tensiometer_80", "tdr_water_40", "tdr_water_80",
        "tdr_salt_40", "tdr_salt_80", "eto (mm/day)", "vpd (kPa)", "irrigation"
    ]
    column_labels = [field_name_mapping[col] for col in available_columns]

    # Multiselect dropdown for column selection
    selected_columns = st.multiselect(
        "Select Columns to Combine and Correlate with Frond Growth Rate:",
        options=column_labels,
        default=[]  # No columns selected by default
    )

    if selected_columns:
        # Map selected labels back to column names
        selected_columns_mapped = [reverse_mapping[label] for label in selected_columns]

        # Create a combined feature as the average of the selected columns
        filtered_data['combined_feature'] = filtered_data[selected_columns_mapped].mean(axis=1)

        # Compute the correlation of the combined feature with `frond_growth_rate`
        correlation_value = filtered_data[['combined_feature', 'frond_growth_rate']].corr().iloc[0, 1]

        # Display the correlation value
        st.subheader(f"Combined Correlation with Frond Growth Rate: {correlation_value:.2f}")

        # Line plot to visualize the relationship
        fig = px.scatter(
            filtered_data,
            x='combined_feature',
            y='frond_growth_rate',
            trendline="ols",
            title='Combined Feature vs. Frond Growth Rate',
            labels={'combined_feature': 'Combined Feature', 'frond_growth_rate': 'Frond Growth Rate'},
            template="plotly_white"
        )
        st.plotly_chart(fig)

    else:
        st.write("Select at least one column to calculate the correlation.")

elif visualization_type == "Run Prediction":
    st.header("Run Frond Growth Rate Prediction")

    # Input fields for prediction
    irrigation_values = st.text_input(
        "Irrigation Values (Comma-separated liters per day):",
        placeholder="e.g., 5, 7, 6"
    )
    vpd = st.number_input(
        "Vapor Pressure Deficit (VPD) Today:",
        min_value=0.0, step=0.1, format="%.2f"
    )
    tdr_salt_80 = st.number_input(
        "TDR Salt at 80cm:",
        min_value=0.0, step=0.1, format="%.2f"
    )
    prediction_start_date = st.date_input(
        "Prediction Start Date:",
        value=datetime.today()
    )

    # Run prediction logic
    if st.button("Run Prediction"):
        try:
            # Parse irrigation values
            irrigation_values_list = [float(x.strip()) for x in irrigation_values.split(",")]

            # Call the prediction function
            predictions, vpd_model_eval, tdr_model_eval, train_model_eval = run_prediction(
                irrigation_values=irrigation_values_list,
                vpd=vpd,
                tdr_salt=tdr_salt_80,
                start_date=prediction_start_date
            )

            st.title("Display DataFrame in Streamlit")
            st.dataframe(predictions)  # Interactive table

            # Display model evaluations
            st.subheader("Model Evaluation")
            eval_data = {
                "Metric": ["MAE", "MSE", "R2"],
                "Frond growth rate prediction model": [train_model_eval["MAE"], train_model_eval["MSE"], train_model_eval["R2"]],
                "VPD Model": [vpd_model_eval["MAE"], vpd_model_eval["MSE"], vpd_model_eval["R2"]],
                "TDR Model": [tdr_model_eval["MAE"], tdr_model_eval["MSE"], tdr_model_eval["R2"]]
            }
            eval_df = pd.DataFrame(eval_data)
            st.table(eval_df)  # Static table

        except Exception as e:
            st.error(f"Error running prediction: {e}")

