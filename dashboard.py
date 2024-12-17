import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.express as px

# Load pre-trained KNN model and preprocessing pipeline
knn_model = joblib.load('trained_knn_model.pkl')
encoder = joblib.load('target_encoder.pkl')

# Load data
data = pd.read_csv("C:/Users/User/Desktop/utem_pj/Train.csv")
data['BizDate'] = pd.to_datetime(data['BizDate'])  # Ensure BizDate is in datetime format

# Define a function to compute features dynamically
def generate_features(selected_date, loc_group, sub_dept, description, unit_price):
    day_of_week = selected_date.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    holiday_period = "Not in Holiday Period"  # Replace with holiday logic if needed

    features = {
        "Qty": 1,  # Placeholder; you may want to allow user input
        "DayOfWeek": day_of_week,
        "HolidayPeriod": holiday_period,
        "IsWeekend": is_weekend,
        "Loc_group": loc_group,
        "UnitPrice": unit_price,
        "Dept": "General",  # Replace with appropriate default
        "SubDept": sub_dept,
        "Category": "General",  # Replace with appropriate default
    }
    return pd.DataFrame([features])

# Streamlit UI
st.markdown("""
<div style="background-color:#2C3539;padding:10px;border-radius:5px;">
    <h1 style="text-align:center;color:white;">Dynamic Pricing Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# 1. Date Selection
selected_date = st.date_input("Select Date", value=datetime.today())
st.write(f"Selected Date: {selected_date}")

# 2. Branch Selection
branches = data['Loc_group'].unique()
selected_branch = st.selectbox("Select Branch", branches, index=0, help="Select a branch for analysis", key="branch")

# 3. Product Type Selection
product_types = data['SubDept'].unique()
selected_type = st.selectbox("Select Product Type", product_types, index=0, help="Select a product category", key="product_type")

# 4. Item Selection
filtered_items = data[data['SubDept'] == selected_type]['Description'].unique()
selected_item = st.selectbox("Select Item", filtered_items, index=0, help="Select a specific item to analyze", key="item")

# Get unit price for the selected item
item_data = data[(data['Description'] == selected_item) & (data['Loc_group'] == selected_branch)]

if not item_data.empty:
    unit_price = item_data['UnitPrice'].iloc[0]
else:
    st.error("No data available for the selected item and branch. Using default unit price.")
    unit_price = 10.0


# st.write(f"Unit Price: {unit_price:.2f} MYR")

# Generate features for prediction
features = generate_features(selected_date, selected_branch, selected_type, selected_item, unit_price)

# Apply preprocessing using encoder
features_preprocessed = encoder.transform(features)

# Predict using the trained model
prediction = knn_model.predict(features_preprocessed)
predicted_discount = prediction[0]

# Calculate Predicted Price
predicted_price = unit_price * (1 - predicted_discount / 100)

# Colorful Box for Unit Price and Predicted Discount, side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="background-color:#3B3131;padding:10px;border-radius:5px;">
        <h3 style="text-align:center;color:white;">Unit Price: {unit_price:.2f} MYR</h3>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background-color:#3B3131;padding:10px;border-radius:5px;">
        <h3 style="text-align:center;color:white;">Predicted Discount: {prediction[0]:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

# Generate features for prediction
features = generate_features(selected_date, selected_branch, selected_type, selected_item, unit_price)

# Apply preprocessing using encoder
features_preprocessed = encoder.transform(features)

# Predict using the trained model
prediction = knn_model.predict(features_preprocessed)
predicted_discount = prediction[0]

# Calculate Predicted Price
predicted_price = unit_price * (1 - predicted_discount / 100)

# Display the predicted price in a highlighted box
st.markdown(f"""
<div style="background-color:#CA762B;padding:10px;border-radius:5px;margin-top:10px;">
    <h2 style="text-align:center;color:white;">Predicted Price: {predicted_price:.2f} MYR</h2>
</div>
""", unsafe_allow_html=True)


# Header for Discount Pattern Section
st.markdown("""
<div style="background-color:#2C3539;padding:10px;border-radius:5px;margin-top:20px;">
    <h2 style="text-align:center;">Discount Pattern by Day of the Week</h2>
</div>
""", unsafe_allow_html=True)

# Filter data for the selected item and product type
discount_data = data[data['Description'] == selected_item]

# Group by DayOfWeek and calculate average discount percentage
discount_trend = discount_data.groupby('DayOfWeek')['DiscountPercentage'].mean().reset_index()

# Plot discount patterns by day of the week
fig2 = px.bar(
    discount_trend, 
    x='DayOfWeek', 
    y='DiscountPercentage', 
    title=f"Discount Patterns for {selected_item} by Day of the Week",
    labels={"DiscountPercentage": "Discount (%)", "DayOfWeek": "Day of the Week"},
    color='DayOfWeek',
    category_orders={"DayOfWeek": [4, 1, 3, 6, 5, 0, 2]}  # Ensure the days are in the correct order
)
st.plotly_chart(fig2)