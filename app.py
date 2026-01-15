import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and feature order
try:
    loaded_model = joblib.load('fire_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_feature_order = joblib.load('feature_order.pkl')

    # Add checks for loaded artifacts
    if not isinstance(loaded_model, (object)):
        st.error("Error: Loaded model is not a valid model object. fire_model.pkl might be corrupted.")
        st.stop()
    if not isinstance(loaded_scaler, (object)):
        st.error("Error: Loaded scaler is not a valid scaler object. scaler.pkl might be corrupted.")
        st.stop()
    if not isinstance(loaded_feature_order, list) or not all(isinstance(f, str) for f in loaded_feature_order):
        st.error("Error: Loaded feature order is not a list of strings. feature_order.pkl might be corrupted.")
        st.stop()

except FileNotFoundError:
    st.error("Model, scaler, or feature order files not found. Please ensure they are in the same directory as this app.py file.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during file loading: {e}")
    st.stop()

st.title('Forest Fire Occurrence Prediction')
st.write('Enter the values for the features to predict if a forest fire will occur.')

# Create input fields for each feature, based on the loaded_feature_order
# We'll create a dictionary to hold user inputs
user_input = {}

# Mapping for months and days from original notebook to Streamlit selection
month_map = {
    'jan': 'month_jan', 'feb': 'month_feb', 'mar': 'month_mar', 'apr': 'month_apr',
    'may': 'month_may', 'jun': 'month_jun', 'jul': 'month_jul', 'aug': 'month_aug',
    'sep': 'month_sep', 'oct': 'month_oct', 'nov': 'month_nov', 'dec': 'month_dec'
}
day_map = {
    'mon': 'day_mon', 'tue': 'day_tue', 'wed': 'day_wed', 'thu': 'day_thu',
    'fri': 'day_fri', 'sat': 'day_sat', 'sun': 'day_sun'
}

# Categorical inputs first for better UX
selected_month_name = st.selectbox('Month', list(month_map.keys()))
selected_day_name = st.selectbox('Day', list(day_map.keys()))

# Populate month and day columns with False initially
for month_col in [col for col in loaded_feature_order if col.startswith('month_')]:
    user_input[month_col] = False
for day_col in [col for col in loaded_feature_order if col.startswith('day_')]:
    user_input[day_col] = False

# Set selected month and day to True
if selected_month_name in month_map: # Check if key exists
    user_input[month_map[selected_month_name]] = True
if selected_day_name in day_map: # Check if key exists
    user_input[day_map[selected_day_name]] = True

# Numeric inputs
# Default values are set to the mean of the training data or reasonable defaults
user_input['X'] = st.slider('X (Spatial coordinate within the Beja region - 1 to 9)', 1, 9, 5)
user_input['Y'] = st.slider('Y (Spatial coordinate within the Beja region - 2 to 9)', 2, 9, 5)
user_input['FFMC'] = st.slider('FFMC (Fine Fuel Moisture Code - 18.7 to 96.2)', 18.0, 97.0, 90.0)
user_input['DMC'] = st.slider('DMC (Duff Moisture Code - 1.1 to 291.3)', 1.0, 292.0, 110.0)
user_input['DC'] = st.slider('DC (Drought Code - 7.9 to 860.6)', 7.0, 861.0, 500.0)
user_input['ISI'] = st.slider('ISI (Initial Spread Index - 0.0 to 56.1)', 0.0, 57.0, 9.0)
user_input['temp'] = st.slider('Temperature (Celsius - 2.2 to 33.3)', 2.0, 34.0, 18.0)
user_input['RH'] = st.slider('Relative Humidity (percent - 15 to 100)', 15, 100, 45)
user_input['wind'] = st.slider('Wind speed (km/h - 0.4 to 9.4)', 0.0, 10.0, 4.0)
user_input['rain'] = st.slider('Rain (mm/m2 - 0.0 to 6.4)', 0.0, 7.0, 0.0)
user_input['area'] = 0.0 # Area is excluded from prediction input

# Convert user inputs to a DataFrame
input_df = pd.DataFrame([user_input])

# Reorder columns to match the feature order used during training using reindex for robustness
final_input_df = input_df.reindex(columns=loaded_feature_order, fill_value=False)

# Convert boolean columns to int for scaling if the scaler was fitted on int/float
for col in final_input_df.select_dtypes(include='bool').columns:
    final_input_df[col] = final_input_df[col].astype(int)

# Scale the input data
scaled_input = loaded_scaler.transform(final_input_df)

if st.button('Predict Fire Occurrence'):
    prediction = loaded_model.predict(scaled_input)
    prediction_proba = loaded_model.predict_proba(scaled_input)

    if prediction[0] == 1:
        st.success(f"Prediction: Fire is likely to occur (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.info(f"Prediction: Fire is unlikely to occur (Probability: {prediction_proba[0][0]:.2f})")

    st.write("### Input Data Provided:")
    st.write(final_input_df)