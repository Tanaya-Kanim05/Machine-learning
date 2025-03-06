import streamlit as st
import pickle
import numpy as np

# Load the data from the pickle file
with open(r'C:\Users\kanim\OneDrive\Desktop\Tanaya kanim\machine learing\columns.pkl', 'rb') as file:
    data_columns = pickle.load(file)

locations = data_columns['locations']
area_types = data_columns['area_types']
availabilities = data_columns['availabilities']
columns = data_columns['data_columns']

# Load the trained model
with open(r'C:\Users\kanim\OneDrive\Desktop\Tanaya kanim\machine learing\model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_price(location, bhk, bath, balcony, sqft, area_type, availability):
    loc_index, area_index, avail_index = -1, -1, -1

    # Ensure the inputs are properly formatted
    location = location.strip().capitalize()
    area_type = area_type.strip().capitalize()
    availability = availability.strip().capitalize()

    # Check if the location exists in columns
    if location != 'Other' and location in columns:
        loc_index = int(np.where(np.array(columns) == location)[0][0])

    # Check if the area_type exists in columns
    if area_type != 'Super Built-Up Area' and area_type in columns:
        area_index = int(np.where(np.array(columns) == area_type)[0][0])

 # Check if the availability exists in columns
    if availability != 'Not Ready' and availability in columns:
        avail_index = int(np.where(np.array(columns) == availability)[0][0])

    # Initialize the input array
    x = np.zeros(len(columns))
    x[0] = bath
    x[1] = balcony
    x[2] = bhk
    x[3] = sqft

    # Set the corresponding indices to 1 if they exist
    if loc_index >= 0:
        x[loc_index] = 1
    if area_index >= 0:
        x[area_index] = 1
    if avail_index >= 0:
        x[avail_index] = 1

    # Predict the price
    return model.predict([x])[0]


# Streamlit App
st.title("Pune House Price Prediction")
st.write("Predict the price of a house based on its features.")

# Form inputs
location = st.selectbox("Location", options=["Other"] + locations)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, step=1)
bath = st.number_input("Number of Bathrooms", min_value=1, step=1)
balcony = st.number_input("Number of Balconies", min_value=0, step=1)
sqft = st.number_input("Total Square Footage", min_value=100, step=10)
area_type = st.selectbox("Area Type", options=area_types)
availability = st.selectbox("Availability", options=availabilities)

if st.button("Predict Price"):
    predicted_price = predict_price(location, bhk, bath, balcony, sqft, area_type, availability)
    st.success(f"The predicted price of the house is: ${predicted_price:,.2f}")