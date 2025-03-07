import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open(r'C:\Users\kanim\OneDrive\Desktop\Tanaya kanim\machine learing\ipl dataset regression\linear_regressor.okl', 'rb') as file:
    linear_regressor = pickle.load(file)

# Team options
teams = [
    'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders',
    'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

def predict_score(batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5):
    temp_array = []

    # One-hot encoding for batting team
    for team in teams:
        temp_array.append(1 if batting_team == team else 0)
    
    # One-hot encoding for bowling team
    for team in teams:
        temp_array.append(1 if bowling_team == team else 0)
    
    # Append numerical inputs
    temp_array.extend([overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5])
    temp_array = np.array([temp_array])
    
    # Make prediction
    predicted_score = int(linear_regressor.predict(temp_array)[0])
    return predicted_score

# Streamlit UI
st.title('IPL First Innings Score Predictor')

# User inputs
batting_team = st.selectbox('Select Batting Team', teams)
bowling_team = st.selectbox('Select Bowling Team', teams)
overs = st.number_input('Overs Completed (>=5)', min_value=5.0, max_value=20.0, step=0.1)
runs = st.number_input('Current Runs', min_value=0, max_value=300, step=1)
wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)
runs_in_prev_5 = st.number_input('Runs in Last 5 Overs', min_value=0, max_value=100, step=1)
wickets_in_prev_5 = st.number_input('Wickets in Last 5 Overs', min_value=0, max_value=10, step=1)

# Predict button
if st.button('Predict Score'):
    final_score = predict_score(batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5)
    st.success(f'Predicted Score Range: {final_score-10} to {final_score+5}')
