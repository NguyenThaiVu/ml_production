import streamlit as st
import requests

# Streamlit frontend
st.title("Streamlit + Flask API Example")

input1 = st.number_input("sepal length (cm)")
input2 = st.number_input("sepal width (cm)")

# Button to trigger API request
if st.button("Predict"):
    # Send request to Flask API
    url = f"http://localhost:5000/predict"
    data = {
        'feature1': float(input1),
        'feature2': float(input2),
    }
    response = requests.post(url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()['result']
        st.success(f"Result prediction: {result}")
    else:
        st.error(f"Error: {response.status_code}")
