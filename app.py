import numpy as np
import streamlit as st
import streamlit as st
import requests


BASE_URL = r"http://localhost:5000"


def main():

    # Front end interface
    st.title("Simple AI service 💦")
    st.write("This application enables to classify the potability of water based on the water composition and water quality metrics")
    st.sidebar.write("Hello Thai Vu Nguyen")

    list_trained_model = ["decision tree", "logistic regression"]
    selected_option = st.selectbox("Select an option:", [None] + list_trained_model)

    # Create 4 text boxes for user input
    feature1 = st.number_input("sepal length (cm)")
    feature2 = st.number_input("sepal width (cm)")
    feature3 = st.number_input("petal length (cm):")
    feature4 = st.number_input("petal width (cm)")

    # Create a submit button
    if st.button("Predict"):

        if selected_option is None:
            st.warning(f"Please load Machine Learning model first!")

        else:            
            # POST request to Flask API
            url = f"{BASE_URL}/predict"
            input_features = {
                'model_name': selected_option,
                'feature1': float(feature1),
                'feature2': float(feature2),
                'feature3': float(feature3),
                'feature4': float(feature4)
            }
            response = requests.post(url, json=input_features)

            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()['prediction']
                st.success(f"Result prediction: {result}")
            else:
                st.error(f"Error: {response.status_code}")




if __name__ == "__main__":
    main()
