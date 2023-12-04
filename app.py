import streamlit as st
import configparser
import joblib
import numpy as np

def load_trained_model(selected_model, config):
    path_trained_model = None
    clf = None

    if str(selected_model).lower() == "decision tree":
        path_trained_model = config['model']['save_model_path']

    if path_trained_model == None:
        st.success(f"Please select model")
        clf = None
    else:
        clf = joblib.load(path_trained_model)
        st.success(f"Success load model: {selected_model}")

    return clf



def main():

    # Load config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    labels = config['data']['labels'].split(',')

    # Front end interface
    st.title("Simple AI service 💦")
    st.write("This application enables to classify the potability of water based on the water composition and water quality metrics")
    st.sidebar.write("Hello Thai Vu Nguyen")

    list_trained_model = ["decision tree", "logistic regression"]
    selected_option = st.selectbox("Select an option:", list_trained_model)

    if selected_option is not None:
        clf = load_trained_model(selected_option, config)


    # Create 4 text boxes for user input
    input1 = st.number_input("sepal length (cm)")
    input2 = st.number_input("sepal width (cm)")
    input3 = st.number_input("petal length (cm):")
    input4 = st.number_input("petal width (cm)")

    # Create a submit button
    if st.button("Predict"):
        X = np.array([input1, input2, input3, input4])
        if len(X.shape) < 2:
            X = np.expand_dims(X, axis=0)
        
        prediction = clf.predict(X)[0]
        st.success(f"This instance is a {labels[prediction]}")



if __name__ == "__main__":
    main()
