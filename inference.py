import os
import sys
import numpy as np
import joblib  # Assuming you've used joblib for model serialization



def load_model(model_path):
    """
    Load the trained machine learning model from a file.
    """
    model = joblib.load(model_path)
    return model

def predict(model, features):
    """
    Make predictions using the trained model.
    
    Args:
        model: The trained machine learning model.
        features: A numpy array or list containing the input features.
        
    Returns:
        float: The predicted output.
    """
    # Ensure features are in the right format (numpy array)
    features = np.array(features).reshape(1, -1)

    # Make predictions
    prediction = model.predict(features)

    return prediction[0]

if __name__ == "__main__":


    model_path = os.path.join('model', 'decision_tree_model.joblib')  # Replace with the actual path to your serialized model

    feature_1 = float(sys.argv[1])
    feature_2 = float(sys.argv[2])
    feature_3 = float(sys.argv[3])
    feature_4 = float(sys.argv[4])

    input_features = [feature_1, feature_2, feature_3, feature_4]  # Replace with your actual input features

    trained_model = load_model(model_path)

    # Make predictions
    result = predict(trained_model, input_features)

    print("Prediction:", result)
