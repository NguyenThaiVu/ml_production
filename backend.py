import configparser
import numpy as np
import joblib

from flask import Flask, jsonify, request


# Load config file
config = configparser.ConfigParser()
config.read('config.ini')
labels = config['data']['labels'].split(',')

clf = None


app = Flask(__name__)

def load_trained_model(model_name):
    path_trained_model = None
    clf = None

    if str(model_name).lower() == "decision tree":
        path_trained_model = config['model']['save_model_path']
        clf = joblib.load(path_trained_model)
    else:
        clf = joblib.load(path_trained_model)

    return clf


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the integer from the request data
        data = request.get_json()

        model_name = data.get('model_name')
        clf = load_trained_model(model_name)

        feature1 = data.get('feature1')
        feature2 = data.get('feature2')
        feature3 = data.get('feature3')
        feature4 = data.get('feature4')
        
        X = np.array([feature1, feature2, feature3, feature4])
        if len(X.shape) < 2:
            X = np.expand_dims(X, axis=0)
        
        prediction = clf.predict(X)[0]
        prediction = labels[prediction]

        return jsonify({'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':

    # Run Flask app
    app.run(debug=True)
