from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def get_message():
    try:
        # Get the integer from the request data
        data = request.get_json()
        feature1 = data.get('feature1')
        feature2 = data.get('feature2')
        result = feature1 + feature2

        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
