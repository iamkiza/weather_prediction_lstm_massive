from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)  # Apply CORS to the app

# Load the SavedModel
try:
    model = tf.saved_model.load("lstm_model")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Function to preprocess input data
def preprocess_input_data(data):
    features = ['Tn', 'Tx', 'RR']
    try:
        X = np.array([[float(day[feature]) for feature in features] for day in data['input']])
        X = X.reshape(-1, 10, 3).astype(np.float32)
        return X
    except KeyError as e:
        raise ValueError(f"Missing expected key in input data: {e}")
    except Exception as e:
        raise ValueError(f"Error processing input data: {e}")

@app.route('/')
def home():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'input' not in data:
            return jsonify({"error": "Invalid input data"}), 400

        print("Received JSON data:", data)

        # Preprocess input data
        X = preprocess_input_data(data)

        # Make predictions using the model
        predictions_lstm = model.signatures['serving_default'](lstm_input=X)['dropout_1'].numpy()

        # Prepare the response
        response = {
            'predictions_lstm': predictions_lstm.tolist(),
        }

        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)