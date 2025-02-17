from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# Load the model
model = mlflow.sklearn.load_model("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the request
    data = request.get_json()
    input_data = pd.DataFrame(data)

    # Make predictions
    prediction = model.predict(input_data)

    # Return predictions as a JSON response
    return jsonify(predictions=prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
