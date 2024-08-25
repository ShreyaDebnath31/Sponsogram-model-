from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app) 


with open('light_gbm2.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    followers = data['followers']
    totalMetrics = data['totalMetrics']
    engagementMetrics = data['engagementMetrics']
    
    input_features = np.array([[followers, totalMetrics, engagementMetrics]])
    prediction = model.predict(input_features)
    
    budget = data['budget']
    roi = (prediction[0] - budget) / budget  
    
    return jsonify({"roi": float(roi)})  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)