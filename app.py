import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Loading the model
knnmodel = pickle.load(open('tuned_knn.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    data_array = np.array(list(data.values())).reshape(1, -1)
    output = knnmodel.predict(data_array)
    if output==0:
        message = 'This account is not Spam!'
    else:
        message = 'This account is Spam!'
    print(message)
    return jsonify({'message': message})

if __name__ == "__main__":
    app.run(debug=True)
