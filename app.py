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

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output = knnmodel.predict(final_input)
    if output==0:
        message = 'not Spam!'
    else:
        message = 'Spam!'
    return render_template('home.html',prediction_text='This account is {}'.format(message))

if __name__ == "__main__":
    app.run(debug=True)
