import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('prediction.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/index',methods=['GET','POST'])
def index():
    return render_template('index.html')




@app.route('/predict',methods=['GET','POST'])
def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        return render_template('predict.html', prediction_text='Volume weighted average price $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)