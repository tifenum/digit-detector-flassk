from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = joblib.load('svm_model.pkl')

def predict_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('L').resize((28, 28))
    img_array = np.array(img).reshape(1, 28*28) / 255.0
    
    prediction = model.predict(img_array)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.files['imageFile'].read()
        
        prediction = predict_image(image_data)

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
