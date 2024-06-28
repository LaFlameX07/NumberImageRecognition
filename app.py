from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io
import seaborn as sns
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('model/digit_recognition.h5')

# Recompile the model after loading
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_test = to_categorical(y_test, 10)

def get_model_metrics():
    test_loss, test_acc = model.evaluate(x_test, y_test)
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(y_test, axis=1)
    class_report = classification_report(actual_classes, predicted_classes)
    conf_matrix = confusion_matrix(actual_classes, predicted_classes)
    return test_acc, class_report, conf_matrix

@app.route('/')
def index():
    test_acc, class_report, conf_matrix = get_model_metrics()
    return render_template('index.html', test_acc=test_acc, class_report=class_report, conf_matrix=conf_matrix)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        img_bytes = io.BytesIO(file.read())
        img = load_img(img_bytes, color_mode='grayscale', target_size=(28, 28))
        img_array = img_to_array(img).reshape(1, 28, 28, 1).astype('float32') / 255
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        image_path = os.path.join('uploads', file.filename)
        return render_template('result.html', predicted_class=predicted_class, image_path=file.filename)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
