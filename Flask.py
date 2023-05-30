from flask import Flask, jsonify
from keras.models import load_model
import cv2
from Preprocessing import Preprocessing

app = Flask(__name__)
#Load The Model
model=load_model("Untitled Folder/MNIST_CNN.h5")

@app.route('/predict/<path:image_path>')
def predict(image_path):
    image = cv2.imread(image_path)

   #Preprocessing and prediction
    prediction,path,prob=Preprocessing(model,image)

    # Return the predictions as JSON response
    return jsonify({'predictions': prediction,
                    'Path_Photo' : path,
                    'Probabilty' : str(prob)})


if __name__ == '__main__':
    app.run()