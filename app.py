from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import logging 
from modules.cropDiamond import crop_image
import keras
# from tensorflow.keras.models import load_model  
from PIL import Image
from keras.preprocessing import image as tf_image
# from tensorflow.keras.preprocessing import image as tf_image
import base64
from pathlib import Path
import threading
# test
classifierModel = None 
validatorModel=None
model_lock = threading.Lock()

# Function to load the model at app startup
model_dir="/home/forge/diamond_filter/image_filter/models"
# models
# model_dir =  "models"
validator_model_path = model_dir+"/validatormodel773.keras"
classifier_model_path = model_dir+'/diamond_classifier_model_Shipready.keras'

def load_model_on_startup():
    global validatorModel
    global classifierModel
    try:
        with model_lock:
            if validatorModel is None:
                logging.info("Loading validator model at startup...")
                validatorModel =  keras.saving.load_model(validator_model_path)
                logging.info("Validator model loaded.")
            if classifierModel is None:
                logging.info("Loading type classifier model at startup...")
                classifierModel =  keras.saving.load_model(classifier_model_path)
                logging.info("Classifier model loaded.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        # Additional logging to ensure models are loaded
        if validatorModel is None:
            logging.error("Validator model is not loaded.")
        if classifierModel is None:
            logging.error("Classifier model is not loaded.")

app = Flask(__name__)

# validator func
def validate_image(img):
    global validatorModel
    if validatorModel is None:
        # In case the model is not loaded for some reason, load it again
        load_model_on_startup()

    pil_image = Image.fromarray(img)
    img = tf_image.smart_resize(pil_image, size=(165, 200))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    prediction = validatorModel.predict(img_array)
    if prediction[0] > 0.5:
        return False,prediction[0]
    else:
        return True,prediction[0]
    
# classify diamond type
def validate_Type_image(img):
    class_labels=['ASSCHER','CUSHION','EMERALD','HEART','MARQUISE','MOVAL','OLD MINER','OVAL','PEAR','PRINCESS','RADIANT','ROUND','TRAPEZE']
    global classifierModel
    if classifierModel is None:
        # In case the model is not loaded for some reason, load it again
        load_model_on_startup()
        
    pil_image = Image.fromarray(img)
    img = tf_image.smart_resize(pil_image, size=(165, 200))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    prediction = classifierModel.predict(img_array)
    # print(prediction)
    class_idx = np.argmax(prediction[0])
    return class_labels[class_idx]

@app.route("/")
def hello_world():
    return "<p>Hello, welcome to image filter!</p>"

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image provided"}), 400

    image_file = request.files['image']

    try:
        # # Convert the uploaded file to a NumPy array
        image_bytes = np.frombuffer(image_file.read(), np.uint8)

        # # Decode the image using OpenCV
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        # image = cv2.imread(image_url)
        if image is None:
            raise ValueError("Invalid image file")

    except Exception as e:
        return jsonify({"success": False, "message": f"Error processing image: {str(e)}"}), 400

    Validity,estimation=validate_image(image)
    estimation_list = estimation.tolist()
    if not Validity:
        return jsonify({"success": False, "message": "Image Rejected","estimation":estimation_list}), 400

    shape=validate_Type_image(image)

    # crop img
    cropped_image = crop_image(image)

    _, buffer = cv2.imencode('.jpg', cropped_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    response = {
            "success": True,
            "shape":shape,
            "valid": Validity  ,
            "estimation":estimation_list,
            "image": image_base64
        }
    print(response)
    
    return jsonify(response), 200

def list_routes(app):
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        print(f"Endpoint: {rule.endpoint:20} Path: {rule.rule:30} Methods: {methods}")
        
if __name__ == '__main__':
    load_model_on_startup()
    # list_routes(app)
    app.run(debug=True)