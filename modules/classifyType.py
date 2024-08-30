from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
# import io
from tensorflow.keras.preprocessing import image as tf_image
 
classifierModel = load_model('models\diamond_classifier_model Shipready.keras')
# Predict
class_labels=['ASSCHER','CUSHION','EMERALD','HEART','MARQUISE','MOVAL','OLD MINER','OVAL','PEAR','PRINCESS','RADIANT','ROUND','TRAPEZE']

def validate_Type_image(img):
    pil_image = Image.fromarray(img)
    img = tf_image.smart_resize(pil_image, size=(165, 200))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    prediction = classifierModel.predict(img_array)
    print(prediction)
    class_idx = np.argmax(prediction[0])
    return class_labels[class_idx]