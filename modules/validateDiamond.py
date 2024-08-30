from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
# import io
from tensorflow.keras.preprocessing import image as tf_image

# Load the model
validatorModel = load_model('models\diamondvalidator.keras')


# Predict

def validate_image(img):
    pil_image = Image.fromarray(img)
    img = tf_image.smart_resize(pil_image, size=(165, 200))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    prediction = validatorModel.predict(img_array)
    if prediction[0] > 0.5:
        return False
    else:
        return True