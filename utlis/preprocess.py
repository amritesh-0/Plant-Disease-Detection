from keras.preprocessing import image
import numpy as np

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to (224, 224)
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1,)
    img_array = img_array / 255.0  # Normalize the image to [0, 1]
    return img_array
