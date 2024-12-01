from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('models/plant_disease_model.keras')

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to (224, 224)
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1,)
    img_array = img_array / 255.0  # Normalize the image to [0, 1]
    return img_array

@app.route('/')
def index():
    return render_template('index.html', prediction=None, uploaded_image=None)  # No prediction or image initially

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the 'file' key exists in the request
        if 'file' not in request.files:
            return render_template('index.html', prediction=None, uploaded_image=None, error="No file part.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction=None, uploaded_image=None, error="No file selected.")

        if file and allowed_file(file.filename):
            # Save the uploaded file securely
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            # Process the image
            img_array = preprocess_image(img_path)

            # Predict using the model
            predictions = model.predict(img_array)
            class_names = ['Early_Symptoms', 'Bacterial_spot', 'Healthy']  # Update with actual class names

            # Get the predicted probabilities and class
            class_probabilities = predictions[0] * 100  # Convert to percentages
            predicted_class = class_names[np.argmax(class_probabilities)]  # Most likely class
            predicted_percentage = round(max(class_probabilities), 2)  # Highest probability

            # Prepare prediction results
            result = {class_names[i]: round(prob, 2) for i, prob in enumerate(class_probabilities)}

            # Return the template with predictions and uploaded image
            return render_template(
                'index.html',
                prediction=result,
                uploaded_image=f"uploads/{filename}",  # Relative path for the static folder
                predicted_percentage=predicted_percentage
            )

        else:
            return render_template('index.html', prediction=None, uploaded_image=None, error="File type not allowed.")

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction=None, uploaded_image=None, error="Error processing the image.")

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  # Create the upload folder if it doesn't exist
    app.run(debug=True)
