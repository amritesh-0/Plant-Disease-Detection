from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('models/plant_disease_model.keras')  # Adjust the path if needed

# Print the model summary
model.summary()
