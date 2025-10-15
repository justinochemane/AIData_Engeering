import tensorflow as tf
import cv2
import numpy as np
import os

# --- Configuration ---
# Define the path to your saved Keras model file.
MODEL_PATH = 'breast_cancer_model_final.keras'

# Define the size that the model expects for input images.
# This MUST match the size you used for training.
IMAGE_SIZE = (128, 128)

# Define the names of the classes in the correct order (0, 1, 2).
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

# --- Model Loading ---
# Load the trained model from the file.
# It's best to load it once when the script starts.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit the script if the model can't be loaded.
    exit()

def predict_image(image_path):
    """
    Loads an image, preprocesses it, and uses the trained model
    to predict its class.

    Args:
        image_path (str): The full path to the image file.

    Returns:
        tuple: A tuple containing the predicted class name (str)
               and the confidence score (float).
    """
    if not os.path.exists(image_path):
        return "Error: Image file not found.", 0.0

    # 1. Load the Image in Grayscale
    # We use cv2.IMREAD_GRAYSCALE to ensure it has 1 channel, just like our training data.
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        return "Error: Could not read image.", 0.0

    # 2. Resize the Image
    # The image must be resized to the exact dimensions the model was trained on.
    resized_array = cv2.resize(img_array, IMAGE_SIZE)

    # 3. Prepare the Image for the Model
    # The model's predict function expects a "batch" of images.
    # We need to add two dimensions to our 128x128 image to make its
    # shape (1, 128, 128, 1).
    # (batch_size, height, width, channels)
    img_batch = np.expand_dims(resized_array, axis=-1) # Add channel dimension
    img_batch = np.expand_dims(img_batch, axis=0)      # Add batch dimension

    # 4. Make the Prediction
    prediction = model.predict(img_batch)
    
    # 5. Interpret the Prediction
    # The prediction is an array of probabilities for each class (e.g., [[0.1, 0.8, 0.1]])
    # We find the index of the highest probability.
    predicted_class_index = np.argmax(prediction[0])
    # We use that index to get the class name.
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    # The confidence is the highest probability value.
    confidence = np.max(prediction[0])

    return predicted_class_name, confidence

# --- Main Execution ---
if __name__ == "__main__":
    # This is a placeholder for the image you will get from your ultrasound sensor.
    # For now, we will test it with an image from your dataset.
    # **IMPORTANT**: Make sure this image is NOT from your training set.
    # It's best to use an image from the original 'test' or 'validation' split.
    
    # Create a placeholder test image path.
    # You MUST replace this with a real path to a test image on your Raspberry Pi.
    # For example: '/home/pi/test_images/benign_test_01.png'
    test_image_path = './test_images/sample_test_image.png'

    print(f"\nAnalyzing image: {test_image_path}...")
    
    # Get the prediction from our function.
    predicted_class, confidence = predict_image(test_image_path)

    # Display the final result to the user.
    if "Error" in predicted_class:
        print(predicted_class)
    else:
        print(f"\n--- DIAGNOSIS ---")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"-----------------")
