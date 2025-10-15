import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import cv2
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = 'breast_cancer_model_final.keras'
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
DISPLAY_SIZE = (300, 300) 

# --- GUI Application ---
class BreastCancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Diagnostic Tool")
        self.root.geometry("400x550") # Increased height for status bar
        self.root.configure(bg="#f0f0f0")

        self.model = None # The model will be loaded later

        # --- Widgets ---
        self.title_label = tk.Label(root, text="Ultrasound Image Analyzer", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=10)

        self.image_label = tk.Label(root, text="No Image Selected", bg="#cccccc", width=42, height=18)
        self.image_label.pack(pady=5)

        self.result_label = tk.Label(root, text="Diagnosis: N/A", font=("Helvetica", 12, "italic"), bg="#f0f0f0")
        self.result_label.pack(pady=10)

        self.select_btn = tk.Button(root, text="Select Image", command=self.select_image_and_predict, font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", state=tk.DISABLED)
        self.select_btn.pack(pady=20, ipadx=10, ipady=5)

        # Status Bar
        self.status_label = tk.Label(root, text="Initializing...", font=("Helvetica", 10), bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Schedule the model loading to start shortly after the window appears
        self.root.after(200, self.load_model)

    def load_model(self):
        """Loads the model after the GUI is visible and updates the status."""
        self.status_label.config(text="Loading model, please wait...")
        self.root.update_idletasks() # Force GUI to update
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.status_label.config(text="Model loaded successfully. Ready.")
            self.select_btn.config(state=tk.NORMAL) # Enable the button now
            print(f"Model '{MODEL_PATH}' loaded successfully.")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Could not load the model file: {e}")
            self.status_label.config(text="Error: Model failed to load.")

    def predict_image(self, image_path):
        """Performs the prediction using the loaded model."""
        if not self.model:
            messagebox.showerror("Error", "Model is not loaded.")
            return "Error", 0.0
        
        try:
            img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, IMAGE_SIZE)
            
            img_batch = np.expand_dims(resized_array, axis=-1)
            img_batch = np.expand_dims(img_batch, axis=0)

            # Use self.model for prediction
            prediction = self.model.predict(img_batch)
            
            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction[0])

            return predicted_class_name, confidence
        except Exception as e:
            return f"Prediction Error: {e}", 0.0

    def select_image_and_predict(self):
        """Handles the button click event."""
        file_path = filedialog.askopenfilename(
            title="Select an Ultrasound Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return

        img = Image.open(file_path)
        img.thumbnail(DISPLAY_SIZE)
        photo = ImageTk.PhotoImage(img)

        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

        self.result_label.config(text="Analyzing...")
        self.root.update_idletasks()

        predicted_class, confidence = self.predict_image(file_path)

        if "Error" in predicted_class:
            self.result_label.config(text=predicted_class)
        else:
            result_text = f"Diagnosis: {predicted_class} ({confidence:.2%})"
            self.result_label.config(text=result_text, font=("Helvetica", 14, "bold"))

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = BreastCancerApp(root)
    root.mainloop()

