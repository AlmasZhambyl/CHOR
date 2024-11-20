import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

model_path = 'model.h5'
try:
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {e}")

unique_labels = [
    'Disc Relaxationrotation', 'Disc Bulgerotation', 'Disc Protrusionrotation',
    'Demyelinating Plaquesrotation', 'Degenerative Disc Diseaserotation',
    'Disc Herniationrotation', 'Annular Disc Relaxationrotation',
    'Hemangiomatarotation', 'Fracturerotation', 'Annular Tearrotation',
    'Osteophytesrotation'
]

def process_image(image_path):
    try:
        image = load_img(image_path, target_size=(150, 150))
        image = img_to_array(image)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")

def predict_image():
    global img_display
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return

    try:
        processed_image = process_image(file_path)
        prediction = model.predict(processed_image)
        predicted_label = unique_labels[np.argmax(prediction)]

        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_display = ImageTk.PhotoImage(img)
        img_label.config(image=img_display)

        result_label.config(text=f"Predicted Label: {predicted_label}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("MRI Prediction")
root.geometry("400x500")

welcome_label = tk.Label(root, text="MRI Image Prediction", font=("Arial", 18))
welcome_label.pack(pady=10)

upload_button = tk.Button(root, text="Upload Image", command=predict_image, font=("Arial", 14))
upload_button.pack(pady=20)

img_label = tk.Label(root)
img_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
