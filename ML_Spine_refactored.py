import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def parse_xml(xml_file):
    if not os.path.exists(xml_file):
        return None
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            label = member.find('name').text
            return label
    except ET.ParseError as e:
        print(f"Error parsing {xml_file}: {e}")
        return None

data_dir = r'C:\Users\almas\Desktop\Final-AML\MRI_ML-INT'

train_image_dir = os.path.join(data_dir, 'train')
valid_image_dir = os.path.join(data_dir, 'valid')

if not os.path.exists(train_image_dir):
    raise FileNotFoundError(f"Training directory not found: {train_image_dir}")
if not os.path.exists(valid_image_dir):
    raise FileNotFoundError(f"Validation directory not found: {valid_image_dir}")

train_images = []
train_labels = []
valid_images = []
valid_labels = []

for image_file in os.listdir(train_image_dir):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(train_image_dir, image_file)
        try:
            image = load_img(image_path, target_size=(150, 150))
            image = img_to_array(image)
            image /= 255.0

            label_path = image_path.replace('.jpg', '.xml')
            label = parse_xml(label_path)

            if label is not None:
                train_images.append(image)
                train_labels.append(label)
        except Exception as e:
            print(f"Error loading or processing {image_file}: {e}")

for image_file in os.listdir(valid_image_dir):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(valid_image_dir, image_file)
        try:
            image = load_img(image_path, target_size=(150, 150))
            image = img_to_array(image)
            image /= 255.0

            label_path = image_path.replace('.jpg', '.xml')
            label = parse_xml(label_path)
            
            if label is not None:
                valid_images.append(image)
                valid_labels.append(label)
        except Exception as e:
            print(f"Error loading or processing {image_file}: {e}")

train_images = np.array(train_images)
valid_images = np.array(valid_images)

unique_labels = [
    'Disc Relaxationrotation', 'Disc Bulgerotation', 'Disc Protrusionrotation',
    'Demyelinating Plaquesrotation', 'Degenerative Disc Diseaserotation',
    'Disc Herniationrotation', 'Annular Disc Relaxationrotation',
    'Hemangiomatarotation', 'Fracturerotation', 'Annular Tearrotation',
    'Osteophytesrotation'
]

label_to_int = {label: i for i, label in enumerate(unique_labels)}

train_labels = to_categorical([label_to_int[label] for label in train_labels], num_classes=len(unique_labels))
valid_labels = to_categorical([label_to_int[label] for label in valid_labels], num_classes=len(unique_labels))

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_images, train_labels,
    epochs=15,
    validation_data=(valid_images, valid_labels)
)

model.save('model.h5')
print("Model saved as model.h5")

train_loss, train_accuracy = model.evaluate(train_images, train_labels, verbose=0)
valid_loss, valid_accuracy = model.evaluate(valid_images, valid_labels, verbose=0)

print(f"Baseline Accuracy (Training Set): {train_accuracy * 100:.2f}%")
print(f"Enhanced Model Accuracy (Validation Set): {valid_accuracy * 100:.2f}%")
