import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# Load an image and convert it to an array
def load_image_from_file(file):
    img_path = os.path.join(file)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    return np.array(img_array)


# Load images and convert them to arrays
def load_images_from_directory(directory, size):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                if image_path.endswith(".JPG") or image_path.endswith(".png"):
                    img = load_img(image_path, target_size=size)
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
    return np.array(images), np.array(labels)
