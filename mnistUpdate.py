import numpy as np
from PIL import Image
from numData import x_test, y_test
from numData import x_train, y_train
import cv2

# Function to load and preprocess a PNG image
def load_image_as_array(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image_array = np.array(image)  # Convert to NumPy array
    return image_array

def update_mnist(new_image, new_label):
    global x_train
    global y_train

    assert new_image.shape == (1,28,28)

    # Concatenate the new data with the existing data
    x_train = np.concatenate((x_train, new_image), axis=0)
    y_train = np.append(y_train, new_label)

    # Save the updated data, overwriting the original file
    np.savez('mnist.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)