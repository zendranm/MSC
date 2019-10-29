# Libraries
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import time
from IPython.display import clear_output

# Constants
data_dir = 'C:/Users/michal/Desktop/data/'
train_person_A = data_dir + 'train_A/'
test_person_A = data_dir + 'test_A/'
train_person_B = data_dir + 'train_B/'
test_person_B = data_dir + 'test_B/'

EPOCHS = 3
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_COLOR = 3

AUTOTUNE = tf.data.experimental.AUTOTUNE # Whats that???

# Load data
def loadImages(src_dir, size=(IMG_HEIGHT, IMG_WIDTH)):
    images = []
    files = glob.glob(src_dir + "*.PNG")
    for filename in files:
        pixels = tf.keras.preprocessing.image.load_img(filename, target_size=size)
        pixels = tf.keras.preprocessing.image.img_to_array(pixels)
        images.append(pixels)
    return images

train_A = loadImages(train_person_A)
test_A = loadImages(test_person_A)
train_B = loadImages(train_person_B)
test_B = loadImages(test_person_B)

print(train_A[0])

# Preprocess data
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image - 127.5) / 127.5
  return image

train_A = tf.data.Dataset.from_tensor_slices(train_A).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_A = tf.data.Dataset.from_tensor_slices(test_A).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_B = tf.data.Dataset.from_tensor_slices(train_B).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_B = tf.data.Dataset.from_tensor_slices(test_B).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Test data
sample_A = next(iter(train_A))
sample_B = next(iter(train_B))

plt.subplot(121)
plt.title('Horse')
plt.imshow(sample_A[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Horse')
plt.imshow(sample_B[0] * 0.5 + 0.5)
plt.show()