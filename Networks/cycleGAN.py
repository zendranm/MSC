# Libraries
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

# Constants
data_dir = 'C:/Users/michal/Desktop/data/'
train_person_A = data_dir + 'train_A/'
test_person_A = data_dir + 'test_A/'
train_person_B = data_dir + 'train_B/'
test_person_B = data_dir + 'test_B/'

EPOCHS = 40
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

AUTOTUNE = tf.data.experimental.AUTOTUNE # Whats that???

# Load data
def loadImages(src_dir):
    images = []
    files = glob.glob(src_dir + "*.PNG")
    for myFile in files:
        image = cv2.imread(myFile)
        images.append(image)
    numpyArray = np.array(images)
    return numpyArray

train_A = loadImages(train_person_A)
test_A = loadImages(test_person_A)
train_B = loadImages(train_person_B)
test_B = loadImages(test_person_B)

# Preprocess data
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

train_A = tf.data.Dataset.from_tensor_slices(train_A).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_A = tf.data.Dataset.from_tensor_slices(test_A).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_B = tf.data.Dataset.from_tensor_slices(train_B).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_B = tf.data.Dataset.from_tensor_slices(test_B).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# # Import models
# generator_g = 0
# generator_f = 0
# discriminator_x = 0
# discriminator_y = 0

# # Show generated samples
# sample_A = next(iter(train_A))
# sample_B = next(iter(train_B))

# to_B = generator_g(sample_A)
# to_A = generator_f(sample_B)
# plt.figure(figsize=(8, 8))
# contrast = 8

# imgs = [sample_A, to_B, sample_B, to_A]
# title = ['A', 'To B', 'B', 'To A']

# for i in range(len(imgs)):
#   plt.subplot(2, 2, i+1)
#   plt.title(title[i])
#   if i % 2 == 0:
#     plt.imshow(imgs[i][0] * 0.5 + 0.5)
#   else:
#     plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.show()

# Loss function

# Checkpoints

# Train

print('Program finished with success')