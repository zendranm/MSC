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
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_COLOR = 3

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
  image = (image - 127.5) / 127.5
  return image

train_A = tf.data.Dataset.from_tensor_slices(train_A).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_A = tf.data.Dataset.from_tensor_slices(test_A).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_B = tf.data.Dataset.from_tensor_slices(train_B).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_B = tf.data.Dataset.from_tensor_slices(test_B).map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# # Test data
# sample_A = next(iter(train_A))
# sample_B = next(iter(train_B))

# plt.subplot(121)
# plt.title('Horse')
# plt.imshow(sample_A[0] * 0.5 + 0.5)

# plt.subplot(122)
# plt.title('Horse')
# plt.imshow(sample_B[0] * 0.5 + 0.5)
# plt.show()

# Import models
from Generators.generator import make_generator_model
from Discriminators.discriminator import make_discriminator_model

generator_g = make_generator_model()
# generator_f = make_generator_model()
discriminator_x = make_discriminator_model()
# discriminator_y = make_discriminator_model()

# Show generated samples
noise = tf.random.normal([IMG_HEIGHT, IMG_WIDTH, IMG_COLOR])
tmp=tf.expand_dims(noise, 0)

tmp2 = generator_g(tmp)
to_B = tf.squeeze(tmp2, 0)

plt.subplot(121)
plt.title('Noise')
plt.imshow(noise)
plt.subplot(122)
plt.title('to_B')
plt.imshow(to_B  * 0.5 * 108 + 0.5)
plt.show()

# sample_A = next(iter(train_A))
# sample_B = next(iter(train_B))

# to_B = generator_g(sample_A)
# to_A = generator_f(noise)
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