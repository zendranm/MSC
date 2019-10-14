# Libraries
import tensorflow as tf
import matplotlib.pyplot as plt

# Constants
data_dir = ''
EPOCHS = 40
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

AUTOTUNE = tf.data.experimental.AUTOTUNE # Whats that???

# Load data
train_A = 0
test_A = 0
train_B = 0
test_B = 0

# Preprocess data
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

train_A = train_A.map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
test_A = train_A.map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
train_B = train_A.map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
test_B = train_A.map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)

# Import models
generator_g = 0
generator_f = 0
discriminator_x = 0
discriminator_y = 0

# Show generated samples
sample_A = next(iter(train_A))
sample_B = next(iter(train_B))

to_B = generator_g(sample_A)
to_A = generator_f(sample_B)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_A, to_B, sample_B, to_A]
title = ['A', 'To B', 'B', 'To A']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()

# Loss function

# Checkpoints

# Train