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
generator_f = make_generator_model()
discriminator_x = make_discriminator_model()
discriminator_y = make_discriminator_model()

# # Show generated samples
sample_A = next(iter(train_A))
# sample_B = next(iter(train_B))

# to_B = generator_g(sample_A)
# to_A = generator_f(sample_B)
# plt.figure(figsize=(8, 8))
# contrast = 10000

# imgs = [sample_A, to_B, sample_B, to_A]
# title = ['A', 'To B', 'B', 'To A']

# for i in range(len(imgs)):
#   plt.subplot(2, 2, i+1)
#   plt.title(title[i])
#   # print(imgs[i][0])
#   if i % 2 == 0:
#     plt.imshow(imgs[i][0] * 0.5 + 0.5)
#   else:
#     plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.show()

# Loss function
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpoints
checkpoint_path = "./checkpoints"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

# Train
def generate_images(model, test_input):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

@tf.function
def train_step(real_x, real_y):
  print("TRAIN STEP")
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_A, train_B)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n+=1

clear_output(wait=True)
# Using a consistent image (sample_horse) so that the progress of the model
# is clearly visible.
generate_images(generator_g, sample_A)

if (epoch + 1) % 5 == 0:
  ckpt_save_path = ckpt_manager.save()
  print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))


print('Program finished with success')