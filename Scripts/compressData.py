# Libraries
import tensorflow as tf
import numpy as np
import glob

# Constants
data_dir = 'C:/Users/michal/Desktop/DiCaprioToDowneyJr/'
train_person_A = data_dir + 'train_A/'
test_person_A = data_dir + 'test_A/'
train_person_B = data_dir + 'train_B/'
test_person_B = data_dir + 'test_B/'

IMG_HEIGHT = 160
IMG_WIDTH = 160

AUTOTUNE = tf.data.experimental.AUTOTUNE # Whats that???

# Load data
def loadImages(src_dir, size=(IMG_HEIGHT, IMG_WIDTH)):
    images = list()
    files = glob.glob(src_dir + "*.PNG")
    for filename in files:
        pixels = tf.keras.preprocessing.image.load_img(filename, target_size=size)
        pixels = tf.keras.preprocessing.image.img_to_array(pixels)
        images.append(pixels)
    return np.asarray(images)

train_A = loadImages(train_person_A)
test_A = loadImages(test_person_A)
train_B = loadImages(train_person_B)
test_B = loadImages(test_person_B)

# load dataset A
dataA1 = train_A
dataA2 = test_A
dataA = np.vstack((dataA1, dataA2))
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB1 = train_B
dataB2 = test_B
dataB = np.vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
filename = 'AtoB.npz'
np.savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)