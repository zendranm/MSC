# Libraries
import tensorflow as tf
import numpy as np
import glob

# Constants
name = 'DiCaprioToDowneyJr_Small_VAE'
data_dir = 'C:/Users/michal/Desktop/' + name + '/'
train_person_A = data_dir + 'train_A/'
test_person_A = data_dir + 'test_A/'
train_person_B = data_dir + 'train_B/'
test_person_B = data_dir + 'test_B/'

IMG_HEIGHT = 160
IMG_WIDTH = 160

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

# save as compressed numpy array
filename = name + '.npz'
np.savez_compressed(filename, train_A, test_A, train_B, test_B)
print('Saved dataset: ', filename)