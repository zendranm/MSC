# Libraries
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import glob

tfd = tfp.distributions

# Load data
def load_data(filename):
    data = np.load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

# Load data
def load_images(src_dir, size):
    images = list()
    files = glob.glob(src_dir + "*.PNG")
    for filename in files:
        pixels = tf.keras.preprocessing.image.load_img(filename, target_size=size)
        pixels = tf.keras.preprocessing.image.img_to_array(pixels)
        images.append(pixels)
    return images

# Preprocess data
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image - 127.5) / 127.5
  return image

# Encoder
def make_encoder(data, code_size):
    x = tf.keras.layers.Flatten()(data)
    x = tf.keras.layers.Dense(200, tf.nn.relu)(x)
    x = tf.keras.layers.Dense(200, tf.nn.relu)(x)
    loc = tf.keras.layers.Dense(code_size)(x)
    scale = tf.keras.layers.Dense(code_size, tf.nn.softplus)(x)
    return tfd.MultivariateNormalDiag(loc, scale)

# Prior
def make_prior(code_size):
    loc = tf.zeros(code_size)
    scale = tf.ones(code_size)
    return tfd.MultivariateNormalDiag(loc, scale)

# Decoder
def make_decoder(code, data_shape):
    x = code
    x = tf.keras.layers.Dense(200, tf.nn.relu)(x)
    x = tf.keras.layers.Dense(200, tf.nn.relu)(x)
    logit = tf.keras.layers.Dense(np.prod(data_shape))(x)
    logit = tf.reshape(logit, [-1] + data_shape)
    return tfd.Independent(tfd.Bernoulli(logit), 2)

def plot_codes(ax, codes, labels):
    ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
    ax.set_aspect('equal')
    ax.set_xlim(codes.min() - .1, codes.max() + .1)
    ax.set_ylim(codes.min() - .1, codes.max() + .1)
    ax.tick_params(axis='both', which='both', left='off', bottom='off', labelleft='off', labelbottom='off')


def plot_samples(ax, samples):
    for index, sample in enumerate(samples):
        ax[index].imshow(sample, cmap='gray')
        ax[index].axis('off')