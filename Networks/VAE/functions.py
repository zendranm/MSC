import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    with np.load(filename) as f:
        x_train, x_test = f['arr_0'], f['arr_1']
        y_train, y_test = f['arr_2'], f['arr_3']
        return (x_train, y_train), (x_test, y_test)

def show_single_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
    plt.show()

def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = tf.keras.models.Sequential()
    encoder.add(tf.keras.layers.InputLayer(img_shape))
    encoder.add(tf.keras.layers.Flatten())
    encoder.add(tf.keras.layers.Dense(code_size))

    # The decoder
    decoder = tf.keras.models.Sequential()
    decoder.add(tf.keras.layers.InputLayer((code_size,)))
    decoder.add(tf.keras.layers.Dense(np.prod(img_shape)))
    decoder.add(tf.keras.layers.Reshape(img_shape))

    return encoder, decoder

def visualize(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(np.clip(img + 0.5, 0, 1))

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    plt.imshow(np.clip(reco + 0.5, 0, 1))
    plt.show()