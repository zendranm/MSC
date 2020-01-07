import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


def load_data(filename):
    with np.load(filename) as f:
        x_train, x_test = f['arr_0'], f['arr_1']
        y_train, y_test = f['arr_2'], f['arr_3']
        return (x_train, y_train), (x_test, y_test)

def show_image(x):
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
    decoder.add(tf.keras.layers.Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
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

npz_path = 'C:/Users/michal/Desktop/DiCaprioToDowneyJr_test.npz'
# (train_images, _), (test_images, _) = load_data(npz_path)
(_, train_images), (_, test_images) = load_data(npz_path)

train_images = train_images.astype('float32') / 255.0 - 0.5
test_images = test_images.astype('float32') / 255.0 - 0.5

# print(train_images.max(), train_images.min())
# show_image(test_images[0])

IMG_SHAPE = train_images.shape[1:]
encoder, decoder = build_autoencoder(IMG_SHAPE, 1000)

inp = tf.keras.layers.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = tf.keras.models.Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

print(autoencoder.summary())

history = autoencoder.fit(x= train_images, y= train_images, epochs=10, validation_data= [test_images, test_images])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

for i in range(2):
    img = test_images[i]
    visualize(img,encoder,decoder)

print("Program finished successfully!")