import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Functions
from functions import load_data
from functions import show_single_image
from functions import build_autoencoder
from functions import visualize

# Load and prepare data
npz_path = 'C:/Users/michal/Desktop/DiCaprioToDowneyJr_test.npz'
(x_train, y_train), (x_test, y_test) = load_data(npz_path)

x_train = x_train.astype('float32') / 255.0 - 0.5
x_test = x_test.astype('float32') / 255.0 - 0.5
y_train = y_train.astype('float32') / 255.0 - 0.5
y_test = y_test.astype('float32') / 255.0 - 0.5

# Veryfie loaded data
# print(x_train.max(), x_train.min())
# show_single_image(x_train[0])

# Prepare two autoencoders
IMG_SHAPE = x_train.shape[1:]

encoder_X, decoder_X = build_autoencoder(IMG_SHAPE, 600)
inp_X = tf.keras.layers.Input(IMG_SHAPE)
code = encoder_X(inp_X)
reconstruction_X = decoder_X(code)
autoencoder_X = tf.keras.models.Model(inp_X, reconstruction_X)
autoencoder_X.compile(optimizer='adamax', loss='mse')
# print(autoencoder_X.summary())

encoder_Y, decoder_Y = build_autoencoder(IMG_SHAPE, 600)
inp_Y = tf.keras.layers.Input(IMG_SHAPE)
code = encoder_Y(inp_Y)
reconstruction_Y = decoder_Y(code)
autoencoder_Y = tf.keras.models.Model(inp_Y, reconstruction_Y)
autoencoder_Y.compile(optimizer='adamax', loss='mse')
# print(autoencoder_Y.summary())

# Train both autoencoders
history_X = autoencoder_X.fit(x= x_train, y= x_train, epochs=10, validation_data= [x_test, x_test])
history_Y = autoencoder_X.fit(x= x_train, y= x_train, epochs=10, validation_data= [x_test, x_test])

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# for i in range(2):
#     img = x_test[i]
#     visualize(img,encoder_X,decoder_X)

# for i in range(2):
#     img = y_test[i]
#     visualize(img,encoder_Y,decoder_Y)

# Deepfake
for i in range(3):
    img = x_test[i]
    visualize(img,encoder_X,decoder_Y)

print("Program finished successfully!")