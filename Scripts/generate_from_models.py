import tensorflow as tf
import os
from PIL import Image, ImageOps
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

images_path = 'C:/Users/michal/Desktop/MSC/Test_Images/subject_X/scene_1/'
models_path = 'C:/Users/michal/Desktop/MSC/Models/AE/'

images = os.listdir(images_path)
epoch = '350'

encoder = tf.keras.models.load_model(models_path + 'encoder_150.h5')
decoder_XY = tf.keras.models.load_model(models_path + 'decoder_XY_150.h5')
decoder_X = tf.keras.models.load_model(models_path + 'decoder_X_' + epoch + '.h5')
decoder_Y = tf.keras.models.load_model(models_path + 'decoder_Y_' + epoch + '.h5')

i = 0
feature_range=(-1.0, 1.0)
min, max = feature_range

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(7, 7.1))
axes[0, 0].set_title("Original")
axes[0, 1].set_title("Decoder XY")
axes[0, 2].set_title("Decoder Y")
axes[0, 3].set_title("Decoder X")

gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.1, hspace=0.1) # set the spacing between axes.

for image in images:

    # Image preprocessing and latent face
    image = Image.open(images_path + image)
    image = asarray(image)
    image = image/255.0
    image = image * (max - min) + min
    latent_face = encoder.predict(image[None])

    # Original image
    image_scaled = (image - min) / (max - min)
    axes[i, 0].imshow(image_scaled) 
    axes[i, 0].axis('off')
    
    # Decoder XY
    reconstructed = decoder_XY.predict(latent_face)[0]
    image_scaled = (reconstructed - min) / (max - min)
    axes[i, 1].imshow(image_scaled)
    axes[i, 1].axis('off')

    # Decoder Y
    reconstructed = decoder_Y.predict(latent_face)[0]
    image_scaled = (reconstructed - min) / (max - min)
    axes[i, 3].imshow(image_scaled)
    axes[i, 3].axis('off')

    # Decoder X
    reconstructed = decoder_X.predict(latent_face)[0]
    image_scaled = (reconstructed - min) / (max - min)
    axes[i, 2].imshow(image_scaled)
    axes[i, 2].axis('off')
    i += 1

fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()

print("Finished successfully")