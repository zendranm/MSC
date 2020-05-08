import tensorflow as tf
import os
from PIL import Image, ImageOps
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

images_path = 'C:/Users/michal/Desktop/MSC/Test_Images/subject_A/scene_1/'
models_path = 'C:/Users/michal/Desktop/MSC/Models/AE/'

images = os.listdir(images_path)

encoder = tf.keras.models.load_model(models_path + 'encoder_150.h5')
decoder_XY = tf.keras.models.load_model(models_path + 'decoder_XY_150.h5')
decoder_X = tf.keras.models.load_model(models_path + 'decoder_X_350.h5')
# decoder_Y = tf.keras.models.load_model(models_path + 'decoder_Y')


i = 0

feature_range=(-1.0, 1.0)
min, max = feature_range

plt.figure(figsize = (7,7))
gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0, hspace=0) # set the spacing between axes. 

for image in images:

    # Image preprocessing and latent face
    image = Image.open(images_path + image)
    image = asarray(image)
    image = image/255.0
    image = image * (max - min) + min

    latent_face = encoder.predict(image[None])

    # Original image
    plt.subplot(gs1[i])
    image_scaled = (image - min) / (max - min)
    plt.imshow(image_scaled)
    plt.axis('off')
    i += 1

    # Decoder XY
    reconstructed = decoder_XY.predict(latent_face)[0]

    plt.subplot(gs1[i])
    image_scaled = (reconstructed - min) / (max - min)
    plt.imshow(image_scaled)
    plt.axis('off')
    i += 1

    # Decoder X
    reconstructed = decoder_X.predict(latent_face)[0]

    plt.subplot(gs1[i])
    image_scaled = (reconstructed - min) / (max - min)
    plt.imshow(image_scaled)
    plt.axis('off')
    i += 1

    # Decoder Y
    reconstructed = decoder_X.predict(latent_face)[0]

    plt.subplot(gs1[i])
    image_scaled = (reconstructed - min) / (max - min)
    plt.imshow(image_scaled)
    plt.axis('off')
    i += 1

plt.savefig("image.png",bbox_inches='tight',dpi=100)
plt.show()

print("Finished successfully")