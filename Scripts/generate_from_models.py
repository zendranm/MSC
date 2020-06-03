import tensorflow as tf
import os
from PIL import Image, ImageOps
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

source_path = 'C:/Users/michal/Desktop/MSC/Test_Images/'
subjects = ["subject_X/", "subject_Y/"]
scenes = ["scene_1/", "scene_2/", "scene_3/", "scene_4/"]

models_path = 'C:/Users/michal/Desktop/MSC/Models/VAE/'
epoch = '450'
method = "vae"
encoder = tf.keras.models.load_model(models_path + 'encoder_100.h5')
decoder_XY = tf.keras.models.load_model(models_path + 'decoder_XY_100.h5')
decoder_X = tf.keras.models.load_model(models_path + 'decoder_X_' + epoch + '.h5')
decoder_Y = tf.keras.models.load_model(models_path + 'decoder_Y_' + epoch + '.h5')

destination_path = "C:/Users/michal/Desktop/MSC/Generated_Figures/"

feature_range=(-1.0, 1.0)

def create_and_save_figure(images_path, images, f_range, final_name, person):
    i = 0
    min, max = f_range

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(7, 7.1))

    if (person == "x"):
        axes[0, 0].set_title("Original")
        axes[0, 1].set_title("Decoder XY")
        axes[0, 2].set_title("Decoder X")
        axes[0, 3].set_title("Decoder Y")
    else :
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

        if (person == "x"):
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
        else :
            # Decoder Y
            reconstructed = decoder_Y.predict(latent_face)[0]
            image_scaled = (reconstructed - min) / (max - min)
            axes[i, 2].imshow(image_scaled)
            axes[i, 2].axis('off')

            # Decoder X
            reconstructed = decoder_X.predict(latent_face)[0]
            image_scaled = (reconstructed - min) / (max - min)
            axes[i, 3].imshow(image_scaled)
            axes[i, 3].axis('off')
            i += 1

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # plt.show()
    plt.savefig(destination_path + final_name)

for subject in subjects:
    for scene in scenes:
        images = os.listdir(source_path + subject + scene)
        person = ""
        if(subject == "subject_X/"):
            person = "x"
        else: person = "y"
        name = method + "_" + person + "_" + scene[:-1] + "_" + epoch + ".png"
        create_and_save_figure(source_path + subject + scene, images, feature_range, name, person)



print("Finished successfully")