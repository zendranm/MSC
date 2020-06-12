import tensorflow as tf
import os
from PIL import Image, ImageOps
from numpy import asarray
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import keras
import numpy as np

source_path = 'C:/Users/michal/Desktop/MSC/Test_Images/'
subjects = ["subject_X/", "subject_Y/"]
scenes = ["scene_1/", "scene_2/", "scene_3/", "scene_4/"]

models_path = 'C:/Users/michal/Desktop/MSC/Models/CycleGAN/'
method = "cyclegan"
cust = {'InstanceNormalization': InstanceNormalization}
X_to_Y = keras.models.load_model(models_path + 'g_model_AtoB_013000.h5', cust, compile=False)
Y_to_X = keras.models.load_model(models_path + 'g_model_BtoA_013000.h5', cust, compile=False)

destination_path = "C:/Users/michal/Desktop/MSC/Generated_Figures/"

feature_range=(-1.0, 1.0)

def create_and_save_figure(images_path, images, f_range, final_name, person):
    i = 0
    min, max = f_range

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(5.25, 7.1))

    if (person == "x"):
        axes[0, 0].set_title("Original")
        axes[0, 1].set_title("X to Y")
        axes[0, 2].set_title("Y to X")
    else :
        axes[0, 0].set_title("Original")
        axes[0, 1].set_title("Y to X")
        axes[0, 2].set_title("X to Y")
    

    gs1 = gridspec.GridSpec(4, 3)
    gs1.update(wspace=0.1, hspace=0.1) # set the spacing between axes.

    for image in images:

        # Image preprocessing and latent face
        image = Image.open(images_path + image)
        image = asarray(image)
        image = image/255.0
        image = image * (max - min) + min

        # Original image
        image_scaled = (image - min) / (max - min)
        axes[i, 0].imshow(image_scaled) 
        axes[i, 0].axis('off')

        image = np.expand_dims(image, axis=0)
        print(image.shape)

        if (person == "x"):
            # X to Y
            reconstructed = X_to_Y.predict(image)[0]
            image_scaled = (reconstructed - min) / (max - min)
            axes[i, 1].imshow(image_scaled)
            axes[i, 1].axis('off')

            # Y to X
            image = np.expand_dims(reconstructed, axis=0)
            reconstructed = Y_to_X.predict(image)[0]
            image_scaled = (reconstructed - min) / (max - min)
            axes[i, 2].imshow(image_scaled)
            axes[i, 2].axis('off')
            i += 1
        else :
            # Y to X
            reconstructed = Y_to_X.predict(image)[0]
            image_scaled = (reconstructed - min) / (max - min)
            axes[i, 1].imshow(image_scaled)
            axes[i, 1].axis('off')

            # X to Y
            image = np.expand_dims(reconstructed, axis=0)
            reconstructed = X_to_Y.predict(image)[0]
            image_scaled = (reconstructed - min) / (max - min)
            axes[i, 2].imshow(image_scaled)
            axes[i, 2].axis('off')
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
        name = method + "_" + person + "_" + scene[:-1] + "_13000" + ".png"
        create_and_save_figure(source_path + subject + scene, images, feature_range, name, person)



print("Finished successfully")