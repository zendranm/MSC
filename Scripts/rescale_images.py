import os
from PIL import Image

# Constants
data_dir = 'C:/Users/michal/Desktop/dataset_160/'
train_person_A = data_dir + 'train_A/'
test_person_A = data_dir + 'test_A/'
train_person_B = data_dir + 'train_B/'
test_person_B = data_dir + 'test_B/'

new_image_size = 160

# Functions
def change_image_size(new_size, image):
    im = Image.open(image)
    im = im.resize((new_size, new_size))
    im.save(image)


def check_all_images(folder_name, new_size):
    images = os.listdir(folder_name)

    for image in images:
        change_image_size(new_size, folder_name + image)

check_all_images(train_person_A, new_image_size)
check_all_images(test_person_A, new_image_size)
check_all_images(train_person_B, new_image_size)
check_all_images(test_person_B, new_image_size)

print("Rescaleing done")