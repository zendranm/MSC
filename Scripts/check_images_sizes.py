import os
from PIL import Image

# Constants
data_dir = 'C:/Users/michal/Desktop/dataset_tmp/'
train_person_A = data_dir + 'train_A/'
test_person_A = data_dir + 'test_A/'
train_person_B = data_dir + 'train_B/'
test_person_B = data_dir + 'test_B/'

expected_image_size = 224

# Functions
def check_image_size(expected_size, image):
    im = Image.open(image)
    width, height = im.size
    if(width != expected_size and height != expected_size):
        print(image + ", height: " + str(height) + ", width: " + str(width))


def check_all_images(folder_name, expected_size):
    images = os.listdir(folder_name)

    for image in images:
        check_image_size(expected_size, folder_name + image)

check_all_images(train_person_A, expected_image_size)
check_all_images(test_person_A, expected_image_size)
check_all_images(train_person_B, expected_image_size)
check_all_images(test_person_B, expected_image_size)

print("Checking done")