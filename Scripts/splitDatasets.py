# Libraries
import glob
import os
import shutil

# Constants
data_dir = 'C:/Users/michal/Desktop/DiCaprioToDowneyJr/'
src_person_A = data_dir + 'src_person_A/'
src_person_B = data_dir + 'src_person_B/'
train_person_A = data_dir + 'train_A/'
test_person_A = data_dir + 'test_A/'
train_person_B = data_dir + 'train_B/'
test_person_B = data_dir + 'test_B/'

# Functions
def prepare_directory(new_folder):
    if os.path.isdir(new_folder) == True:
        print('Directory: ' + new_folder + ' exists')
        shutil.rmtree(new_folder)
    else:
        print('Creating directory: ' + new_folder)

    os.mkdir(new_folder)

def split_frames_into_two_folders(src_folder, train_folder, test_folder):
    all_frames = os.listdir(src_folder)
    counter = 0 
    for frame in all_frames:
        if(counter % 2 == 0):
            shutil.copyfile(src_folder + frame, train_folder + frame)
        else:
            shutil.copyfile(src_folder + frame, test_folder + frame)
        counter += 1

prepare_directory(train_person_A)
prepare_directory(test_person_A)
prepare_directory(train_person_B)
prepare_directory(test_person_B)
split_frames_into_two_folders(src_person_A, train_person_A, test_person_A)
split_frames_into_two_folders(src_person_B, train_person_B, test_person_B)
print("Spliting datasets done")