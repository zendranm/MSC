import os
import shutil
import cv2

# Constants
root_folder = 'C:/Users/michal/Desktop/data/'
src_folder = root_folder + 'id02019/'

# Functions
def prepare_directory(base_folder, new_folder_name):
    if os.path.isdir(base_folder + new_folder_name) == True:
        print('exist')
        shutil.rmtree(base_folder + new_folder_name)
    else:
        print('no exists')

    os.mkdir(base_folder + 'src/')

def copy_and_rename_videos(old_src_folder, new_src_folder):
    counter = 0
    subfolders = os.listdir(old_src_folder)
    for subfolder in subfolders:
        files = os.listdir(old_src_folder + subfolder + '/')
        for file_name in files:
            counter += 1
            shutil.copy(old_src_folder + subfolder + '/' + file_name, new_src_folder)
            os.rename(new_src_folder + file_name, new_src_folder + "src_video_" + str(counter) + '.mp4')

def extract_frames_from_video(video_path, video_name):
    vidcap = cv2.VideoCapture(video_path + video_name)
    success, image = vidcap.read()
    counter = 0
    while success:
        cv2.imwrite(video_path + "frame%d.jpg" % counter, image)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        counter += 1

# prepare_directory(root_folder, 'src/')
# copy_and_rename_videos(src_folder, root_folder + 'src/')

extract_frames_from_video(root_folder + 'src/', 'src_video_32.mp4')

# Here decompose every mp4 file into single frames

# Here cut only face out of every frame