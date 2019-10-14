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

def extract_and_save_frames_from_video(video_path, dest_path, counter):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        cv2.imwrite(dest_path + "frame%d.png" % counter, image)
        success,image = vidcap.read()
        counter += 1
    vidcap.release()
    return counter

def extract_and_save_every_nth_frame_from_video(video_path, dest_path, counter, nth):
    frames_to_skip = 0
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        cv2.imwrite(dest_path + "frame%d.png" % counter, image)
        frames_to_skip += nth
        vidcap.set(1, frames_to_skip)
        success,image = vidcap.read()
        counter += 1
    vidcap.release()
    return counter

def process_videos(old_src_folder, new_src_folder, frames_to_skip):
    frame_counter = 1
    folder_counter = 1
    file_counter = 1
    subfolders = os.listdir(old_src_folder)
    for subfolder in subfolders:
        files = os.listdir(old_src_folder + subfolder + '/')
        for file_name in files:
            video_path = old_src_folder + subfolder + '/' + file_name
            print("Folder: " + str(folder_counter) + ", Video: " + str(file_counter) + ", Name: " + file_name)
            if (frames_to_skip == 0) or (frames_to_skip == 1):
                frame_counter = extract_and_save_frames_from_video(video_path, new_src_folder, frame_counter)
            else:
                frame_counter = extract_and_save_every_nth_frame_from_video(video_path, new_src_folder, frame_counter, frames_to_skip)
            file_counter += 1
        folder_counter += 1

prepare_directory(root_folder, 'src/')
process_videos(src_folder, root_folder + 'src/', 20)

# Here cut only face out of every frame