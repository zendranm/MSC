import os
import shutil
import cv2

# Constants
root_folder = 'C:/Users/michal/Desktop/DiCaprioToDowneyJr_Small_VAE/'
root_person_A = root_folder + 'root_person_A/'
root_person_B = root_folder + 'root_person_B/'
src_person_A = root_folder + 'src_person_A/'
src_person_B = root_folder + 'src_person_B/'

face_cascade = cv2.CascadeClassifier('C:/Users/michal/Desktop/MSC/Scripts/haarcascade_frontalface_default.xml')

frames_to_skip = 10

target_frame_size = 160

haar_sensibility = 1.6

# Functions
def prepare_directory(new_folder):
    if os.path.isdir(new_folder) == True:
        print('Directory: ' + new_folder + ' exists')
        shutil.rmtree(new_folder)
    else:
        print('Creating directory: ' + new_folder)

    os.mkdir(new_folder)

def cut_face_out_of_frame(frame):
    faces = face_cascade.detectMultiScale(frame, haar_sensibility, 5)
    detected_faces = list()

    if (len(faces) == 0):
        return []
    else:
        for (x, y, w, h) in faces:
            cropped_image = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_image, (target_frame_size, target_frame_size))
            detected_faces.append(resized_img)
    
    return detected_faces

def extract_and_save_frames_from_video(video_path, dest_path, counter):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        cropped_images = cut_face_out_of_frame(image)
        if(len(cropped_images) != 0):
            for face in cropped_images:
                cv2.imwrite(dest_path + "frame%d.png" % counter, face)
                counter += 1
        success,image = vidcap.read()
    vidcap.release()
    return counter

def extract_and_save_every_nth_frame_from_video(video_path, dest_path, counter, nth):
    frames_to_skip = 0
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        cropped_image = cut_face_out_of_frame(image)
        cropped_images = cut_face_out_of_frame(image)
        if(len(cropped_images) != 0):
            for face in cropped_images:
                cv2.imwrite(dest_path + "frame%d.png" % counter, face)
                counter += 1
        frames_to_skip += nth
        vidcap.set(1, frames_to_skip)
        success,image = vidcap.read()
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

prepare_directory(src_person_A)
prepare_directory(src_person_B)
process_videos(root_person_A, root_folder + 'src_person_A/', frames_to_skip)
process_videos(root_person_B, root_folder + 'src_person_B/', frames_to_skip)
print("Proccessing done")