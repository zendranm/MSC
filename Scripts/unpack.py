import os
import shutil

base_folder = 'C:/Users/michal/Desktop/data/'
src_folder = base_folder + 'id02019/'

if os.path.isdir(base_folder + 'src/') == True:
    print('exist')
    shutil.rmtree(base_folder + 'src/')
else:
    print('no exists')

os.mkdir(base_folder + 'src/')

i = 0

src_folders = os.listdir(src_folder)
for subfolder in src_folders:
    src_files = os.listdir(src_folder + subfolder + '/')
    for file_name in src_files:
        i = i + 1
        print(subfolder)
        shutil.copy(src_folder + subfolder + '/' + file_name, base_folder + 'src/')
        os.rename(base_folder + 'src/' + file_name, base_folder + 'src/' + "src_img_" + str(i) + '.mp4')
