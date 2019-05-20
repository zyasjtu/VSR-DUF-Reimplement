import random
import numpy as np
import glob
from utils import LoadImage

dataset_dir = '/volume/data/youku/bmp/'
train_lr_list = [dataset_dir + 'Youku_' + str(x).zfill(5) + '_l/' for x in range(0,150)]
train_hr_list = [dataset_dir + 'Youku_' + str(x).zfill(5) + '_h_GT/' for x in range(0,150)]
valid_lr_list = [dataset_dir + 'Youku_' + str(x).zfill(5) + '_l/' for x in range(150,200)]
valid_hr_list = [dataset_dir + 'Youku_' + str(x).zfill(5) + '_h_GT/' for x in range(150,200)]

def get_x(path, T_in):
    dir_frames=glob.glob(path+"*.bmp")
    dir_frames.sort()
    frames=[]
    for f in dir_frames:
        frames.append(LoadImage(f))
    frames = np.asarray(frames)
    frames_padded = np.lib.pad(frames, pad_width=((T_in // 2, T_in // 2), (0, 0), (0, 0), (0, 0)), mode='constant')
    return frames,frames_padded

def get_y(path):
    dir_frames = glob.glob(path+"*.bmp")
    dir_frames.sort()
    frames = []
    for f in dir_frames:
        frames.append(LoadImage(f))
    frames = np.asarray(frames)

    y_true=[]
    for i in range(len(frames)):
        y_true.append(frames[i][np.newaxis,np.newaxis,:,:,:])
    y_true=np.asarray(y_true)
    return frames,y_true

if __name__ == '__main__':
    idx = range(len(train_lr_list))
    random.shuffle(idx)
    for i in idx:
        print(train_lr_list[i], train_hr_list[i])
