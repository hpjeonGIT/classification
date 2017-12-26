#
# Read existing image files and make links into new path
# Sub-folders of image path are re-produced in target paths
# python3 sample_images_toPath.py
#
import os
import sys
import numpy as np
import time
import random

Existing_image_path = '/tmp/IMAGE/150/'
Train_path = '/tmp/KERAS/train/'
Val_path = '/tmp/KERAS/valid/'

Sampling_rate = 0.7
Nlimit =1000
def main():
    dir_name = []
    for subds, dirs, files in os.walk(Existing_image_path):
        for any in dirs:
            dir_name.append(any)

    for any in dir_name:
        if (not os.path.exists(Train_path + any)):
            os.makedirs(Train_path+any)
        if (not os.path.exists(Val_path + any)):
            os.makedirs(Val_path+any)
 
    for any in dir_name:
        src_path = Existing_image_path+any+'/'
        trn_path = Train_path+any+'/'
        val_path = Val_path+any+'/'
        for subds, dirs, files in os.walk(src_path):
            n = 0
            for f0 in files:
                if n >= Nlimit:
                    break
                x = random.random()
                if x < Sampling_rate:
                    n += 1
                    os.symlink(src_path+f0, trn_path+f0)
                else: 
                    os.symlink(src_path+f0, val_path+f0)
                    
if __name__ == '__main__':
    a_time = time.time()
    main()
    print("walll time = ", time.time() - a_time)

