import os
from PIL import Image
from PIL import ImageChops
import numpy as np
import math
#import cv2

OFF = 1
SCALE = 2#2, 3, or 4
X=-1

if __name__ == '__main__':
    data_path = './train_data/train_image/'
    save_path = './train_data/train_image_samples/train_down2/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    count =0
    num =0
    filenames = os.listdir(data_path)
        #if 'png' in filename or 'tif' in filename]
    for filename in filenames:  
        pic_path = os.path.join(data_path, filename)
        img = Image.open(pic_path)
		
        IMG = img.resize((img.size[0]//SCALE, img.size[1]//SCALE),Image.ANTIALIAS)
        IMG.save(os.path.join(save_path,filename))
        count+=1
        num=0
        print(count)
