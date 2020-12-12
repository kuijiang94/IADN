import os
from PIL import Image
from PIL import ImageChops
import numpy as np
import math
#import cv2

OFF = 1
SCALE = 4
X=-1

if __name__ == '__main__':
    data_path = './train_data/train_image/'
    #file = os.listdir(data_path)
    save_path = './train_data/train_image_samples/train_crop/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    count =0
    num =0
    # files = [
        # os.path.join(data_path, filename)
        # for filename in os.listdir(data_path)
        # if 'png' in filename or 'tif' in filename or 'jpg' in filename]
    filenames = os.listdir(data_path)
    for filename in filenames:
        pic_path = os.path.join(data_path, filename)
        img = Image.open(pic_path)
        print(img.size[0])
        if img.size[0]<96:
            img = img.resize((96, img.size[1]),Image.ANTIALIAS)
            print(filename)
        if img.size[1]<96:
            img = img.resize((img.size[0], 96),Image.ANTIALIAS)
            print(filename)
        for i in range(0,img.size[0],96):
            for j in range(0,img.size[1],96):
                if i+96 <= img.size[0] and j+96 <= img.size[1]: 
                    IMG = img.crop([i,j,i+96,j+96])
                    IMG.save(os.path.join(save_path,'{}_{}.png'.format(count,num)))
                    num+=1

        count+=1
        num=0
        print(count)
