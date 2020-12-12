import os
import glob
import shutil
import cv2
#import dlib
import numpy as np
from tqdm import tqdm
import requests
import tarfile

def preprocess():
    print('... loading data')

    save_path2 = './train_data/npy'
    if not os.path.exists(save_path2):
        #os.mkdir(save_path1)
        os.mkdir(save_path2)

    files = glob.glob('./train_data/train_image_samples/*')
    print(files)
    paths = np.array(
        [e for x in [glob.glob(os.path.join(file, '*')) 
        for file in files] for e in x])
    #np.random.shuffle(paths)
    print(paths)

    r = int(len(paths) * 0.999)
    train_paths = paths[:r]
    test_paths = paths[r:]

    x_train = []
    pbar = tqdm(total=(len(train_paths)))
    for i, d in enumerate(train_paths):
        pbar.update(1)
        img = cv2.imread(d)
        if img is None:
            continue
        x_train.append(img)
        name = "{}.png".format("{0:08d}".format(i))
        #imgpath = os.path.join('train_data/raw/train_image', name)
        #cv2.imwrite(imgpath, img)
    #print(x_train)
    pbar.close()

    x_test = []
    pbar = tqdm(total=(len(test_paths)))
    for i, d in enumerate(test_paths):
        pbar.update(1)
        img = cv2.imread(d)

        if img is None:
            continue
        x_test.append(img)
        name = "{}.png".format("{0:05d}".format(i))
        #imgpath = os.path.join('train_data/raw/test_image', name)
        #cv2.imwrite(imgpath, img)
    pbar.close()

    x_train = np.array(x_train, dtype=np.uint8)
    x_test = np.array(x_test, dtype=np.uint8)
    np.save('./train_data/npy/train_gt.npy', x_train)
    np.save('./train_data/npy/test_gt.npy', x_test)


def main():
    preprocess()


if __name__ == '__main__':
    main()

