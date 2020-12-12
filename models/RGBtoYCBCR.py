import numpy as np
import math
import tensorflow as tf

# input feature maps is of the form: N-C-(WH)/(HW)
# ex. spatial_pyramid:
#	[[1, 1], [2, 2], [3, 3], [4, 5]]
# each row is a level of pyramid with nxm pooling
def rgbtoycbcr(image, dtype=np.float32):
    #assert image.ndim == 4
    batch_size = image.shape[0]
    h = image.shape[1]
    w = image.shape[2]  
    num_channels = image.shape[3] 
    in_img_type = image.dtype
    image.astype(np.float32)
    if in_img_type != np.uint8:
        #image *= 255.
        image = np.uint8(np.clip((image+1)*127.5,0,255.0))
    ima_r = image[:, :, :, 0]
    ima_g = image[:, :, :, 1]
    ima_b = image[:, :, :, 2]
    #获取亮度,即原图的灰度拷贝
    ima_y = 0.256789 * ima_r + 0.504129 * ima_g + 0.097906 * ima_b + 16
    #获取蓝色分量
    ima_cb = -0.148223 * ima_r - 0.290992 * ima_g + 0.439215 * ima_b + 128
    #获取红色分量
    ima_cr = 0.439215 * ima_r - 0.367789 * ima_g - 0.071426 * ima_b + 128
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]'''
    #if in_img_type == np.uint8:
        #ima_y = ima_y.round()
    #else:
    ima_y = ima_y/255. -1.
    ima_y_1 = np.reshape(ima_y, (-1,h,w,1))
    #print(ima_y_1.shape)
    return ima_y_1.astype(dtype)

def RGBtoYCBCR(image, dtype=tf.float32):
	return tf.py_func(rgbtoycbcr, [image], dtype)
