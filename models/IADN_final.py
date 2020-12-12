import tensorflow as tf
import numpy as np
import sys
sys.path.append('../utils')
#sys.path.append('../vgg19')
from layer import *
from spp_layer import *
from BasicConvLSTMCell import *
from RGBtoYCBCR import *
from ssim_loss import *

class Model:
    def __init__(self, x, x_rain, is_training, batch_size, spt):
        self.spt = spt
        self.batch_size = batch_size
        n,w,h,c = x_rain.get_shape().as_list()
        self.weight = w//4
        self.height = h//4
        self.stage_num = 10
        x_rain_noise = x_rain + tf.random_normal(shape=tf.shape(x_rain), stddev= 3 / 255.0)
        self.rain_res = self.generator(x_rain_noise, is_training, False)
        #self.rain_real = x_rain - x
        self.imitation = x_rain - self.rain_res# + self.res_infor
        self.all_loss, self.derain_loss, self.SSIM_LOSS = self.inference_losses(x, self.imitation)
		
    def generator(self, rain, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):

            with tf.variable_scope('ini'):#1.68kb
                rain_ini = deconv_layer(
                    rain, [3, 3, 32, 3], [self.batch_size, self.weight*4, self.height*4, 32], 1)
                rain_ini = prelu(rain_ini)
                
            long_connection = res_in = rain_ini
            rain_image = []
            # stage number
            for n in range(self.stage_num):# 37.5
                ############ image_r
                ############ image_r
                with tf.variable_scope('image_r_stage_{}'.format(n)):#1.68kb
                    with tf.variable_scope('FUSE_r_{}'.format(n)):
                        FUSE = conv_layer(tf.concat([rain_ini,res_in], 3), [1, 1, 64, 64], 1)####9KB
                        FUSE = prelu(FUSE)
                    with tf.variable_scope('down1_r_{}'.format(n)):
                        r_down1 = conv_layer(FUSE, [3, 3, 64, 32], 2)####9KB
                        r_down1 = prelu(r_down1)
                    res_short1 = res_input1 = r_down1
                    for m in range(3):
                        with tf.variable_scope('group_{}_RCAB{}'.format(n+1,m+1)):
                            res_input1 = self.RCABE(res_input1, 4)
                    # with tf.variable_scope('fuse_{}'.format(n)):#144
                        # select_fea = deconv_layer(
                            # res_short1, [1, 1, 32, 32], [self.batch_size, self.weight*2, self.height*2, 32], 1)#2
                        # select_fea = prelu(select_fea)
                    with tf.variable_scope('up_r_{}'.format(n)):#144
                        res_mem_out1 = deconv_layer(
                            res_input1, [3, 3, 32, 32], [self.batch_size, self.weight*4, self.height*4, 32], 2)#2
                        res_mem_out1 = prelu(res_mem_out1)
                    res_in += res_mem_out1
                    with tf.variable_scope('stage_r_{}'.format(n)):#144
                        rain_stage = deconv_layer(
                            res_mem_out1, [3, 3, 3, 32], [self.batch_size, self.weight*4, self.height*4, 3], 1)#2
                        #rain_stage = prelu(rain_stage)
                rain_image.append(rain_stage)
            rain_cat = tf.concat([rain_ for rain_ in rain_image], 3)
            #fuse_fea = tf.concat([stage1,stage2,stage3],3)   
            # channel attention
            with tf.variable_scope('channel_attention'):
                with tf.variable_scope('channel_1'):
                    channel = prelu(deconv_layer(rain_cat, [3, 3, self.stage_num*3, self.stage_num*3], [self.batch_size, self.weight*4, self.height*4, self.stage_num*3], 1))
                with tf.variable_scope('channel_2'):
                    channel = prelu(deconv_layer(channel, [3, 3, self.stage_num*3, self.stage_num*3], [self.batch_size, self.weight*4, self.height*4, self.stage_num*3], 1))
                channel_att = tf.reduce_mean(channel, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
                #channel_att = tf.reshape(channel_att,[-1,1,1,64])
                with tf.variable_scope('channel_3'):
                    channel_att = prelu(deconv_layer(channel_att, [1, 1, 16, self.stage_num*3], [self.batch_size, 1, 1, 16], 1))
                with tf.variable_scope('channel_4'):
                    channel_att = deconv_layer(channel_att, [1, 1, self.stage_num*3, 16], [self.batch_size, 1, 1, self.stage_num*3], 1)
                    channel_att = tf.nn.sigmoid(channel_att)
                channel_fea = channel_att*rain_cat + rain_cat
            # pixel attention
            with tf.variable_scope('pixel_attention'):
                with tf.variable_scope('ud8'):
                    pixel = prelu(deconv_layer(channel_fea, [3, 3, self.stage_num*3, self.stage_num*3], [self.batch_size, self.weight*4, self.height*4, self.stage_num*3], 1))
                with tf.variable_scope('ud9'):
                    pixel = prelu(deconv_layer(pixel, [3, 3, self.stage_num*3, self.stage_num*3], [self.batch_size, self.weight*4, self.height*4, self.stage_num*3], 1))
                with tf.variable_scope('ud10'):
                    pixel_att = prelu(deconv_layer(pixel, [1, 1, 16, self.stage_num*3], [self.batch_size, self.weight*4, self.height*4, 16], 1))
                with tf.variable_scope('ud11'):
                    pixel_att = deconv_layer(pixel_att, [1, 1, self.stage_num*3, 16], [self.batch_size, self.weight*4, self.height*4, self.stage_num*3], 1)
                    pixel_att = tf.nn.sigmoid(pixel_att)
                pixel_fea = pixel_att*channel_fea + channel_fea
                    
            with tf.variable_scope('ini_down1'):
                rain_ini_down = conv_layer(pixel_fea, [3, 3, 30, 64], 2)####9KB
                rain_ini_down = prelu(rain_ini_down)
            with tf.variable_scope('non_local1'):#1.68kb
                #b, w, h, channel = input.get_shape().as_list()
                with tf.variable_scope('f1_1'):#144
                    f1_1 = deconv_layer(
                        rain_ini_down, [1, 1, 32, 64], [self.batch_size, self.weight*2, self.height*2, 32], 1)#2
                    f1_1 = prelu(f1_1)
                f1_1_re = tf.reshape(f1_1, [-1, self.weight*2*self.height*2, 32])##### B-WH-C
                #f1_1_re_T = tf.transpose(f1_1_re, perm=[0, 2, 1]) ##### B-WH-C
                with tf.variable_scope('f1_2'):#144
                    f1_2 = deconv_layer(
                        rain_ini_down, [1, 1, 32, 64], [self.batch_size, self.weight*2, self.height*2, 32], 1)#2
                    f1_2 = prelu(f1_2)
                f1_2_T = tf.transpose(f1_2, perm=[0, 3, 1, 2])
                maxpool_f1 = tf_spatial_pyramid_pooling(f1_2_T, self.spt, tf.float32)##### B-C-S
                non_local_1 = tf.nn.softmax(tf.matmul(f1_1_re, maxpool_f1))#B-WH-S 矩阵相乘
                #print(non_local_1.shape)
				
                with tf.variable_scope('g_1'):#144
                    g_1 = deconv_layer(
                        rain_ini_down, [1, 1, 32, 64], [self.batch_size, self.weight*2, self.height*2, 32], 1)#2
                    g_1 = prelu(g_1)
                g_1_T = tf.transpose(g_1, perm=[0, 3, 1, 2])
                maxpool_g1 = tf_spatial_pyramid_pooling(g_1_T, self.spt, tf.float32)##### B-C-S
                maxpool_g1_T = tf.transpose(maxpool_g1, perm=[0, 2, 1])##### B-S-C
				
                attentions_1 = tf.matmul(non_local_1, maxpool_g1_T)#B-WH-C
                attentions_1_re = tf.reshape(attentions_1, [self.batch_size, self.weight*2, self.height*2, 32])
                with tf.variable_scope('z_1'):#144
                    z_1 = deconv_layer(
                        attentions_1_re, [1, 1, 64, 32], [self.batch_size, self.weight*2, self.height*2, 64], 1)#2
                    z_1 = prelu(z_1)
                y_1 = tf.add(rain_ini_down, z_1)
                
            with tf.variable_scope('ini_up'):#1.68kb
                y_1_up = deconv_layer(
                    y_1, [3, 3, 32, 64], [self.batch_size, self.weight*4, self.height*4, 32], 2)
                y_1_up = prelu(y_1_up)
				
            with tf.variable_scope('fusion'):
                rain_fea = deconv_layer(
                    y_1_up, [3, 3, 3, 32], [self.batch_size, self.weight*4, self.height*4, 3], 1)#2
        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return rain_fea
		
    def RCABE(self, input, reduction):
        b, w, h, channel = input.get_shape()  # (B, W, H, C)
        f = tf.layers.conv2d(input, channel, 3, padding='same', activation=lrelu)  # (B, W, H, C)tf.nn.relu
        f = tf.layers.conv2d(f, channel, 3, padding='same')  # (B, W, H, C)
        x_mean = tf.reduce_mean(f, axis=(1, 2), keep_dims=True)  # (B, 1, 1, C)
        x_stdv = tf.sqrt(tf.reduce_mean((f - x_mean) ** 2, axis=(1, 2), keep_dims=True))
        x_in = x_stdv + x_mean
        x = tf.layers.conv2d(x_in, channel // reduction, 1, activation=lrelu)  # (B, 1, 1, C // r)
        x = tf.layers.conv2d(x, channel, 1, activation=tf.nn.sigmoid)  # (B, 1, 1, C)
        x = tf.multiply(f, x)  # (B, W, H, C)
        x = tf.add(input, x)
        return x
        
    def RCAB(self, input, reduction):
        b, w, h, channel = input.get_shape()  # (B, W, H, C)
        f = tf.layers.conv2d(input, channel, 3, padding='same', activation=prelu)  # (B, W, H, C)tf.nn.relu
        f = tf.layers.conv2d(f, channel, 3, padding='same')  # (B, W, H, C)
        x = tf.reduce_mean(f, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
        x = tf.layers.conv2d(x, channel // reduction, 1, activation=prelu)  # (B, 1, 1, C // r)
        x = tf.layers.conv2d(x, channel, 1, activation=tf.nn.sigmoid)  # (B, 1, 1, C)
        x = tf.multiply(f, x)  # (B, W, H, C) pixel-wise 元素相乘
        x = tf.add(input, x)
        return x
		
    def Nonlocal(self, input):
        b, w, h, channel = input.get_shape().as_list()
        with tf.variable_scope('f1'):#144
            f1 = deconv_layer(
                input, [1, 1, channel, channel], [b, w, h, channel], 1)#2
            f1 = prelu(f1)
        f1_re = tf.reshape(f1, [-1, w*h, channel])
        #f1_re_T = tf.transpose(f1_re, perm=[0, 2, 1])
        with tf.variable_scope('f2'):#144
            f2 = deconv_layer(
                input, [1, 1, channel, channel], [b, w, h, channel], 1)#2
            f2 = prelu(f2)
        f2_re = tf.reshape(f2, [-1, w*h, channel])
        f2_re_T = tf.transpose(f2_re, perm=[0, 2, 1])
        with tf.variable_scope('g'):#144
            g = deconv_layer(
                input, [1, 1, channel, channel], [b, w, h, channel], 1)#2
            g = prelu(g)
        g_re = tf.reshape(g, [-1, w*h, channel])
        non_local = tf.nn.softmax(tf.matmul(f1_re, f2_re_T))#NXN 矩阵相乘
        attentions = tf.matmul(non_local, g_re)#NXN 矩阵相乘
        attentions_re = tf.reshape(attentions, [-1, w, h, channel])
        with tf.variable_scope('z'):#144
            z = deconv_layer(
                attentions_re, [1, 1, channel, channel], [b, w, h, channel], 1)#2
            z = prelu(z)
        y = tf.add(input, z)
        return y

    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
        """
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer
		returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        """
        out_pool_size = [8, 6, 4]
        for i in range(len(out_pool_size)):
            h_strd = h_size = math.ceil(float(previous_conv_size[0]) / out_pool_size[i])
            w_strd = w_size = math.ceil(float(previous_conv_size[1]) / out_pool_size[i])
            pad_h = int(out_pool_size[i] * h_size - previous_conv_size[0])
            pad_w = int(out_pool_size[i] * w_size - previous_conv_size[1])
            new_previous_conv = tf.pad(previous_conv, tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]]))
            max_pool = tf.nn.max_pool(new_previous_conv,
                ksize=[1,h_size, h_size, 1],
                strides=[1,h_strd, w_strd,1],
                padding='SAME')
            if (i == 0):
                spp = tf.reshape(max_pool, [num_sample, -1])
            else:
                spp = tf.concat(axis=1, values=[spp, tf.reshape(max_pool, [num_sample, -1])])

        return spp
  
    def downscale2(self, x):
        K = 2
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled
    def downscale4(self, x):
        K = 4
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled
      
    def Laplacian(self, x):
        weight=tf.constant([
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]
        ])
        frame=tf.nn.conv2d(x,weight,[1,1,1,1],padding='SAME')
        #frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame))) * 255, tf.uint8)
        return frame

    def DSRCAB(self, input, reduction):
        b, w, h, channel = input.get_shape()  # (B, W, H, C)
        #depthwise_filter = (3, 3, channel, 1)
        #pointwise_filter = (1, 1, channel, channel)
        #strides = [1, 1, 1, 1]
        #f = tf.nn.separable_conv2d(input, depthwise_filter,pointwise_filter,strides, padding='same',rate=None,name=None,data_format=None)
        #f = tf.nn.separable_conv2d(f,depthwise_filter,pointwise_filter,strides,padding='same',rate=None,name=None,data_format=None)
        width_multiplier = 1
        f = self._depthwise_separable_conv(input, channel, width_multiplier)
        f = self._depthwise_separable_conv(f, channel, width_multiplier)
        x = tf.reduce_mean(f, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
        x = tf.layers.conv2d(x, channel // reduction, 1, activation=prelu)  # (B, 1, 1, C // r)
        x = tf.layers.conv2d(x, channel, 1, activation=tf.nn.sigmoid)  # (B, 1, 1, C)
        x = tf.multiply(f, x)  # (B, W, H, C)
        x = tf.add(input, x)
        return x
	
    def _depthwise_separable_conv(self, inputs, num_pwc_filters, width_multiplier, downsample=False):##width_multiplier=1
        """ Helper function to build the depth-wise separable convolution layer.
        """
        #num_pwc_filters = np.round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1
        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                num_outputs=None,
                                                stride=_stride,
                                                depth_multiplier=1,
                                                kernel_size=[3, 3])
        #bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
        pointwise_conv = slim.convolution2d(depthwise_conv,
                                        num_pwc_filters*width_multiplier,
                                        kernel_size=[1, 1])
        #bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return pointwise_conv
        
    def inference_losses(self, x, imitation):
            
        def inference_mse_loss(frame_hr, frame_sr):
            content_base_loss = tf.reduce_mean(tf.sqrt((frame_hr - frame_sr) ** 2+(1e-3)**2))
            return tf.reduce_mean(content_base_loss)

        derain_loss = inference_mse_loss(x, imitation)
        #rain_loss = inference_mse_loss(rain_real, rain_res)
        x_edge = self.Laplacian(x)
        imitation_edge = self.Laplacian(imitation)
        edge_loss = inference_mse_loss(x_edge, imitation_edge)
        y_channel_x = RGBtoYCBCR(x)
        y_channel_imi = RGBtoYCBCR(imitation)
        SSIM_LOSS = tf_ssim(y_channel_x, y_channel_imi)

        all_loss = 1*derain_loss - 0.1*SSIM_LOSS+ 0.06*edge_loss #0.05

        return (all_loss, derain_loss, SSIM_LOSS)

