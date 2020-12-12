import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../utils')
from layer import *
from IADN import Model#NRFN5YSP10
import load_rain
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start_learning_rate = 4e-4#4e-4#4e-4
batch_size = 32#16
#spt = np.array([[1, 1], [3, 3], [6, 6], [8, 8]])
spt = np.array([[1, 1], [3, 3], [6, 6]])#, [8, 8]
def train():
    x = tf.placeholder(tf.float32, [None, 96, 96, 3])#128,96
    x_rain = tf.placeholder(tf.float32, [None, 96, 96, 3])#128,96
    is_training = tf.placeholder(tf.bool, [])

    model = Model(x, x_rain, is_training, batch_size, spt)
    sess = tf.Session()
    with tf.variable_scope('IADN'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    #opt = tf.train.AdamOptimizer(learning_rate=learning_rate)#1e-5
    learning_rate=tf.train.exponential_decay(start_learning_rate,global_step, 15000, decay_rate=0.90,staircase=False)+3e-6
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(model.all_loss, global_step=global_step, var_list=model.g_variables)
    #d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
    init = tf.global_variables_initializer() 
    sess.run(init)

    # Restore the VGG-19 network
    #var = tf.global_variables()
    #vgg_var = [var_ for var_ in var if "vgg19" in var_.name]
    #saver = tf.train.Saver(vgg_var)
    #saver.restore(sess, vgg_model)

    # Restore the network
    if tf.train.get_checkpoint_state('IADN/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'IADN/epoch70')
    vars_all=tf.trainable_variables()
    print ('Params:',np.sum([np.prod(v.get_shape().as_list()) for v in vars_all]))
    # Load the data
    x_train, x_test, x_train_rain, x_test_rain = load_rain.load()

    # Train the model
    n_iter = int(len(x_train) / batch_size)
    while True:
        epoch = int(sess.run(global_step) / n_iter ) + 1#2
        print('epoch:', epoch)
        #np.random.shuffle(x_train)
        for i in tqdm(range(n_iter)):
            x_batch = normalize(x_train[i*batch_size:(i+1)*batch_size])
            x_batch_rain = normalize(x_train_rain[i*batch_size:(i+1)*batch_size])
            j = np.random.randint(0,8,size=1)
            x_batch = rotate(x_batch,j)
            x_batch_rain = rotate(x_batch_rain,j)#edge_loss, model.edge_loss
            all_loss, derain_loss, SSIM_LOSS, _ = sess.run([model.all_loss, model.derain_loss, model.SSIM_LOSS, train_op], feed_dict={x: x_batch, x_rain: x_batch_rain, is_training: True})
            format_str = ('lr: %.7f, global_step: %d, epoch: %d, all_loss: %.5f, derain_loss: %.5f, SSIM_LOSS: %.5f')
            print((format_str % (sess.run(learning_rate), sess.run(global_step), epoch, all_loss, derain_loss, SSIM_LOSS)))
        # Validate self.all_loss, self.derain_loss, self.edge_loss
        #raw = normalize(x_test[:batch_size])
        #raw_rain = normalize(x_test_rain[:batch_size])
        #rain_res, fake = sess.run([model.rain_res, model.imitation],feed_dict={x: raw, x_rain: raw_rain, is_training: False})
        #save_img([raw_rain, rain_res, fake, raw], ['rain', 'rain_res', 'clean', 'Ground Truth'], epoch)
        #save_img([raw_rain, rain_fea, stage1, stage2, stage3, fake, raw], ['rain', 'rain_fea', 'stage1', 'stage2', 'stage3', 'clean', 'Ground Truth'], epoch)

        # Save the model
        saver = tf.train.Saver()
        save_path = 'IADN'
        if epoch>0:
            saver.save(sess, os.path.join(save_path, 'epoch{}'.format(epoch)), write_meta_graph=False)

def save_img(imgs, label, epoch):
    for i in range(batch_size):
        fig = plt.figure()
        for j, img in enumerate(imgs):
            #im = np.uint8((img[i]+1)*127.5)
            im = np.uint8(np.clip((img[i]+1)*127.5,0,255.0))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            fig.add_subplot(1, len(imgs), j+1)
            plt.imshow(im)
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')
            plt.gca().get_xaxis().set_ticks_position('none')
            plt.gca().get_yaxis().set_ticks_position('none')
            plt.xlabel(label[j])
        seq_ = "{0:09d}".format(i+1)
        epoch_ = "{0:09d}".format(epoch)
        path = os.path.join('result', seq_, '{}.png'.format(epoch_))
        if os.path.exists(os.path.join('result', seq_)) == False:
            os.mkdir(os.path.join('result', seq_))
        plt.savefig(path)
        plt.close()


def normalize(images):
    return np.array([image/127.5 - 1 for image in images])
    
if __name__ == '__main__':
    train()