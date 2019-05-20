# -*- coding:utf-8 -*-
import tensorflow as tf
from utils import BatchNorm,Conv3D,DynFilter3D,depth_to_space_3D,Huber
import numpy as np
import glob
from tensorflow.python.framework import graph_util
import dataset_parser
import random

# Size of input temporal radius
T_in = 7
# Upscaling factor
R = 4

# Gaussian kernel for downsampling
def gkern(kernlen=13, nsig=1.6):
    import scipy.ndimage.filters as fi
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)

h = gkern(13, 1.6)  # 13 and 1.6 for x4
h = h[:, :, np.newaxis, np.newaxis].astype(np.float32)

# Network
H_out_true=tf.placeholder(tf.float32,shape=(1,1,None,None,3),name='H_out_true') 

is_train = tf.placeholder(tf.bool, shape=[],name='is_train') # Phase ,scalar

# L_ = DownSample(H_in, h, R)
L =  tf.placeholder(tf.float32, shape=[None, T_in, None, None, 3],name='L_in')

# build model
stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]
# [1, 3, 3, 3, 64] [filter_depth, filter_height, filter_width, in_channels,out_channels]
x = Conv3D(tf.pad(L, sp, mode='CONSTANT'), [1, 3, 3, 3, 64], [1, 1, 1, 1, 1], 'VALID', name='conv1')

F = 64
G = 32
for r in range(3): 
    t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
    t = tf.nn.relu(t)
    t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

    t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
    t = tf.nn.relu(t)
    t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
               name='Rconv' + str(r + 1) + 'b')

    x = tf.concat([x, t], 4)
    F += G
for r in range(3, 6):
    t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
    t = tf.nn.relu(t)
    t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

    t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
    t = tf.nn.relu(t)
    t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
               name='Rconv' + str(r + 1) + 'b')

    x = tf.concat([x[:, 1:-1], t], 4)
    F += G

# sharen section
x = BatchNorm(x, is_train, name='fbn1')
x = tf.nn.relu(x)
x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='conv2')
x = tf.nn.relu(x)

# R
r = Conv3D(x, [1, 1, 1, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='rconv1')
r = tf.nn.relu(r)
r = Conv3D(r, [1, 1, 1, 256, 3 * 16], [1, 1, 1, 1, 1], 'VALID', name='rconv2')

# F
f = Conv3D(x, [1, 1, 1, 256, 512], [1, 1, 1, 1, 1], 'VALID', name='fconv1')
f = tf.nn.relu(f)
f = Conv3D(f, [1, 1, 1, 512, 1 * 5 * 5 * 16], [1, 1, 1, 1, 1], 'VALID', name='fconv2')

ds_f = tf.shape(f)
f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, 16])
f = tf.nn.softmax(f, dim=4)

Fx=f
Rx =r

x=L
x_c = []
for c in range(3):
    t = DynFilter3D(x[:, T_in // 2:T_in // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :], [1, 5, 5])  # [B,H,W,R*R]
    t = tf.depth_to_space(t, R)  # [B,H*R,W*R,1]
    x_c += [t]
x = tf.concat(x_c, axis=3)  # [B,H*R,W*R,3] Tensor("concat_9:0", shape=(?, ?, ?, 3), dtype=float32)

x = tf.expand_dims(x, axis=1) # Tensor("ExpandDims_3:0", shape=(?, 1, ?, ?, 3), dtype=float32)
Rx = depth_to_space_3D(Rx, R)  # [B,1,H*R,W*R,3] Tensor("Reshape_6:0", shape=(?, ?, ?, ?, ?), dtype=float32)
x += Rx # Tensor("add_18:0", shape=(?, ?, ?, ?, 3), dtype=float32) 

out_H=tf.clip_by_value(x,0,1,name='out_H')

cost=Huber(y_true=H_out_true,y_pred=out_H,delta=0.01)
psnr=tf.image.psnr(H_out_true[0,0]*255,out_H[0,0]*255,max_val=255)

learning_rate=0.001
learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32,name='learning_rate')
learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

# total train epochs
num_epochs=100


saver = tf.train.Saver()
# Session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(num_epochs):
        if global_step!=0 & np.mod(global_step,10)==0:
            sess.run(learning_rate_decay_op)

        print("-------------------------- Epoch {:3d} ----------------------------".format(global_step))
        idx = range(len(dataset_parser.train_lr_list))
        random.shuffle(idx)
        total_train_loss = 0
        total_train_psnr = 0
        for i in idx:
            x_train_data, x_train_data_padded = dataset_parser.get_x(dataset_parser.train_lr_list[i], T_in)
            _, y_train_data = dataset_parser.get_y(dataset_parser.train_hr_list[i])

            print("---------- optimize sess.run start ----------" + dataset_parser.train_lr_list[i])
            for j in range(x_train_data.shape[0]):
                in_L = x_train_data_padded[j:j + T_in]  # select T_in frames
                in_L = in_L[np.newaxis, :, :, :, :]
                sess.run(optimizer,feed_dict={H_out_true:y_train_data[j],L:in_L,is_train: True})

            print("---------- cost sess.run start -----------" + str(total_train_loss))
            for j in range(x_train_data.shape[0]):
                in_L = x_train_data_padded[j:j + T_in]  # select T_in frames
                in_L = in_L[np.newaxis, :, :, :, :]
                train_loss,train_psnr = sess.run([cost,psnr], feed_dict={H_out_true: y_train_data[j], L: in_L, is_train: True})
                total_train_loss = total_train_loss + train_loss
                total_train_psnr += train_psnr

        total_valid_loss = 0
        total_valid_psnr = 0

        for i in range(len(dataset_parser.valid_lr_list)):
            x_valid_data, x_valid_data_padded = dataset_parser.get_x(dataset_parser.valid_lr_list[i], T_in)
            _, y_valid_data = dataset_parser.get_y(dataset_parser.valid_hr_list[i])

            print("---------- valid sess.run start -----------" + str(total_valid_loss))
            for j in range(x_valid_data.shape[0]):
                in_L = x_valid_data_padded[j:j + T_in]  # select T_in frames
                in_L = in_L[np.newaxis, :, :, :, :]
                valid_loss,valid_psnr = sess.run([cost, psnr], feed_dict={H_out_true: y_valid_data[j], L: in_L, is_train: True})
                total_valid_loss = total_valid_loss + valid_loss
                total_valid_psnr += valid_psnr

        avg_train_loss=total_train_loss/100.0/len(dataset_parser.train_lr_list)
        avg_train_psnr=total_train_psnr/100.0/len(dataset_parser.train_lr_list)
        avg_valid_loss=total_valid_loss/100.0/len(dataset_parser.valid_lr_list)
        avg_valid_psnr=total_valid_psnr/100.0/len(dataset_parser.valid_lr_list)
        print("Epoch - {:2d}, avg train loss: {:.7f}, avg train psnr: {:.7f}, avg valid loss: {:.7f}, avg valid psnr: {:.7f}".format(global_step, avg_train_loss, avg_train_psnr, avg_valid_loss, avg_valid_psnr))

        if global_step==0:
            with open('./logs/pb_graph_log.txt', 'w') as f:
                f.write(str(sess.graph_def)) 
            var_list = tf.global_variables()
            with open('./logs/global_variables_log.txt','w') as f:
                f.write(str(var_list)) 

        tf.train.write_graph(sess.graph_def, '.', './checkpoint/duf_'+str(global_step)+'.pbtxt')
        saver.save(sess, save_path="./checkpoint/duf",global_step=global_step)
