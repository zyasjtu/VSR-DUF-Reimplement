import numpy as np
from utils import LoadImage,Huber
import glob
import tensorflow as tf
from PIL import Image
from tensorflow.python.platform import gfile
import dataset_parser
import random
import os

T_in=7

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = './model/duf-psnr.pb'

    with gfile.FastGFile(output_graph_path,"rb") as f:
        output_graph_def.ParseFromString(f.read())
        # fix nodes
        for node in output_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op='Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        _ = tf.import_graph_def(output_graph_def,name="")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input = sess.graph.get_tensor_by_name("L_in:0")
        output = sess.graph.get_tensor_by_name("out_H:0")
        is_train=sess.graph.get_tensor_by_name('is_train:0')

        idx = range(len(dataset_parser.test_lr_list))
        random.shuffle(idx)
        for i in idx:
            rst_dir = './result/' + os.path.basename(dataset_parser.test_lr_list[i].rstrip('l/')) + 'h'
            if not (os.path.exists(rst_dir)):
                os.mkdir(rst_dir)
            print(dataset_parser.test_lr_list[i], rst_dir)
            x_data, x_data_padded = dataset_parser.get_x(dataset_parser.test_lr_list[i], T_in)
            for j in range(x_data.shape[0]):
                in_L = x_data_padded[j:j + T_in]  # select T_in frames
                in_L = in_L[np.newaxis, :, :, :, :]
                y_out = sess.run(output, feed_dict={input: in_L, is_train: False})
                Image.fromarray(np.clip(np.around(y_out[0, 0]*255), 0, 255).astype(np.uint8)).save(rst_dir + '/{:05}.bmp'.format(j))
