#!/usr/bin/python
# -*- coding: UTF-8 -*-

## author: huht


import numpy as np
import matplotlib.pyplot as plt
import os
import sys

caffe_root = '/home/huht/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe

import threading
class ThreadClass(threading.Thread):
    def __init__(self, image):
        threading.Thread.__init__(self)
        self.image = image       
    def run(self):
        plt.imshow(self.image)
        plt.show()
        
#############################################################
####
####       download data and transfer them into lmdb
###
#############################################################

wk_caffe = '/home/huht/wk_caffe/examples/'    ## work_shop directory

######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##        loading minist data
######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if os.path.isfile(wk_caffe + 'data/mnist/train-images-idx3-ubyte') and \
   os.path.isfile(wk_caffe + 'data/mnist/train-labels-idx1-ubyte') and \
   os.path.isfile(wk_caffe + 'data/mnist/t10k-images-idx3-ubyte') and \
   os.path.isfile(wk_caffe + 'data/mnist/t10k-labels-idx1-ubyte'):
    print 'mnist data found ...'
else:
    os.chdir(wk_caffe)
    os.system('data/mnist/get_mnist.sh')

######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##        transform mnist data to lmdb
######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.chdir(wk_caffe)
if os.path.isfile(wk_caffe + '01_learning_lenet_mnist/mnist_test_lmdb/data.mdb') and \
   os.path.isfile(wk_caffe + '01_learning_lenet_mnist/mnist_test_lmdb/lock.mdb') and \
   os.path.isfile(wk_caffe + '01_learning_lenet_mnist/mnist_train_lmdb/data.mdb') and \
   os.path.isfile(wk_caffe + '01_learning_lenet_mnist/mnist_train_lmdb/lock.mdb'):
    print 'mnist lmdb found ...'
else:
    os.system('01_learning_lenet_mnist/create_mnist.sh')


#############################################################
####
####    create the net config, output train/test prototxt files
###
#############################################################
from caffe import layers as L, params as P

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)
    
    return n.to_proto()
    
with open('01_learning_lenet_mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('01_learning_lenet_mnist/mnist_train_lmdb', 64)))
    
with open('01_learning_lenet_mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('01_learning_lenet_mnist/mnist_test_lmdb', 100)))



#############################################################
####
####    running according to lenet_auto_solver.prototxt
###
#############################################################
    
## show lenet_auto_solver.prototxt
with open('01_learning_lenet_mnist/lenet_auto_solver.prototxt', 'r') as f:
    print f.read()

## running   
'''
it is better run this in the terminal directly
'''
os.system('01_learning_lenet_mnist/train_lenet.sh')










