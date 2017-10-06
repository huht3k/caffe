#!/usr/bin/python
# -*- coding: UTF-8 -*-


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
####       load model from model zoo
###
#############################################################

wk_caffe = '/home/huht/wk_caffe/examples/'    ## work_shop directory
if os.path.isfile(wk_caffe + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    os.system('/home/huht/caffe/scripts/download_model_binary.py /home/huht/wk_caffe/examples/models/bvlc_reference_caffenet')


#############################################################
####
####       define net
###
#############################################################
model_def = wk_caffe + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = wk_caffe + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)

#############################################################
####
####       data preprocessing and loading
###
#############################################################
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)


# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})   

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(50,        # batch size: change from 10 to 50
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})   

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


image = caffe.io.load_image(wk_caffe + 'images/cat.jpg')
transformed_image = transformer.preprocess('data', image)

        
t = ThreadClass(image)
t.start()

net.blobs['data'].data[...] = transformed_image

#############################################################
####
####       running testing
###
#############################################################

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

output = net.forward()    ### 


output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()


# load ImageNet labels
labels_file = wk_caffe + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    os.system('/home/huht/wk_caffe/examples/data/ilsvrc12/get_ilsvrc_aux.sh')
    
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]










