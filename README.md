# caffe
  ## examples: This is a tutorial for caffe
   ### modifying all the ipynb files in caffe/examples into python files.
     00_classification.py   <== modified from caffe/examples/00-classification.ipynb
     In this example we'll classify an image with the bundled CaffeNet model 
        work space:  /home/huht/wk_caffe/examples
        input image:  images/cat.jpg
        net config:   models/bvlc_reference_caffenet/deploy.prototxt
        trained CaffeNet model weights: models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
     
     01_learning_lenet.py    <== modified from caffe/examples/01-learning_lenet.ipynb
     In this example, we'll explore learning with Caffe in Python, using the fully-exposed Solver interface.
        work space:                                    /home/huht/wk_caffe/examples
        input data and downloading data shell script:  data/mnist
        net config, solver, lmdb, training shell script, trained model ==>   01_learning_lenet_mnists
        
     detection.ipynb     <== modified from caffe/examples/detection.ipynb after panda updating
        df.sort(), df.order()  ==> df.sort_values() 
       
        
