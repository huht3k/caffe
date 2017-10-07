#!/usr/bin/env sh
set -e

/home/huht/caffe/build/tools/caffe train --solver=01_learning_lenet_mnist/lenet_auto_solver.prototxt $@
