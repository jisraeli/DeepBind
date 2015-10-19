# Copyright (c) 2015, Andrew Delong and Babak Alipanahi All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# Author's note: 
#     This file was distributed as part of the Nature Biotechnology 
#     supplementary software release for DeepBind. Users of DeepBind
#     are encouraged to instead use the latest source code and binaries 
#     for scoring sequences at
#        http://tools.genes.toronto.edu/deepbind/
# 
import sys, os
import numpy as np
import argparse
from smat import *

#os.environ["PYTHONUNBUFFERED"] = "1"  # Disable output buffering
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser(description="Train a convolutional neural net on MNIST and print out the error rates.")
parser.add_argument("--device",type=int,default=None,help="Device # to use (which GPU). Default is 0.")
parser.add_argument("--show_filters",action="store_true",default=False,help="Plot the filters after each update, showing in a popup window.")
parser.add_argument("--f64",action="store_true",default=False,help="Use float64, if supported by the GPU. Default is float32.")
args = parser.parse_args()

if args.device is not None:
    set_backend_options(device=args.device)

dt = float64 if args.f64 else float32
set_default_dtype(dt)

print "Using device", get_backend_info().device
print "Checking cuDNN ...",
try:    cudnn_dll()
except: quit("Failed to load cuDNN shared library. Quitting.");
print "OK."

##############################################################################
#  Functions for loading DATA
##############################################################################

# Load a data as of pairs (X,Y) where:
#    X is a (batchsize x inputsize) matrix of inputs
#    Y is a (batchsize x outputsize) matrix of corresponding targets
#  MNIST has 60000 training examples total.
with np.load("data/mnist/mnist_train.npz") as mnist_file:
    inputs_train  = asarray(mnist_file['X'], dtype=dt)/255   # Load 60000 x 784 matrix of training inputs, scaled to range [0,1]
    targets_train = asarray(mnist_file['Y'], dtype=dt)       # Load 60000 x 10 matrix of training targets

with np.load("data/mnist/mnist_test.npz") as mnist_file:
    inputs_test  = asarray(mnist_file['X'], dtype=dt)/255    # Load 10000 x 784 matrix of testing inputs, scaled to range [0,1]
    targets_test = asarray(mnist_file['Y'], dtype=dt)        # Load 10000 x 10 matrix of testing targets

# Generate minibatches out of the full dataset.
trainsize = len(inputs_train)
testsize  = len(inputs_test)
batchsize = 100            ; assert trainsize % batchsize == 0 # Make sure we use all training examples
batches_train = [(inputs_train[i:i+batchsize], targets_train[i:i+batchsize]) for i in range(0, trainsize, batchsize)]
batches_test  = [(inputs_test [i:i+batchsize], targets_test [i:i+batchsize]) for i in range(0, testsize,  batchsize)]

##############################################################################
#  CONVNET CONFIGURATION
##############################################################################

# Configuration of neural net layers (number of filters, hidden units, etc)
input_w = 28          # Width  of MNIST image
input_h = 28          # Height of MNIST image
input_c = 1           # Number of input channels for MNIST

# Layer 1 is convolution, so call it C1
C1_filter_c = 32  # Number of filters in C1
C1_filter_w = 5   # Width  of filters in C1
C1_filter_h = 5   # Height of filters in C1
C1_stride   = 1   # Stride of C1
C1_w = (input_w-C1_filter_w)//C1_stride + 1  # Width  of C1 output featuremap
C1_h = (input_h-C1_filter_h)//C1_stride + 1  # Height of C1 output featuremap

# Layer 2 is pooling, so call it P1
P1_mode = "max"   # Pooling type ("max" or "avg")
P1_window_w = 3   # Width  of pooling windows in P1
P1_window_h = 3   # Height of pooling windows in P1
P1_stride   = 2   # Stride of P1
P1_w = (C1_w-P1_window_w)//P1_stride + 1  # Width  of P1 output featuremap
P1_h = (C1_h-P1_window_h)//P1_stride + 1  # Height of P1 output featuremap

# Layer 3 is fully connected, so call it F1
F1_size = 1000         # Number of neurons in F1
F1_dropout_rate = 0.5  # Dropout rate for F1

# Layer 4 is fully connected softmax, so call it F2
F2_size = 10      # 10 classes representing digits 0..9

##############################################################################
#  CONVNET PARAMETER ALLOCATION and INITIALIZATION
##############################################################################

# Count how many parameters there are in total for each later,
# so that we can allocate one big vector P for all parameters
num_params = { 'C1_weights' : C1_filter_c*C1_filter_w*C1_filter_h * (input_c),
               'C1_bias'    : C1_filter_c,
               'F1_weights' : F1_size * (C1_filter_c*P1_w*P1_h),
               'F1_bias'    : F1_size,
               'F2_weights' : F2_size * (F1_size),
               'F2_bias'    : F2_size }
num_params_total = sum(num_params.values())
P = zeros(num_params_total, dt)
P_grad = zeros_like(P)

# Slice the param vector P into parameters for each layer.
_ = 0  # temp counter
C1_weights = P[_:_+num_params['C1_weights']].reshape((C1_filter_c,  input_c*C1_filter_h*C1_filter_w)); _ += C1_weights.size;
C1_bias    = P[_:_+num_params['C1_bias'   ]];                                                          _ += C1_bias.size;
F1_weights = P[_:_+num_params['F1_weights']].reshape((C1_filter_c*P1_w*P1_h, F1_size));                _ += F1_weights.size;
F1_bias    = P[_:_+num_params['F1_bias'   ]].reshape((1,         F1_size));                            _ += F1_bias.size;
F2_weights = P[_:_+num_params['F2_weights']].reshape((F1_size,   F2_size));                            _ += F2_weights.size;
F2_bias    = P[_:_+num_params['F2_bias'   ]].reshape((1,         F2_size));                            _ += F2_bias.size;
assert _ == num_params_total

# Slice the gradient vector P_grad into parameters for each layer.
_ = 0
C1_weights_grad = P_grad[_:_+num_params['C1_weights']].reshape((C1_filter_c,  input_c*C1_filter_h*C1_filter_w)); _ += C1_weights_grad.size;
C1_bias_grad    = P_grad[_:_+num_params['C1_bias'   ]];                                                          _ += C1_bias_grad.size;
F1_weights_grad = P_grad[_:_+num_params['F1_weights']].reshape((C1_filter_c*P1_w*P1_h, F1_size));                _ += F1_weights_grad.size;
F1_bias_grad    = P_grad[_:_+num_params['F1_bias'   ]].reshape((1,         F1_size));                            _ += F1_bias_grad.size;
F2_weights_grad = P_grad[_:_+num_params['F2_weights']].reshape((F1_size,   F2_size));                            _ += F2_weights_grad.size;
F2_bias_grad    = P_grad[_:_+num_params['F2_bias'   ]].reshape((1,         F2_size));                            _ += F2_bias_grad.size;
assert _ == num_params_total

# Initialize parameters in P for each layer with a different random scale.
def set_rand(target, scale):
    target.ravel()[:] = randn(target.size) * scale
set_rand(C1_weights, 0.01);    C1_bias += 0.0001;    # Initialize biases as small positive values
set_rand(F1_weights, 0.01);    F1_bias += 0.0001;
set_rand(F2_weights, 0.01);

##############################################################################
#  CONVNET FORWARDPROP / BACKPROP
##############################################################################

# Send input mini-batch through our convnet.
# Returns final outputs and, if targets are given, returns gradients as well.
def eval_convnet(inputs, targets=None):

    # Network parameters stored in global variables, for simplicity of this demo
    global C1_weights, C1_bias
    global F1_weights, F1_bias
    global F2_weights, F2_bias

    # Gradients of parameters also stored in global variables, for simplicity
    global C1_weights_grad, C1_bias_grad
    global F1_weights_grad, F1_bias_grad
    global F2_weights_grad, F2_bias_grad
    
    # Forward propagate C1
    C1_hidden = relu(conv2(inputs, input_w, input_h, C1_weights, C1_filter_w, C1_filter_h, bias=C1_bias, stride=C1_stride))

    # Forward propagate P1
    P1_hidden = pool2(P1_mode, C1_hidden, C1_w, C1_h, P1_window_w, P1_window_h, stride=P1_stride)

    # Forward propagate F1
    F1_hidden = relu(dot(P1_hidden, F1_weights) + F1_bias)
    F1_hidden, F1_mask = dropout(F1_hidden, F1_dropout_rate, test_mode=(targets is None))

    # Forward propagate F2
    F2_hidden = softmax(dot(F1_hidden, F2_weights) + F2_bias)

    # If no targets provided (no gradient requested), just return the predictions of final layer
    if targets is None:
        return F2_hidden

    # Compute residuals
    F2_delta = F2_hidden - targets

    # Backward propagate F2_delta
    F2_bias_grad[:] = sum(F2_delta, axis=0)
    F2_weights_grad[:] = dot_tn(F1_hidden, F2_delta)
    F1_delta = dot_nt(F2_delta, F2_weights)

    # Backward propagate F1_delta
    F1_delta = dropout_grad(F1_delta, F1_mask)  # Backprop through dropout after F1 layer
    F1_delta *= relu_grad(F1_hidden)            # Backprop through relu after F1 layer
    F1_bias_grad[:] = sum(F1_delta, axis=0)
    F1_weights_grad[:] = dot_tn(P1_hidden, F1_delta)
    P1_delta = dot_nt(F1_delta, F1_weights)

    # Backward propagate P1_delta
    C1_delta = pool2_grad(P1_mode, C1_hidden, C1_w, C1_h, P1_window_w, P1_window_h, P1_hidden, P1_delta, stride=P1_stride)

    # Backward propagate C1_delta
    C1_delta *= relu_grad(C1_hidden)  # Backprop through relu after C1 layer
    conv2_biasgrad(C1_bias, C1_delta, C1_bias_grad)
    conv2_filtersgrad(inputs, input_w, input_h, C1_filter_w, C1_filter_h, C1_delta, C1_weights_grad, stride=C1_stride)

##############################################################################
#  Functions for PRINTING
##############################################################################

def make_filter_grid(filter_weights, filter_w, filter_h, max_cols=8):
    
    # Determine the range [-vmin, vmax] to map to [0, 255]
    vmin = float(filter_weights.min())
    vmax = float(filter_weights.max())
    vmax, vmin = max(vmax, -vmin), min(-vmax, vmin)

    # Scale all filters to range [0, 255] and reshape to a single (n, width, height) array
    images = ((filter_weights.asnumpy() - vmin) / (vmax - vmin) * 255).astype(np.uint8).reshape((-1, filter_h, filter_w))
    n = len(images)

    # Create a big image to store the filters, then copy each filter into its slot in the grid
    num_cols = min(n, max_cols)
    num_rows = (n + num_cols - 1) // num_cols
    grid = np.zeros((num_rows*(filter_h+1)+1, num_cols*(filter_w+1)+1), np.uint8)
    for col in range(num_cols):
        for row in range(num_rows):
            i = row*num_cols + col
            if i < len(images):
                grid[1+row*(filter_h+1):(row+1)*(filter_h+1),
                     1+col*(filter_w+1):(col+1)*(filter_w+1)] = images[i]

    return grid


def error_rate(batches):
    predictions = np.vstack([eval_convnet(inputs).asnumpy() for inputs, targets in batches])
    targets     = np.vstack([targets.asnumpy()              for inputs, targets in batches])
    num_errors  = np.sum( predictions[np.where(targets==1)] != predictions.max(axis=1) )
    return 100.*num_errors/len(predictions)


filter_plot_img = None   # Global variable to hold reference to the filter image currently being plotted in a pyplot window
def plot_filter_grid():
    global filter_plot_img
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    filter_grid = make_filter_grid(C1_weights, C1_filter_w, C1_filter_h)
    if filter_plot_img is None:
        # The first time we plot the filters, use imshow and pop up a new window
        filter_plot_img = plt.imshow(filter_grid, cmap=cm.Greys_r, interpolation="NEAREST")
        plt.show(block=False)
    else:
        # When we want to just update the filters, replace the data and give pyplot event loop a chance to draw
        filter_plot_img.set_data(filter_grid)
        plt.pause(0.001)


def print_status(epoch=None):
    update_interval = 5
    if epoch is not None:
        print ".",
    if epoch is None or (epoch+1) % update_interval == 0:  # Only print status every 5 epochs.
        time_per_epoch = toc() / update_interval
        train_error = error_rate(batches_train)
        test_error  = error_rate(batches_test)
        status_msg = "start" if epoch is None else ("epoch[%d]"% (epoch+1))
        time_msg   = "(%.2fs/epoch)" % time_per_epoch if epoch is not None else ""
        print "\n%s: %.2f%% train err, %.2f%% test err %s " % (status_msg, train_error, test_error, time_msg),
        if args.show_filters:
            plot_filter_grid()
        tic()

tic()
print_status()

##############################################################################
#  SGD TRAINING LOOP
##############################################################################

# Parameters of SGD training
num_epoch  = 50
learn_rate = 0.02
momentum   = 0.90

# Allocate array to store momentum of every parameter; updated during training
P_momentum = zeros_like(P)

tic("training time")

# Start training!
for epoch in range(num_epoch):

    for inputs, targets in batches_train:

        # Generate compute per-layer gradient based on targets
        eval_convnet(inputs, targets)

        # Gradient step with very basic momentum
        P_grad *= -learn_rate/batchsize
        P_momentum *= momentum
        P_momentum += P_grad
        P += P_momentum

    # Print current classification error on training data
    print_status(epoch)
    
print "\nTotal training time = %.1fs" % toc("training time")
