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
import numpy as np
import argparse
from smat import *

parser = argparse.ArgumentParser(description="Train a 784-600-400-10 neural net on MNIST and print out the error rates.")
parser.add_argument("--activation",type=str,dest="activation",metavar="[logistic|tanh|relu]",default="relu",help="Activation function to use. Default is relu.")
parser.add_argument("--device",type=int,default=None,help="Device # to use (which GPU). Default is 0.")
parser.add_argument("--f64",action="store_true",default=False,help="Use float64, if supported by the GPU. Default is float32.")
args = parser.parse_args()

if args.activation == "logistic":
    def func(A):          return 1./(1+exp(-A))  # logistic sigmoid activation function, returns Z=logistic(A)
    def dfunc(Z):         return Z-Z**2          # derivative d/dx(logistic)(x) = logistic(x)-logistic(x)^2
elif args.activation == "tanh":
    def func(A):          return tanh(A)         # tanh sigmoid activation function, returns Z=tanh(A)
    def dfunc(Z):         return 1-Z**2          # derivative d/dx(tanh)(x) = 1-tanh(x)^2
elif args.activation == "relu":
    def func(A):          return maximum(0, A)   # 'rectified linear' (relu) activation function, returns Z=max(0,A)
    def dfunc(Z):         return sign(Z)         # derivative d/dx(relu)(x) = sign(max(0,x))
else:
    quit("Unrecognized activation function \"%s\"." % args.activation)

if args.device is not None:
    set_backend_options(device=args.device)

dt = float64 if args.f64 else float32
set_default_dtype(dt)

print "Using device", get_backend_info().device

##############################################################################
#  Functions for loading DATA
##############################################################################

# Load a data as of pairs (X,Y) where:
#    X is a (batchsize x inputsize) matrix of inputs
#    Y is a (batchsize x outputsize) matrix of corresponding targets
#  MNIST has 60000 training examples total.
with np.load("data/mnist/mnist_train.npz") as mnist_file:
    Xtrain = asarray(mnist_file['X'], dtype=dt)/255*2-1   # Load 60000 x 784 matrix of training inputs, scaled to range [-1,1]
    Ytrain = asarray(mnist_file['Y'], dtype=dt)           # Load 60000 x 10 matrix of training targets

with np.load("data/mnist/mnist_test.npz") as mnist_file:
    Xtest  = asarray(mnist_file['X'], dtype=dt)/255*2-1   # Load 10000 x 784 matrix of testing inputs, scaled to range [-1,1]
    Ytest  = asarray(mnist_file['Y'], dtype=dt)           # Load 10000 x 10 matrix of testing targets

##############################################################################
#  Script for TRAINING
##############################################################################

# Generate minibatches out of the full dataset.
trainsize = Xtrain.shape[0]
batchsize = 150            ; assert trainsize % batchsize == 0 # Make sure we use all training examples
batches   = [(Xtrain[i:i+batchsize],Ytrain[i:i+batchsize]) for i in range(0,trainsize,batchsize)]

# Size of each neural net layer
inputsize  = 28*28   # MNIST dataset is 28x28 pixel images, so 784 inputs
layersize1 = 600     # Number of neurons in first layer (filters)
layersize2 = 400     # Number of neurons in second layer
outputsize = 10      # 10 classes representing digits 0..9

# Parameters of the network
def randweights(n, m):
    return rand(n, m)*0.002-0.001  # initialize to small values [-0.001,0.001]
W1 = randweights(inputsize, layersize1);  b1 = randweights(1, layersize1)
W2 = randweights(layersize1, layersize2); b2 = randweights(1, layersize2)
W3 = randweights(layersize2, outputsize); b3 = randweights(1, outputsize)

# Evaluate our 3-layer neural network using weights W1,W2,W3 above.
# Returns final outputs and, if targets Y are given, returns gradients as well.
def nnet_eval(X, Y=None):
    global W1,W2,W3,b1,b2,b3
    
    # Forward propagate minibatch inputs X to generate predictions Z3
    Z1 = func(dot( X, W1) + b1);         # Z1 = outputs for layer 1
    Z2 = func(dot(Z1, W2) + b2);         # Z2 = outputs for layer 2
    Z3 = softmax(dot(Z2, W3) + b3);      # Z3 = outputs for layer 3 (final output)

    if Y is None:
        return Z3  # If no gradient requested, just return the predictions

    # Backward propagate error between Z3 and targets Y
    D3 = (Z3-Y)/batchsize             # Backprop prediction error as delta to layer 3
    D2 = dfunc(Z2)*dot_nt(D3, W3)     # Backprop layer 3 deltas to layer 2
    D1 = dfunc(Z1)*dot_nt(D2, W2)     # Backprop layer 2 deltas to layer 1

    # Compute gradient of training error w.r.t. network weights
    dW3 = dot_tn(Z2, D3);    db3 = sum(D3, axis=0)  # Gradient w.r.t. W3
    dW2 = dot_tn(Z1, D2);    db2 = sum(D2, axis=0)  # Gradient w.r.t. W2
    dW1 = dot_tn( X, D1);    db1 = sum(D1, axis=0)  # Gradient w.r.t. W1

    return Z3,dW1,dW2,dW3,db1,db2,db3            # Return predictions and gradients

##############################################################################
#  Functions for PRINTING
##############################################################################

def error_rate(X, Y):
    Z = nnet_eval(X).asnumpy()
    num_errors = np.sum( Z[np.where(Y.asnumpy()==1)] != Z.max(axis=1) )
    return 100.*num_errors/X.shape[0]

def print_status(epoch=None):
    update_interval = 10
    if epoch is None or (epoch+1) % update_interval == 0:  # Only print status every 5 epochs.
        time_per_epoch = toc() / update_interval
        train_error = error_rate(Xtrain, Ytrain)
        test_error  = error_rate(Xtest, Ytest)
        status_msg = "start" if epoch is None else ("epoch[%d]"% (epoch+1))
        time_msg   = "(%.2fs/epoch)" % time_per_epoch if epoch is not None else ""
        print "%s: %.2f%% train err, %.2f%% test err %s" % (status_msg, train_error, test_error, time_msg)
        tic()

tic()
print_status()

##############################################################################
#  TRAINING LOOP
##############################################################################

# Parameters of SGD training
num_epoch  = 50
learn_rate = 0.02
momentum   = 0.90

# Current momentum of the weights
mW1 = zeros_like(W1); mb1 = zeros_like(b1)
mW2 = zeros_like(W2); mb2 = zeros_like(b2)
mW3 = zeros_like(W3); mb3 = zeros_like(b3)

tic("training time")

# Start training!
for epoch in range(num_epoch):

    for X,Y in batches:

        # Generate predictions Z, along with per-layer gradient based on targets Y
        Z,dW1,dW2,dW3,db1,db2,db3 = nnet_eval(X, Y)

        # Gradient step with very basic momentum
        for P,dP,mP in zip(( W1,  W2,  W3,  b1,  b2,  b3),
                           (dW1, dW2, dW3, db1, db2, db3),
                           (mW1, mW2, mW3, mb1, mb2, mb3)):
            dP *= -learn_rate
            mP *= momentum
            mP += dP
            P  += mP

    # Print current classification error on training data
    print_status(epoch)
    
print "Total training time = %.1fs" % toc("training time")