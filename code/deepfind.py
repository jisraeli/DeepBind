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
import sys
import argparse
import numpy as np
from os import listdir
from os.path import join


sys.path.append("libs/smat/py")
from smat import *



# change the activation function, cane be [logistic|tanh|relu]
activation_func = "relu"

# change to the desired GPU (if not default)
picked_device = None
if picked_device is not None:
    set_backend_options(device=picked_device)

# change to dt64 for double precision
dt = float32
set_default_dtype(dt)

data_root = '../data/deepfind/'

print "Started DeepFind: scoring SNVs in promoters using DeepBind predictions"
print "Training DeepFind using", get_backend_info().device

##############################################################################
#  Neural Network
##############################################################################

# Evaluate our 1-layer neural network using weights W1,W2 above.
# Returns final outputs and, if targets Y are given, returns gradients as well.

def nnet_eval(X, Y=None, mode="sigmoid"):
    global W1,W2,b1,b2
    #****************************************************************
    # Forward propagate minibatch inputs X to generate predictions Z2
    A1 = dot(X, W1) + b1    
    Z1 = f(A1)          # Z1 = outputs for layer 1
    if Y is not None:
        if dropoutrate > 0:
            mask = bernoulli(Z1.shape, 1 - dropoutrate, dtype=dt)
            Z1 *= mask
    else:
        Z1 *= 1 - dropoutrate
    A2 = dot(Z1, W2) + b2     
    Z2 = logistic(A2)    # Z2 = output

    if Y is None:
        if mode == "sigmoid":
            return Z2  # If no gradient requested, just return the predictions
        else:
            return A2  # If no gradient requested, just return the predictions before passing the sigmoid

    #****************************************************************
    # Backward propagate error between Z2 and targets Y
    D2 = (Z2-Y)/X.shape[0]         # Backprop prediction error as delta to layer 2
    D1 = df(Z1)*dot_nt(D2, W2)     # Backprop layer 2 deltas to layer 1
    if dropoutrate > 0:
        D1 *= mask                 # Apply the dropout mask

    # Compute gradient of training error w.r.t. network weights
    dW2 = dot_tn(Z1, D2) + l2norm*W2    
    db2 = sum(D2, axis=0)  # Gradient w.r.t. W2
    dW1 = dot_tn(X , D1) + l2norm*W1    
    db1 = sum(D1, axis=0)  # Gradient w.r.t. W1

    return Z2,dW1,dW2,db1,db2      # Return predictions and gradients

# Activation functions
if   activation_func == "logistic":
    def f(A):          return 1./(1+exp(-A))  # logistic sigmoid activation function, returns Z=logistic(A)
    def df(Z):         return Z-Z**2          # derivative d/dx(logistic)(x) = logistic(x)-logistic(x)^2
elif activation_func == "tanh":
    def f(A):          return tanh(A)         # tanh sigmoid activation function, returns Z=tanh(A)
    def df(Z):         return 1-Z**2          # derivative d/dx(tanh)(x) = 1-tanh(x)^2
elif activation_func == "relu":
    def f(A):          return maximum(0, A)   # 'rectified linear' (relu) activation function, returns Z=max(0,A)
    def df(Z):         return sign(Z)         # derivative d/dx(relu)(x) = sign(max(0,x))
else:
    quit("Unrecognized activation function \"%s\"." % activation_func)

# Set neural network's weights to small random values
def randweights(n, m):
    return rand(n, m)*0.002-0.001  # initialize to small values [-0.001,0.001]


##############################################################################
#  Functions for PRINTING
##############################################################################

def calc_auc(z, y, want_curve = False):
   """Given predictions z and 0/1 targets y, computes AUC with optional ROC curve"""
   z = z.ravel()
   y = y.ravel()
   assert len(z) == len(y)

# Remove any pair with NaN in y    
   m = ~np.isnan(y)
   y = y[m]
   z = z[m]
   assert np.all(np.logical_or(y==0, y==1)), "Cannot calculate AUC for non-binary targets"

   order = np.argsort(z,axis=0)[::-1].ravel()   # Sort by decreasing order of prediction strength
   z = z[order]
   y = y[order]
   npos = np.count_nonzero(y)      # Total number of positives.
   nneg = len(y)-npos              # Total number of negatives.
   if nneg == 0 or npos == 0:
       return (np.nan,None) if want_curve else 1

   n = len(y)
   fprate = np.zeros((n+1,1))
   tprate = np.zeros((n+1,1))
   ntpos,nfpos = 0.,0.
   for i,yi in enumerate(y):
       if yi: ntpos += 1
       else:  nfpos += 1
       tprate[i+1] = ntpos/npos
       fprate[i+1] = nfpos/nneg
   auc = float(np.trapz(tprate,fprate,axis=0))
   if want_curve:
       curve = np.hstack([fprate,tprate])
       return auc, curve
   return auc


def error_rate(X, Y):
    Z = np.vstack([nnet_eval(tX).asnumpy() for tX in X])
    L = np.vstack([tY.asnumpy() for tY in Y])               
    return calc_auc(Z,L)

def print_status(epoch=None):
    update_interval = 25
    if epoch is None or epoch > 0:  # Only print status every 5 epochs.
        time_per_epoch = toc() / update_interval        
        test_error  = error_rate(Xvd, Yvd)
        status_msg = "start:     " if epoch is None else ("epoch[%03d]:"% (epoch+1))
        time_msg   = "(%.2fs/epoch)" % time_per_epoch if epoch is not None else ""
        if epoch is None or (epoch+1) % update_interval == 0: 
            train_error = error_rate(Xtr, Ytr)
            print "\t%s %.2f%% train auc, %.2f%% validation auc %s" % (status_msg, 100*train_error, 
                                                                     100*test_error, time_msg)
            sys.stdout.flush()
            tic()
        return test_error


##############################################################################
#  Functions for loading DATA
##############################################################################

add_cons   = True  # add conservation information
add_feats  = True  # add dist to TSS and is_transversion features
add_tfs    = True  # add predictions from TFs for wildtype and mutant
do_masking = True  # match genes of positive and negative sets




# input files and directories
pos_dir        = join(data_root, 'tfs_simulated')
neg_dir        = join(data_root, 'tfs_derived')
cons_pos_file  = join(data_root, 'simulated_cons.npz')
cons_neg_file  = join(data_root, 'derived_cons.npz')
mask_file      = join(data_root, 'matching_mask.npz')
pos_feat_file  = join(data_root, 'simulated_feats.npz') 
neg_feat_file  = join(data_root, 'derived_feats.npz')


tfs_to_consider = join(data_root, 'TFs_to_consider.txt')
with open(tfs_to_consider) as fh:
    considered_files = [line.strip('\r\n') for line in fh]

# function for loading the TF predictions in NPZ files
def load_data(save_dir, considered_files=None):
    files = listdir(save_dir)
    factor_names = []
    
    if considered_files is None:
        dim = 2*(len(files))
    else:
        dim = 2*(len(considered_files))

    cnt = 0
    for file_name in sorted(files):
        # ignore the TFs that are not picked       
        if considered_files is not None and file_name[:-4] not in considered_files:
            continue
            
        factor_names.append(file_name[:-4]) # remove the .npz suffix
        with np.load(join(save_dir, file_name)) as data:
            p = data['pred']
            if cnt == 0:
                # initiliaze the feature matrix
                X = np.empty((p.shape[0]/2,dim))
            X[:,2*cnt]   = p[::2]               # wild type predictions
            X[:,2*cnt+1] = p[1::2] - p[::2]     # mutant - wildtype predictions
        cnt += 1    
    return X, factor_names

print "*******************************************************************"
print 'Loading the data...'
# Loading Training data
pX, pfactors = load_data(pos_dir, considered_files)
nX, nfactors = load_data(neg_dir, considered_files)

print 'Adding prediction for %d TFs' % len(pfactors)
for pf, nf in zip(pfactors, nfactors):
    if not pf == nf:
        print 'Mismatched TFs!'    

# Combine psoitive and negative sets            
X = np.vstack([pX, nX])
Y = np.vstack([np.ones((pX.shape[0],1)), np.zeros((nX.shape[0],1))])



# Add conservation
if add_cons:
    print 'Adding conservation'
    with np.load(cons_pos_file) as data:
        pC = data['cons']
    with np.load(cons_neg_file) as data:
        nC = data['cons']                 
    C = np.vstack((pC, nC))    
    # add conservation information to TF features
    X = np.hstack((X, C))    

# Add two features: transversion_flag for mutations
# and the normalized distance to the closest TSS in (0, 1]
if add_feats:
    print 'Adding two extra features'
    with np.load(pos_feat_file) as data:
        pF = data['feats']
    with np.load(neg_feat_file) as data:
        nF = data['feats']        
    X = np.hstack([X, np.vstack([pF, nF])])    
        
# Applying the mask and matching the genes in the
# simulated and derived alleles    
if do_masking:
    print 'Matching genes'
    with np.load(mask_file) as data:
        c_mask = np.hstack([data['pos_mask'], data['neg_mask']])                            
    X = X[c_mask]
    Y = Y[c_mask]
    
num, dim = X.shape               
print 'Data is loaded\nsample size: %d, dimensions:%d' % (num, dim)

# randomly permuting the data
np.random.seed(1234)
shuffled_idx = np.random.permutation(num)
X[:] = X[shuffled_idx]
Y[:] = Y[shuffled_idx]

# Setting up cross-validation indices
num_fold = 5
sp = np.linspace(0, num, num_fold+1).astype(np.int)
splits = np.empty((num_fold, 2), dtype=np.int)
for i in range(num_fold):
    splits[i,:] = [sp[i], sp[i+1]]
    
def split_data(splits, fold_id, num_fold):
    all_splits = set(np.arange(num_fold))
    ts = np.mod(num_fold-1+fold_id, num_fold)
    vd = np.mod(num_fold-2+fold_id, num_fold)
    tr = list(all_splits - set([ts, vd]))
    idx_ts = np.arange(splits[ts,0],splits[ts,1])
    idx_vd = np.arange(splits[vd,0],splits[vd,1])
    idx_tr = np.arange(splits[tr[0],0],splits[tr[0],1])
    for i in range(1, len(tr)):
        idx_tr = np.hstack([idx_tr, np.arange(splits[tr[i],0],splits[tr[i],1])])
    return idx_tr, idx_vd, idx_ts


# Perform cross validation on the data
# Parameters of SGD training
batchsize   = 128
num_epoch   = 250
learn_rate  = 5e-4
momentum    = 0.90
dropoutrate = 0.5
layersize1  = 200      # Number of neurons in first layer 
outputsize  = 1        # 2 classes, 1 sigmoid
l2norm      = 1e-6
inputsize   = dim

# Holding folds' information 
perfs    = list()
test_auc = list()

tic("total time")
Y_hat = np.empty(num)
for fold_id in range(num_fold):
    best_va_auc = 0.    
    test_fold = np.mod(num_fold-1+fold_id, num_fold)+1
    print "*******************************************************************"
    print "Training a neural network and testing on fold %d" % (test_fold)
    tic("fold time")
    temp_perfs  = list()
    
    # weights and biases and their momentums
    W1  = randweights(inputsize, layersize1);   b1  = randweights(1, layersize1)    
    W2  = randweights(layersize1, outputsize);  b2  = randweights(1, outputsize) 
    mW1 = zeros_like(W1);                       mb1 = zeros_like(b1)
    mW2 = zeros_like(W2);                       mb2 = zeros_like(b2)

    # form the GPU based testing and trainng sets
    # Uploading the Training data to GPU
    idx_tr, idx_vd, idx_ts = split_data(splits, fold_id, num_fold)
    
    num_tr = idx_tr.shape[0]
    num_ts = idx_ts.shape[0]
    num_vd = idx_vd.shape[0]
    
    Xtr = None; Xts = None; Xvd = None
    Ytr = None; Yts = None; Yvd = None
    
    Xtr = [asarray(X[idx_tr[i:np.min([i+batchsize, num_tr])]], dtype=dt) for i in range(0, num_tr, batchsize)]
    Ytr = [asarray(Y[idx_tr[i:np.min([i+batchsize, num_tr])]], dtype=dt) for i in range(0, num_tr, batchsize)]
    Xts = [asarray(X[idx_ts[i:np.min([i+batchsize, num_ts])]], dtype=dt) for i in range(0, num_ts, batchsize)]
    Yts = [asarray(Y[idx_ts[i:np.min([i+batchsize, num_ts])]], dtype=dt) for i in range(0, num_ts, batchsize)]
    Xvd = [asarray(X[idx_vd[i:np.min([i+batchsize, num_vd])]], dtype=dt) for i in range(0, num_vd, batchsize)]
    Yvd = [asarray(Y[idx_vd[i:np.min([i+batchsize, num_vd])]], dtype=dt) for i in range(0, num_vd, batchsize)]
    
    print_status()
    tic("training time")
    ##############################################################################
    #  TRAINING LOOP
    ##############################################################################
    # Start training!
    for epoch in range(num_epoch):
        nb = len(Xtr)   # number of minibatches
        rand_b = np.random.permutation(nb)
        for j in range(nb):
            i  = rand_b[j]
            tX = Xtr[i]
            tY = Ytr[i]
            # Generate predictions Z, along with 
            # per-layer gradient based on targets Y
            Z,dW1,dW2,db1,db2 = nnet_eval(tX, tY)
            # Gradient step with very basic momentum
            for P,dP,mP in zip(( W1,  W2,  b1,  b2),
                               (dW1, dW2, db1, db2),
                               (mW1, mW2, mb1, mb2)):
                dP *= -learn_rate
                mP *= momentum
                mP += dP
                P  += mP
        # store the parameters at each epoch            
        va_auc = print_status(epoch)
        if va_auc > best_va_auc:
            best_params = [W1.copy(), W2.copy(), b1.copy(), b2.copy()]
            best_va_auc = va_auc
        temp_perfs.append(va_auc)
    
    W1, W2, b1, b2 = best_params

    t_Yts_hat = np.vstack([nnet_eval(tX, mode='presigmoid').asnumpy() for tX in Xts])
    t_Yts     = Y[idx_ts]
    test_auc.append(100*calc_auc(t_Yts_hat, t_Yts))
    
    print "\nFold %d test AUC: %.2f, fold running time = %.1fs" % (test_fold, test_auc[-1], toc("fold time"))
    print "*******************************************************************"
    perfs.append(temp_perfs)


print "Average Test AUC: %.2f, Total running time = %.1fs" % (np.mean(test_auc), toc("total time"))

