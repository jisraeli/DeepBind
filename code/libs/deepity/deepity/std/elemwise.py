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
from .. import globals
from ..node import node
import smat as sm
import numpy as np
from .. import _ext

class elemwise(node):
    def __init__(self,iplugs,oplugs):
        super(elemwise,self).__init__(iplugs,oplugs)

    def _calc_shapes(self,X,Z):
        if   X._shape: Z._shape = X._shape  # All elemwise functions have same output dim as input dim
        elif Z._shape: X._shape = Z._shape


class linear(elemwise):
    """Linear node. Computes Z = X"""
    def __init__(self):     super(linear,self).__init__(["X"],["Z"])
    def _fprop(self,X):     return X
    def _bprop(self,dZ):    return dZ


class exp(elemwise):
    """Exponential node. Computes Z = exp(X)"""
    def __init__(self):   super(exp,self).__init__(["X"],["Z"])
    def _fprop(self,X):     return sm.exp(X) if X  is not None else None
    def _bprop(self,dZ,Z):  return dZ*Z      if dZ is not None else None    # Z = exp(X)


class sqr(elemwise):
    """Square node. Computes Z = X**2"""
    def __init__(self):   super(sqr,self).__init__(["X"],["Z"])
    def _fprop(self,X):     return sm.square(X) if X  is not None else None
    def _bprop(self,dZ,X):  return 2*dZ*X       if dZ is not None else None


class relu(elemwise):
    """Rectified-linear node. Computes Z = max(0,X)"""
    def __init__(self):     super(relu,self).__init__(["X"],["Z"])
    def _fprop(self,X):
        if "disable_relu" in globals.flags:
            return X
        return sm.maximum(0,X) if X  is not None else None
    def _bprop(self,dZ,Z):  return dZ*sm.sign(Z)   if dZ is not None else None   # sign(Z) will be 0 or +1, never -1


class rectify(relu):
    def __init__(self):   super(rectify,self).__init__()


class wrelu(elemwise):
    """Weakly rectified-linear node. Computes Z = max(slope*X,X) for some 0<slope<1"""
    def __init__(self,slope=0.1):
        super(wrelu,self).__init__(["X"],["Z"])
        self.slope = slope
    
    def _fprop(self,X):
        return sm.maximum(self.slope*X,X) if X is not None else None
    
    def _bprop(self,dZ,Z,X):
        if dZ is None:
            return None
        S = sm.sign(X)
        S = sm.maximum(-self.slope,S)  # where X<0 dZ*slope, otherwise dZ*1
        return dZ*abs(S)


class tanh(elemwise):
    """Tanh node. Computes Z = tanh(X)"""
    def __init__(self):     super(tanh,self).__init__(["X"],["Z"])
    def _fprop(self,X):     return sm.tanh(X)  if X  is not None else None
    def _bprop(self,dZ,Z):  return dZ*(1-Z**2) if dZ is not None else None      # tanh'(x) = 1-tanh(x)^2


class logistic(elemwise):
    """Logistic sigmoid node. Computes Z = logistic(X)."""
    def __init__(self):     super(logistic,self).__init__(["X"],["Z"])
    def _fprop(self,X):     return sm.logistic(X) if X  is not None else None
    def _bprop(self,dZ,Z):  return dZ*(Z-Z**2)    if dZ is not None else None     # logisitic'(x) = logisitic(x)-logisitic(x)^2


class dropout(elemwise):
    """
    Dropout node.
    If the global "train_mode" flag is set, computes Z = X*M where M is a bernoulli mask with p=(1-rate).
    Otherwise, computes Z = X*(1-rate).
    """
    def __init__(self, rate=0.5, activerange=None):
         super(dropout,self).__init__(["X"],["Z"])
         self.M = None
         self.rate = rate
         self.activerange = activerange
    
    def _fprop(self,X):
        if X is None:
            return None
        if np.isscalar(self.rate):
            if self.rate == 0:
                return X
            self.rate = sm.asarray([self.rate for i in range(self.ninst)], dtype=X.dtype)
        elif not isinstance(self.rate, sm.sarray):
            self.rate = sm.asarray(self.rate,dtype=X.dtype)

        if "train_mode" in globals.flags:
            Z,self.M = _ext.dropout_fp_train(X, self.rate, "reverse_complement" in globals.flags)
        else:
            Z = _ext.dropout_fp_test(X, self.rate)
        return Z

    def _bprop(self,dZ):
        if np.isscalar(self.rate) and self.rate == 0:
            return dZ
        if "train_mode" in globals.flags:
            dX = _ext.dropout_bp_tr(dZ,self.M)
            self.M = None
        else:
            dX = _ext.dropout_bp_te(dZ,self.rate)
        return dX

    def _slice_inst(self,i):
        assert self.rate.shape[0] % self.ninst == 0
        chunksize = self.rate.shape[0] // self.ninst
        self.rate = self.rate[i*chunksize:(i+1)*chunksize].copy()

################ trainable nodes #################


class bias(elemwise):
    """Bias node. Computes Z = X + b"""
    def __init__(self, init=0.0, init_mu=0, negdecay=None, viz=True, start_training=0):
        super(bias ,self).__init__(["X","b"], ["Z"])
        self.init = init
        self.init_mu = init_mu
        self.viz = viz
        self.negdecay = negdecay
        self.start_training = start_training
    def _fprop(self,X,b):   return X + b  if X is not None else None                     # broadcast row-vector b
    def _bprop(self,dZ):
        if dZ is None:
           return (None,0)
        db = sm.sum(dZ,axis=0)
        if self.negdecay:
            db += self.negdecay
        if self.start_training > globals.flags.get("step",0):
            db *= 0
        return (dZ,db)     # (dX,db)
    def _calc_shapes(self,X,Z,b):
        # First make sure X and b have equal number of columns
        if   X._shape and     b._shape: assert X._shape[1] == b._shape[1]
        elif X._shape and not b._shape: b._shape = (1,X._shape[1])
        elif b._shape and not X._shape: X._shape = (None,b._shape[1])

        # Then set the shape of Z to match the shape of X
        elemwise._calc_shapes(self,X,Z)

    def _slice_inst(self,i):
        self.b.fpval = self.b.fpval[0,i*self.b.shape[1]:(i+1)*self.b.shape[1]].copy()
        #print "bias: ", self.b.fpval

    def getfilters(self):
        if self.viz:
            F = self.b.fpval.asnumpy().reshape((-1,1,1))
            if self.X.srcs[0].node.__class__.__name__ == "corr1ord":
                # HACK: since corr1ord node subtracts a constant to ensure
                #       each filter column has mean zero, we need to take
                #       add the total "visualization bias" to our own value,
                #       so that the visualization is still showing a 
                #       correct (equivalent) model
                filter_biases = self.X.srcs[0].node.getfilters(want_bias=True)
                if filter_biases is not None:
                    filter_biases = filter_biases.sum(axis=1)
                    F += filter_biases.reshape((-1,1,1))
            return F


class scale(elemwise):
    """Scale node. Computes Z = X*w"""
    def __init__(self, init=0.0, init_mu=1.0, viz=False):
        super(scale, self).__init__(["X","w"], ["Z"])
        self.init = init
        self.init_mu = init_mu
        self.viz = viz
    def _fprop(self,X,w):      return X*w   if X is not None else None                      # broadcast row-vector w
    def _bprop(self,X,w,Z,dZ): return (dZ*w,sm.sum(dZ*X,axis=0)) # (dX,dw)
    def _calc_shapes(self,X,Z,w):
        # First make sure X and w have equal number of columns
        if   X._shape and     w._shape: assert X._shape[1] == w._shape[1]
        elif X._shape and not w._shape: w._shape = (1,X._shape[1])
        elif w._shape and not X._shape: X._shape = (None,w._shape[1])

        # Then set the shape of Z to match the shape of X
        elemwise._calc_shapes(self,X,Z)

    def _slice_inst(self,i):
        self.w.fpval = self.w.fpval[0,i*self.w.shape[1]:(i+1)*self.w.shape[1]].copy()
        #print "scale: ", self.w.fpval

    def getfilters(self):
        if self.viz:
            F = self.w.fpval.asnumpy().reshape((-1,1,1))
            return F
