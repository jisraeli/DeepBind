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
from ..node import node
from .. import globals
from .. import _ext
from smat import *

class full(node):
    """
    A fully-connected affine function Z = dot(X,W).
    Also outputs a cost value cost = decay*sum(abs(W))
    ishape is used to visualize weights as if entering this layer (e.g. set to (28,28) for layer connected to MNIST inputs)
    oshape is used to visualize weights as if exiting this layer (e.g. set to (28,28) for layer connected to auto-encoder output on MNIST)
    """
    def __init__(self, num_units, weight_decay=None, init_scale=None, init_bias=None, ishape=None, oshape=None):
        super(full,self).__init__(["X","W"],["Z","cost"])
        self.size  = num_units
        self.init  = init_scale   if init_scale  is not None else 0.001
        self.init_mu = init_bias  if init_bias   is not None else 0.0
        self.decay = weight_decay
        self.ishape = ishape
        self.oshape = oshape
        self.zero_cost = None
        self.W.inst_axis = 0  # the ROWS of W grow when there are multiple instances

    def getfilters(self):
        if not self.W.shape:
            return None
        if self.ishape:
            F = self.W.fpval.asnumpy().T
            F = np.require(F,requirements=['C'])
            F = F.reshape((self.size,) + self.ishape)
            return F
        if self.oshape:
            F = self.W.fpval.asnumpy()
            F = np.require(F,requirements=['C'])
            F = F.reshape((self.W.shape[0],) + self.oshape)
            return F


    def _slice_inst(self,i):
        if self.W.shape:
            self.W.fpval = self.W.fpval[i*self.W.shape[0]:(i+1)*self.W.shape[0],:].copy()

    def _fprop(self,X,W):
        if X is None:
            return (None,0)
        Z = _ext.blockwise_dot(X, W, self.ninst)
        cost = self._fprop_cost(W)
        return (Z,cost)

    def _bprop(self, X, W, dZ):
        dX = _ext.blockwise_dot_nt(dZ, W, self.ninst)    if (self.X.has_upstream() or ("want_bprop_inputs" in globals.flags)) else None
        dW = _ext.blockwise_dot_tn(X, dZ, self.ninst, W) if (self.W.has_upstream() or ("want_bprop_inputs" in globals.flags)) else None
        self._bprop_cost(W, dW)
        return (dX,dW)

    def _fprop_cost(self,W):
        # Now compute 'cost', only used when evaluating cost function.
        # If we're in the middle of a gradient computation, then
        # we don't need cost to be forward propagated. However, if
        # we're doing a feed-forward test mode computation, then 
        # we do need a cost to be fprop'd
        if isinstance(self.decay,np.ndarray):
            self.decay = asarray(self.decay,dtype=W.dtype) if self.decay.size > 1 else np.asscalar(self.decay)
        if self.zero_cost is None:
            self.zero_cost = zeros((1,self.ninst))
        cost = self.zero_cost
        if (self.decay is not None) and ("bprop_mode" not in globals.flags) and (globals.flags.get("weight_decay_start",0) <= globals.flags.get("step",0)):
            if self.ninst == 1:
                cost = sum(abs(W)) * self.decay
            else:
                C = W.reshape((self.ninst,-1))  # Put each separate weight matrix into its own row.
                C = sum(abs(C),axis=1)          # Sum the absolute values across each row.
                cost = C.T*self.decay           # Turn into row vector of costs, weighted by decay coefficient.
        return cost

    def _bprop_cost(self, W, dW):
        # Backprop weight decay to dW, if any
        if (self.decay is not None) and (globals.flags.get("weight_decay_start",0) <= globals.flags.get("step",0)):
            if self.ninst == 1:
                if dW is not None:
                    dW += float(self.decay)*sign(W)
            else:
                # Add a separate decay for each instance
                _ext.madd_bcast(sign(W),self.decay,W.size/self.ninst,dW)

    def _calc_shapes(self,X,W,Z):
        # First make sure (X.ncol) = (W.nrow)
        if   X._shape and     W._shape: assert X._shape[1] == W._shape[0]
        elif X._shape and not W._shape: W._shape = (X._shape[1],self.size)  # W is (num inputs x num outputs)
        elif W._shape and not X._shape: X._shape = (None,W._shape[0]) 

        # Output dimension is determined by 'size' of this node (number of hidden units)
        Z._shape = (None,self.size)


class combine(full):
    """
    Effectively splices the input matrices X0...Xk and then implements 
    a fully-connected layer between those stacked matrices.
    However, the matrices are never explicitly stacked, and instead the
    matrix multiple is broken down into blocks, so all operations are
    in-place without the extra copying/temporary memory.
    """
    def __init__(self, num_inputs, size, decay=None, init=None, init_mu=None, start_training=None, ishape=None, oshape=None):
        # Create input attributes X0..Xk for k=num_sources-1
        super(full,self).__init__(["X%d"%i for i in range(num_inputs)] + ["W"],["Z","cost"])
        self.num_inputs = num_inputs
        self.size  = size
        self.init  = init      if init    is not None else 0.001
        self.init_mu = init_mu if init_mu is not None else 0.0
        self.decay = decay
        self.start_training = start_training
        self.ishape = ishape
        self.oshape = oshape
        self.zero_cost = None
        self.Xsizes = None
        self.W.inst_axis = 0  # the ROWS of W grow when there are multiple instances

    def _fprop(self, W):
        X = [p.fpval    for p in self.iplugs[:self.num_inputs]]
        Xbroadcast = [self.ninst > 1 and (X[t].shape[1] == self.iplugs[t].shape[1] if X[t] is not None else True) for t in range(len(X))]
        Z = _ext.blockwise_dot_combined(X, W, self.ninst, Xbroadcast)
        
        cost = self._fprop_cost(W)
        return (Z,cost)

    def _bprop(self, W, dZ):
        # TODO: avoid backprop to a specific Xi plug if it has no trainable nodes connected upstream 
        X = [p.fpval    for p in self.iplugs[:self.num_inputs]]
        Xbroadcast = [self.ninst > 1 and (X[t].shape[1] == self.iplugs[t].shape[1] if X[t] is not None else True) for t in range(len(X))]

        # If we aren't allowed to start training a particular component, then kill its gradient for the time being.
        dWmask = None
        if self.start_training:
            assert len(X) == len(self.start_training)
            dWmask = [start < globals.flags.get("step",0) for start in self.start_training]

        dX = _ext.blockwise_dot_nt_combined(X, dZ, W, self.ninst, Xbroadcast)
        dW = _ext.blockwise_dot_tn_combined(X, dZ, W, self.ninst, Xbroadcast, dWmask)

        self._bprop_cost(W, dW)

        return tuple(dX) + (dW,)

    def _calc_shapes(self,W,Z):
        # First make sure sum([X.ncol for X in iplugs]) = (W.nrow)
        Xsize_total = 0
        for X in self.iplugs[:self.num_inputs]:
            assert X.srcs, "%s has no input plugs; cannot determine shape" % X.name
            X.srcs[0]._calc_shape([self]) # Calculate upstream shapes without recursively visiting ourselves by mistake
            X._shape = X.srcs[0]._shape
            #assert X._shape, "%s's shape was not defined; combine layer cannot backward-propagate shape to its inputs." % X.name
            if not X._shape:
                X._shape = (None,0)   # Treat it as an empty matrix
            Xsize_total += X._shape[1]

        if W._shape: assert Xsize_total == W._shape[0]
        else: W._shape = (Xsize_total,self.size)  # W is (num inputs x num outputs)

        # Output dimension is determined by 'size' of this node (number of hidden units)
        Z._shape = (None,self.size)
