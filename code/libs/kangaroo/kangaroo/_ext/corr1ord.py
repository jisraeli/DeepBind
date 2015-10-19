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
import deepity
import deepity._ext
import numpy as np
from smat import *
from . import kangaroo_smat
from . import dropoutord

class corr1ord(deepity.node):
    """
    A 1D correlation on an ordinal sequence.
    """
    def __init__(self, nfilter, fsize=None, decay=0, init=1e-3, nchannel=4, name=None, constraint=None, start_training=0):
        super(corr1ord,self).__init__(["X","W"],["Z","cost"],name)
        self.nfilter = nfilter
        self.fsize   = fsize
        self.decay   = decay
        self.init    = init
        self.nchannel = nchannel
        self.constraint = constraint
        self.start_training = start_training


    def _slice_inst(self,i):
        self.W.fpval = self.W.fpval[:,i*self.nfilter:(i+1)*self.nfilter]

    def enforce_constraints(self, W):
        if self.constraint is None:
            return
        if self.constraint in ("pfm", "psam"):
            F = exp(W)  # Each column will be 4 elements of a filter column
            for i in range(self.fsize):
                Fi = F[i*4:(i+1)*4,:]
                if self.constraint == "pfm":
                    Fi[:] = Fi / sum(Fi,axis=0)          # Normalize
                else:
                    Fi[:] = Fi / max(Fi,axis=0)          # Normalize
            W[:] = log(F)
        elif type(self.constraint) == tuple:
            lo,hi = self.constraint
            W[:] = minimum(W,hi)
            W[:] = maximum(W,lo)
        else:
            raise NotImplementedError("constraint \"%s\" not implemented" % str(self.constraint))

    def getfilters(self, want_bias=False):
        F = self.W.fpval.asnumpy().T
        F = np.require(F,requirements=['C'])
        F  = F.reshape((self.nfilter,self.fsize,self.nchannel))
        if isinstance(self.constraint,str):
            F = exp(F)
            bias = None
        else:
            bias = F.mean(axis=2).reshape((self.nfilter,self.fsize,1)) # shift bias from filter positions to the actual bias, so that mean is always zero at each filter position (100% equivalent model, just a reparameterization)
            if want_bias:
                return bias
            F -= bias
        F2 = np.empty((self.nfilter,self.nchannel,self.fsize),F.dtype)
        for i in range(self.nfilter):
            F2[i,:,:] = F[i,:,:].T
        return F2


    def _fprop(self,X,W):
        if isinstance(self.decay,np.ndarray):
            self.decay = asarray(self.decay,dtype=W.dtype) if self.decay.size > 1 else np.asscalar(self.decay)

        Z = zeros((X.shape[0],W.shape[1]))
        kangaroo_smat.corr1ord(W,X,Z,self.nchannel)


        if "train_mode" not in deepity.globals.flags:
            upstream = self.X.srcs[0].node
            if type(upstream) == dropoutord.dropoutord:
                if upstream.rate > 0:
                    Z *= (1-upstream.rate)

        cost = 0
        if (self.decay is not None) and ("bprop_mode" not in deepity.globals.flags) and (deepity.globals.flags.get("weight_decay_start",0) <= deepity.globals.flags.get("step",0)):
            C = sum(abs(W),axis=0)          # Sum the absolute values across each row.
            C = C.reshape((self.ninst,-1))  # Put each separate weight matrix into its own row.
            cost = sum(C,axis=1).T*self.decay           # Turn into row vector of costs, weighted by decay coefficient.

        return (Z,cost)


    def _bprop(self,X,W,dZ):
        dW = zeros_like(W)
        kangaroo_smat.corr1ord_bprop_W(dW,X,dZ,self.nchannel) # backprop to filters

        # Add a separate decay for each instance
        if (self.decay is not None) and (deepity.globals.flags.get("weight_decay_start",0) <= deepity.globals.flags.get("step",0)):
            deepity._ext.madd_bcast(sign(W),self.decay,W.shape[0]/self.ninst,dW)

        #assert not self.X.has_upstream()  # backprop to dX not implemented

        dX = None
        if ("want_bprop_inputs" in deepity.globals.flags):
            dX = zeros((X.size,4), W.dtype)
            kangaroo_smat.corr1ord_bprop_X(dX,W,dZ,self.nchannel) # backprop to filters

        if self.start_training > deepity.globals.flags.get("step",0):
            dW *= 0

        return (dX,dW)


    def _requirements(self):
        assert self.fsize is not None, "Must set corr1ord.fsize attribute before any operation that requires shape"
        return { "sequence_padding" : self.fsize - 1 }


    def _calc_shapes(self,X,W,Z):
        assert self.fsize is not None, "Must set corr1ord.fsize attribute before any operation that requires shape"

        # X must be a single 1D sequence of variable length
        if not X._shape: X._shape = (None,1)

        # W size is determined entirely by internal parameters
        if not W._shape: W._shape = (self.fsize*self.nchannel,self.nfilter)

        # Output dimension is determined by 'nfilter' of this node (number of feature maps)
        Z._shape = (None,self.nfilter)



class motif_scan(corr1ord):
    def __init__(self, num_motifs, motif_len=None, weight_decay=0, init_scale=1e-3):
        super(motif_scan,self).__init__(nfilter=num_motifs,fsize=num_motifs,decay=weight_decay,init=init_scale)
