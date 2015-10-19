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
from ..node import node
from smat import *
from .. import _ext
from . import globals

class loss_node(node):
    """Calculate a loss function applied to several instances."""
    def __init__(self, ngroup=1):
        super(loss_node,self).__init__(["Z","Y","Ymask"],["loss","Zmask"]) # Zmask is used to kill a certain prediction's contribution to the gradient, e.g. if reverse complement mode is enabled then half the rows will be disabled. It is a computed value, an output.
        self.ngroup = ngroup
        self.Y.trainable = False
        self.Ymask.trainable = False 
        self.batchmean = True

    def _calc_loss(self, Z, Y):
        raise NotImplementedError()
        
    def _fprop(self, Z, Y, Ymask):
        Zmask = self._calc_Zmask(Z)

        L = self._calc_loss(Z,Y)

        if Ymask is not None:
            #ratio = asarray(sum(M,axis=0),dtype=float32) / M.shape[0]
            #print ratio
            _ext.maskout(Ymask,L)

        if Zmask is not None:
            _ext.maskout(Zmask,L)

        L = mean(L,axis=0)     # Mean loss along each column

        if self.ngroup == 1:
            # Output a single overall mean loss associated with each instance.
            # Coming into this if-statement, loss matrix L will be a concatenation
            # of ninst row vectors:
            #  [ z_1, ..., z_k ]
            # where each z_k is a row vector of outputs for model instance k.
            # So, we need to reshape it to be
            #  [ z_1 ;
            #    ...
            #    z_k ]
            # then take the sum along columns to get a column vector, and then
            # transpose that back into a row vector with 'ninst' values.
            assert L.shape[1]//self.ninst == self.Y.shape[1]
            L = L.reshape((-1,self.Y.shape[1]))
            L = self._scale_loss(sum(L,axis=1).reshape((1,-1)))     # Mean of (sum of output component losses) over all cases in minibatch
            return (L, Zmask)
    
        raise NotImplementedError("Not finished")

        # Multiple-output case 1: one column per output 
        m = Z.ncol
        if m == self.ngroup:
            return (self._scale_loss(L), Zmask)           # Just return current row-vector of errors

        # Multiple-output case 2: several columns per output
        L = L.reshape((-1,m/self.ngroup))   # Make enough columns for just this group
        L = sum(L,axis=1)                   # Calculate sum in those columns
        L = L.reshape((1,-1))               # Convert back to row-vector.
        L = self._scale_loss(L)/self.ninst
        return (L, Zmask)

    def _bprop(self, Z, Y, Ymask):
        if self.ngroup == 1:
            self.Zmask._fpval = self._calc_Zmask(Z)
            dZ = self._calc_dZ(Z,Y)
            if Ymask is not None:
                _ext.maskout(Ymask,dZ)
            if self.Zmask._fpval is not None:
                _ext.maskout(self.Zmask._fpval,dZ)
            if self.batchmean:
                dZ *= (1./Z.nrow)
            return dZ
        raise NotImplementedError("Not finished")

    def _calc_dZ(self, Z, Y):
        return Z-Y

    def _calc_Zmask(self, Z):
        Zmask = None
        if "reverse_complement" in globals.flags:
            Zmask = zeros(Z.shape, bool)
            _ext.calc_Zmask(Z, Zmask)
        return Zmask

    def _calc_shapes(self, Z, Y, loss):
        # Make sure Z and Y have same number of columns
        if   Z._shape and     Y._shape: assert Z._shape[1] == Y._shape[1]
        elif Z._shape and not Y._shape: Y._shape = (None,Z._shape[1])
        elif Y._shape and not Z._shape: Z._shape = (None,Y._shape[0]) 

        # Output dimension is always scalar
        loss._shape = (1,1)


class mse(loss_node):
    """Mean squared error."""
    def __init__(self,ngroup=1):
        super(mse,self).__init__(ngroup)
        
    def _calc_loss(self,Z,Y):    return (Z-Y)**2        # Elementwise squared errors
    def _scale_loss(self,loss):
        return 0.5*loss        # Divide by 2


class nll(loss_node):
    """Negative log-likelihood of a softmax or logistic layer."""
    def __init__(self,ngroup=1):
        super(nll,self).__init__(ngroup)
        
    def _calc_loss(self,Z,Y):
        # If only a single output, then treat it as probability of class label 1
        if Y.shape[1] == self.ninst:
            return log(maximum(1e-15,Z))*Y + log(maximum(1e-15,1-Z))*(1-Y) 
        
        # Otherwise, treat it as a multiclass problem
        return log(maximum(1e-15,Z))*Y                  # Elementwise negative log-likelihood, with max(eps,Z) to avoid NaNs
    
    def _scale_loss(self,loss):  return -loss           # Negate
    

class hinge(loss_node):
    """Bidirectional hinge loss. Penalizes case "Z>0,Y=0" by cost Z, and case "Z<1,Y=1" by cost 1-Z."""
    def __init__(self):
        super(hinge,self).__init__(1)
        
    def _calc_loss(self,Z,Y):
        return maximum(0, (1-Y)*Z+Y*(1-Z))

    def _calc_dZ(self, Z, Y):
        L = (1-Y)*Z+Y*(1-Z)
        dZ = 1-2*Y
        _ext.maskout(L>0,dZ)
        return dZ

    def _scale_loss(self,loss):
        return loss
        




