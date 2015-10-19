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
import smat as sm
import numpy as np
from . import kangaroo_smat

class maxpool(deepity.node):
    """
    Max pooling node.
        X = input values  (N x M)
        R = input regions (P x 2)
        Z = output values  (P x M) where Z[i,j] = max(X[R[i,0]:R[i,1],j],axis=0)
    where 
        M = number of feature maps,
        N = number of input positions, 
        P = number of output positions
    """
    def __init__(self, regions=None):
        super(maxpool,self).__init__(["X","R"],["Z"])
        self.I = None
        self.regions = regions or [(0,None)]

    def _fprop(self,X,R):
        if deepity.globals.flags.get("collect_featuremaps",False):
            old = deepity.globals.flags.pop("collect_featuremaps")
            fmaps = old if isinstance(old, list) else [] 
            fmaps += self._collect_featuremaps(X,R)
            deepity.globals.flags.push("collect_featuremaps",fmaps)

        Z,self.I = kangaroo_smat.poolrgn(X,R,ptype="max",
                                         want_argmax = "bprop_mode" in deepity.globals.flags)

        
        if "collect_argmax" in deepity.globals.flags:
            deepity.globals.flags.pop("collect_argmax")
            deepity.globals.flags.push("collect_argmax", self.I.asnumpy())
        elif "force_argmax" in deepity.globals.flags:
            _I = deepity.globals.flags.get("force_argmax")
            _X = X.asnumpy()
            _Z = _X.ravel()[_I.ravel()].reshape(Z.shape)
            Z = sm.asarray(_Z)

        return Z

    def _bprop(self,X,R,dZ):
        dX = sm.zeros_like(X)
        kangaroo_smat.poolrgn_bprop(dX,R,dZ,self.I,ptype="max")
        self.I = None
        return (dX,None)

    def _calc_shapes(self,X,R,Z):
        # X.shape[1] must match Z.shape[1]
        if   X._shape and     Z._shape: assert X._shape[1] == Z._shape[1]
        if   X._shape and not Z._shape: Z._shape = (None,X._shape[1])
        if   Z._shape and not X._shape: X._shape = (None,Z._shape[1])

    def _requirements(self):
        return { "sequence_pooling" : self.regions }

    def _collect_featuremaps(self,X,R):
        X = X.asnumpy()
        R = R.asnumpy()
        batch_fmaps = []
        for i in range(len(R)):
            fmaps = X[R[i,0]:R[i,1],:].T.copy()
            batch_fmaps.append(fmaps)
        return batch_fmaps


class avgpool(deepity.node):
    """
    Average pooling node.
        X = input values  (N x M)
        R = input regions (P x 2)
        Z = output values  (P x M) where Z[i,j] = mean(X[R[i,0]:R[i,1],j],axis=0)
    where 
        M = number of feature maps,
        N = number of input positions, 
        P = number of output positions
    """
    def __init__(self, regions=None):
        super(avgpool,self).__init__(["X","R"],["Z"])
        self.regions = regions or [(0,None)]

    def _fprop(self,X,R):
        Z = kangaroo_smat.poolrgn(X,R,ptype="avg",want_argmax=False)
        return Z

    def _bprop(self,X,R,dZ):
        dX = sm.zeros_like(X)
        kangaroo_smat.poolrgn_bprop(dX,R,dZ,None,ptype="avg")
        return (dX,None)

    def _calc_shapes(self,X,R,Z):
        # X.shape[1] must match Z.shape[1]
        if   X._shape and     Z._shape: assert X._shape[1] == Z._shape[1]
        if   X._shape and not Z._shape: Z._shape = (None,X._shape[1])
        if   Z._shape and not X._shape: X._shape = (None,Z._shape[1])

    def _requirements(self):
        return { "sequence_pooling" : self.regions }

class allpool(deepity.node):
    """
    Max *and* average pooling node.
        X = input values  (N x M)
        R = input regions (P x 2)
        Z = output values  (P x 2M) where Z[i,2*j+0] = max( X[R[i,0]:R[i,1],j],axis=0)
                                    and   Z[i,2*j+1] = mean(X[R[i,0]:R[i,1],j],axis=0)
    where 
        M = number of feature maps,
        N = number of input positions, 
        P = number of output positions
    """
    def __init__(self, regions=None):
        super(allpool,self).__init__(["X","R"],["Z"])
        self.I = None
        self.regions = regions or [(0,None)]

    def _fprop(self,X,R):
        if deepity.globals.flags.get("collect_featuremaps",False):
            old = deepity.globals.flags.pop("collect_featuremaps")
            fmaps = old if isinstance(old, list) else [] 
            fmaps += self._collect_featuremaps(X,R)
            deepity.globals.flags.push("collect_featuremaps",fmaps)

        Z,self.I = kangaroo_smat.poolrgn(X,R,ptype="all",
                                         want_argmax = "bprop_mode" in deepity.globals.flags)
        
        if "collect_argmax" in deepity.globals.flags:
            deepity.globals.flags.pop("collect_argmax")
            deepity.globals.flags.push("collect_argmax", self.I.asnumpy())
        elif "force_argmax" in deepity.globals.flags:
            _I = deepity.globals.flags.get("force_argmax")
            _X = X.asnumpy()
            _Zmax = _X.ravel()[_I.ravel()].reshape(_I.shape)
            _Z = Z.asnumpy()
            _Z[:,np.arange(0,Z.shape[1],2)] = _Zmax
            Z = sm.asarray(_Z)

        return Z

    def _bprop(self,X,R,dZ):
        dX = sm.zeros_like(X)
        kangaroo_smat.poolrgn_bprop(dX,R,dZ,self.I,ptype="all")
        self.I = None
        return (dX,None)

    def _calc_shapes(self,X,R,Z):
        # Z.shape[1] must be 2*X.shape[1]
        if   X._shape and     Z._shape: assert 2*X._shape[1] == Z._shape[1]
        if   X._shape and not Z._shape: Z._shape = (None,2*X._shape[1])
        if   Z._shape and not X._shape: X._shape = (None,Z._shape[1]/2)

    def _requirements(self):
        return { "sequence_pooling" : self.regions }

    def _collect_featuremaps(self,X,R):
        X = X.asnumpy()
        R = R.asnumpy()
        batch_fmaps = []
        for i in range(len(R)):
            fmaps = X[R[i,0]:R[i,1],:].T.copy()
            batch_fmaps.append(fmaps)
        return batch_fmaps
