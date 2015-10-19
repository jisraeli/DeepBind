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

class dropoutord(deepity.std.elemwise):
    """
    Dropout on ordinal sequence input.
    """
    def __init__(self, rate):
        super(dropoutord,self).__init__(["X","regions"],["Z"])
        self.rate = rate
        self.M = None

    def _fprop(self,X,regions):
        if X is None:
            return None

        if "train_mode" in deepity.globals.flags:
            Z,self.M = kangaroo_smat.dropoutord_fp_train(X, self.rate)
            if "reverse_complement" in deepity.globals.flags:
                # TODO: should be done on GPU
                _M = self.M.asnumpy()
                _R = regions.asnumpy()
                padsize = _M.size - _R[-1,-1]
                for i in range(0,len(_R),2):
                    a,b = _R[i]
                    c,d = _R[i+1]
                    _M[c+padsize:d] = np.flipud(_M[a+padsize:b])
                self.M = asarray(_M)
                _Z = X.asnumpy()
                _Z[~_M] = 254
                Z = asarray(_Z)
                pass
        else:
            Z = X
        return Z

    def _bprop(self,dZ):
        self.M = None
        return dZ
