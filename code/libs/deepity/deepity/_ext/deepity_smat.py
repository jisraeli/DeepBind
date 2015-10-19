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
from smat import *
from smat import smat_dll
from os.path import abspath,join,dirname
from ctypes import *

###################################################################
# Declare some useful ctypes based on the C++ types

c_isize_t = smat_dll.c_isize_t
c_usize_t = smat_dll.c_usize_t
c_smat_p  = smat_dll.c_smat_p

###################################################################
# Now create the public 'dll' object exposed to smat.py, with all the methods
# exported by the DLL available for calling
#
_ext_dll = None
def ext_dll():
    global _ext_dll
    if _ext_dll is None:
        _ext_dll = load_extension("deepity_smat")
        _ext_dll.api_gradstep.declare(          None, [c_smat_p,c_smat_p,c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_gradstep_nesterov1.declare(None, [c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_gradstep_nesterov2.declare(None, [c_smat_p,c_smat_p,c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_madd_bcast.declare(        None, [c_smat_p,c_smat_p,c_usize_t,c_smat_p])
        _ext_dll.api_maskout.declare(           None, [c_smat_p,c_smat_p])
        _ext_dll.api_calc_zmask.declare(        None, [c_smat_p,c_smat_p])
        _ext_dll.api_dropout_fp_tr.declare( None, [c_smat_p,c_smat_p,c_smat_p,c_smat_p,c_int])
        _ext_dll.api_dropout_fp_te.declare( None, [c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_dropout_bp_tr.declare( None, [c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_dropout_bp_te.declare( None, [c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_blockwise_dot.declare( None, [c_int,c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_blockwise_dot_nt.declare( None, [c_int,c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_blockwise_dot_tn.declare( None, [c_int,c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_blockwise_dot_combined.declare( None, [c_int,c_int,c_int,c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_blockwise_dot_nt_combined.declare( None, [c_int,c_int,c_int,c_smat_p,c_smat_p,c_smat_p])
        _ext_dll.api_blockwise_dot_tn_combined.declare( None, [c_int,c_int,c_int,c_smat_p,c_smat_p,c_smat_p])
    return _ext_dll

#######################################################################

def gradstep(P,dP,drate,mP,mrate,grad,nesterov=False):
    """
    Performs a gradient update step on parameters P,
    using gradient dP with learning rate (drate), and
    momentum vector mP with momentum rate (mrate).

    grad() must be a function that computes:
       dP[:] = gradient at current P
    where 'grad' is assumed to have references to 
    P and to dP.

    If nesterov is False, the computation is:
       grad()
       mP[:] = drate*dP + mrate*mP
       P[:]  = P + mP

    If nesterov is True,  the computation is:
       P[:]  = P + mrate*mP
       grad()
       mP[:] = drate*dP + mrate*mP
       P[:]  = P + drate*dP
    """
    assert callable(grad)
    if nesterov:
        # P[:] += mrate*mP
        ext_dll().api_gradstep_nesterov1(P._ptr,mP._ptr,mrate._ptr)

        # dP[:] = gradient at P + mrate*mP
        grad()

        # mP[:] = drate*dP + mrate*mP
        # P[:] += drate*dP
        ext_dll().api_gradstep_nesterov2(P._ptr,dP._ptr,drate._ptr,mP._ptr,mrate._ptr)
    else:
        # dP[:] = gradient at P
        grad()

        # mP[:] = drate*dP + mrate*mP
        #  P[:] = P + mP
        ext_dll().api_gradstep(P._ptr,dP._ptr,drate._ptr,mP._ptr,mrate._ptr)
        return

#######################################################################

def madd_bcast(A,b,k,dst):
    """
    Equivalent to dst[i] += A[i] * b[(i/k) % b.size]
    where dst, A and b are all treated as 1D vectors.
    """
    if np.isscalar(b):
        dst += A*b
    else:
        if isinstance(b,np.ndarray):
            b = asarray(b,dtype=A.dtype)
        ext_dll().api_madd_bcast(A._ptr,b._ptr,k,dst._ptr)

#######################################################################

def maskout(M,A):
    """
    Equivalent to A[i] = M[i] ? A[i] : 0 where M is of dtype bool.
    Notice that this replaces NaN with zero, unlike A *= M
    """
    ext_dll().api_maskout(M._ptr,A._ptr)

#######################################################################

def dropout_fp_train(X,rate,matchrows):
    # If matchrows=True, then every pair of rows will have the same mask.
    # Used for dropout with reverse complement enabled.
    Z = empty_like(X)
    M = empty(X.shape,dtype=bool)
    ext_dll().api_dropout_fp_tr(X._ptr,rate._ptr,Z._ptr,M._ptr,matchrows)
    return Z,M

def dropout_fp_test(X,rate):
    Z = empty_like(X)
    ext_dll().api_dropout_fp_te(X._ptr,rate._ptr,Z._ptr)
    return Z

def dropout_bp_tr(dZ,M):
    dX = empty_like(dZ)
    ext_dll().api_dropout_bp_tr(dZ._ptr,M._ptr,dX._ptr)
    return dX

def dropout_bp_te(dZ,rate):
    dX = empty_like(dZ)
    ext_dll().api_dropout_bp_te(dZ._ptr,rate._ptr,dX._ptr)
    return dX

#######################################################################

def blockwise_dot(X, W, nblock):
    """
    Computes Z[:,i] = dot(X[:,i],W[i,:]) for each submatrix indexed here by i.
    Special case: if X.shape[1]*nblock == W.shape[0] then 
                  the computation is dot(X,W[i,:]) each time.
    """
    if nblock == 1:
        return dot(X, W)

    Z = empty((X.shape[0],W.shape[1]*nblock), X.dtype)
    ext_dll().api_blockwise_dot(nblock, X._ptr, W._ptr, Z._ptr)
    return Z

def blockwise_dot_nt(dZ, W, nblock):
    """
    Given Computes dX[:,i] = dot_nt(dZ[:,i],W[i,:]) for each submatrix indexed here by i.
    """
    if nblock == 1:
        return dot_nt(dZ, W)

    dX = empty((dZ.shape[0],W.shape[0]), dZ.dtype)
    ext_dll().api_blockwise_dot_nt(nblock, dZ._ptr, W._ptr, dX._ptr)
    return dX

def blockwise_dot_tn(X, dZ, nblock, W):
    """
    Computes dW[i,:] = dot_tn(X[:,i],dZ[:,i]) for each submatrix indexed here by i.
    """
    if nblock == 1:
        return dot_tn(X, dZ)

    dW = empty_like(W)
    ext_dll().api_blockwise_dot_tn(nblock, X._ptr, dZ._ptr, dW._ptr)
    return dW

def blockwise_dot_combined(X, W, nblock, Xbroadcast):
    """
    A version of blockwise_dot that works by combining a LIST of X matrices,
    each containing its own blocks of columns, rather than a single X matrix.
    """
    Z = empty((X[0].shape[0], W.shape[1]*nblock), X[0].dtype)
    Xoffset = 0
    for Xindex in range(len(X)):
        if X[Xindex] is not None:
            ext_dll().api_blockwise_dot_combined(nblock, Xoffset, Xbroadcast[Xindex], X[Xindex]._ptr, W._ptr, Z._ptr)
            Xoffset += X[Xindex].shape[1] // (1 if Xbroadcast[Xindex] else nblock)
    return Z

def blockwise_dot_nt_combined(X, dZ, W, nblock, Xbroadcast):
    """
    A version of blockwise_dot_nt that generates a LIST of output dX matrices
    each containing its own blocks of columns.
    """
    dX = []
    Xoffset = 0
    for Xindex in range(len(X)):
        if X[Xindex] is not None:
            if not Xbroadcast[Xindex]:
                dX.append(zeros_like(X[Xindex]))
                ext_dll().api_blockwise_dot_nt_combined(nblock, Xoffset, Xbroadcast[Xindex], dX[-1]._ptr, W._ptr, dZ._ptr)
            else:
                dX.append(None) # Can't currently backpropagate to a broadcasted input
            Xoffset += X[Xindex].shape[1] // (1 if Xbroadcast[Xindex] else nblock)
        else:
            dX.append(None) # Can't currently backpropagate to a broadcasted input
    return dX

def blockwise_dot_tn_combined(X, dZ, W, nblock, Xbroadcast, dWmask=None):
    """
    A version of blockwise_dot_tn that fills the rows of the return value dW 
    using a LIST of input X matrices, each containing its own blocks of columns.
    """
    dW = empty_like(W)
    if dWmask is not None:
        assert len(dWmask) == len(X)
        dW[:] = 0
    Xoffset = 0
    for Xindex in range(len(X)):
        if X[Xindex] is not None:
            if dWmask is None or dWmask[Xindex]:
                ext_dll().api_blockwise_dot_tn_combined(nblock, Xoffset, Xbroadcast[Xindex], X[Xindex]._ptr, dW._ptr, dZ._ptr)
            Xoffset += X[Xindex].shape[1] // (1 if Xbroadcast[Xindex] else nblock)
    return dW

def calc_Zmask(Z, Zmask):
    ext_dll().api_calc_zmask(Z._ptr, Zmask._ptr)

