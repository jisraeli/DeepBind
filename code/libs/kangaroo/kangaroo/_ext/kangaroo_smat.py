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
from smat import *
from smat import smat_dll
from os.path import abspath,join,dirname
from ctypes import *
import os

###################################################################
# Declare some useful ctypes based on the C++ types

c_padmode_t = c_int
c_pm_zero_output = 0


c_pooltype_t = c_int
c_pt_max = 0
c_pt_avg = 1
c_pt_sum = 2
c_pt_all = 3
c_isize_t = smat_dll.c_isize_t
c_usize_t = smat_dll.c_usize_t
c_smat_p  = smat_dll.c_smat_p

class c_corr1ord_options_t(Structure):
    _fields_ = [("padmode", c_int),
                ("nchannel", c_usize_t)]
c_corr1ord_options_p = POINTER(c_corr1ord_options_t)

class c_poolrgn_options_t(Structure):
    _fields_ = [("ptype", c_pooltype_t)]
c_poolrgn_options_p = POINTER(c_poolrgn_options_t)

###################################################################
# Now create the public 'dll' object exposed to smat.py, with all the methods
# exported by the DLL available for calling
#
_ext_dll = None
def ext_dll():
    global _ext_dll
    if _ext_dll is None:
        _ext_dll = load_extension("kangaroo_smat")
        _ext_dll.api_corr1ord.declare(       None, [c_smat_p,c_smat_p,c_smat_p,c_corr1ord_options_p])
        _ext_dll.api_corr1ord_bprop_W.declare(None,[c_smat_p,c_smat_p,c_smat_p,c_corr1ord_options_p])
        _ext_dll.api_corr1ord_bprop_X.declare(None,[c_smat_p,c_smat_p,c_smat_p,c_corr1ord_options_p])
        _ext_dll.api_poolrgn.declare(        None, [c_smat_p,c_smat_p,c_smat_p,c_smat_p,c_poolrgn_options_p])
        _ext_dll.api_poolrgn_bprop.declare(  None, [c_smat_p,c_smat_p,c_smat_p,c_smat_p,c_poolrgn_options_p])
        _ext_dll.api_dropoutord_fp_tr.declare(None, [c_smat_p,c_smat_p,c_smat_p,c_float])
        _ext_dll.api_autotune.declare( None, [])
    return _ext_dll

str2pooltype = { "max" : c_pt_max, "avg" : c_pt_avg, "sum" : c_pt_sum, "all" : c_pt_all }

def corr1ord(W,X,Z,nchannel):
    """
    Cross-correlates a set of 1D filters W with a vector of ordinals X. 
    X is a 1xN uint8 matrix, where X[i] is range {0,...,nchannel-1} or the value 255 ("all ordinals").
    W is a (M*nchannel)xP float matrix, where P = num filters, M = filter length.
    Return value Z is an NxP matrix (P feature maps, each of length N).
    NOTE: currently the final M-1 positions of Z will be left uninitialized, 
          so it is up to you to pad the vector with M-1 copies of value 255, if that is
          the desired padding behaviour.
    """
    options = c_corr1ord_options_t()
    options.padmode = c_pm_zero_output
    options.nchannel = nchannel
    ext_dll().api_corr1ord(W._ptr,X._ptr,Z._ptr,byref(options))

def corr1ord_bprop_W(dW,X,dZ,nchannel):
    """
    Backpropagates a matrix of feature map deltas dZ into filter deltas dW.
    All the quantities are of the same dimensions as described in corr1ord.
    """
    options = c_corr1ord_options_t()
    options.padmode = c_pm_zero_output
    options.nchannel = nchannel
    ext_dll().api_corr1ord_bprop_W(dW._ptr,X._ptr,dZ._ptr,byref(options))

def corr1ord_bprop_X(dX,W,dZ,nchannel):
    """
    Backpropagates a matrix of feature map deltas dZ into input sequence deltas dX.
    All the quantities are of the same dimensions as described in corr1ord.
    """
    options = c_corr1ord_options_t()
    options.padmode = c_pm_zero_output
    options.nchannel = nchannel
    ext_dll().api_corr1ord_bprop_X(dX._ptr,W._ptr,dZ._ptr,byref(options))

def poolrgn(unpooledmaps,regions,ptype,want_argmax=False):
    noutputs = 2 if ptype == "all" else 1
    options = c_poolrgn_options_t()
    options.ptype = str2pooltype[ptype]
    nregion = regions.shape[0]
    nfeaturemap = unpooledmaps.shape[1]
    pooledmaps        = empty((nregion,nfeaturemap*noutputs),dtype=unpooledmaps.dtype)
    pooledmaps_argmax = empty((nregion,nfeaturemap),dtype=uindex) if (ptype in ("max","all")  and want_argmax) else None
    ext_dll().api_poolrgn(unpooledmaps._ptr,regions._ptr,pooledmaps._ptr,pooledmaps_argmax._ptr if (pooledmaps_argmax is not None) else None,byref(options))
    if want_argmax:
        return pooledmaps,pooledmaps_argmax
    return pooledmaps,None

def poolrgn_bprop(unpooledmaps,regions,pooledmaps,pooledmaps_argmax,ptype):
    options = c_poolrgn_options_t()
    options.ptype = str2pooltype[ptype]
    pooledmaps_argmax_ptr = pooledmaps_argmax._ptr if (pooledmaps_argmax is not None) and (ptype in ("max","all")) else None
    ext_dll().api_poolrgn_bprop(unpooledmaps._ptr,regions._ptr,pooledmaps._ptr,pooledmaps_argmax_ptr,byref(options))

def dropoutord_fp_train(X,rate):
    # If matchrows=True, then every pair of rows will have the same mask.
    # Used for dropout with reverse complement enabled.
    Z = empty_like(X)
    M = empty(X.shape,dtype=bool)
    ext_dll().api_dropoutord_fp_tr(X._ptr, Z._ptr, M._ptr, rate)
    return Z,M


def autotune():
    ext_dll().api_autotune()
