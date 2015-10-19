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
import numpy.random as npr
import smat as sm

def rpad(string,minlen):
    return "%s%s" % (string," "*(minlen-len(string)))

def numpy_softmax(M,axis=None):
    M = np.exp(M - np.max(M,axis=axis,keepdims=True))
    return M/np.sum(M,axis=axis,keepdims=True)

np.rand     = lambda n,m,dtype=None: np.asarray(npr.rand(n,m),dtype=dtype)
np.randn    = lambda n,m,dtype=None: np.asarray(npr.randn(n,m),dtype=dtype)
np.logistic = lambda A: 1/(1+np.exp(-A))
np.dot_tn   = lambda A,B: np.dot(A.T,B)
np.dot_nt   = lambda A,B: np.dot(A,B.T)
np.dot_tt   = lambda A,B: np.dot(A.T,B.T)
np.nnz      = lambda X,axis=None: np.sum(X!=0,axis=axis)  # numpy doesn't have a nnz that operates only along axes, so make a lambda for it.
np.softmax  = numpy_softmax
np.sync     = lambda: None

dtypes_logical = {np.bool}
dtypes_sint    = {np.int8 ,np.int16 ,np.int32 ,np.int64}
dtypes_uint    = {np.uint8,np.uint16,np.uint32,np.uint64}
dtypes_float   = {np.float32,np.float64}
dtypes_signed  =                   dtypes_sint                       | dtypes_float
dtypes_integral= dtypes_logical  | dtypes_sint     | dtypes_uint
dtypes_numeric =                   dtypes_sint     | dtypes_uint     | dtypes_float
dtypes_generic = dtypes_logical  | dtypes_numeric

def supported(dts): 
    return dts.intersection(sm.get_supported_dtypes())

def assert_all(X): assert(np.all(sm.as_numpy(X)))
def assert_any(X): assert(np.any(sm.as_numpy(X)))
def assert_eq(X,Y): assert(np.all(sm.as_numpy(X) == sm.as_numpy(Y)))
def assert_ne(X,Y): assert(np.any(sm.as_numpy(X) != sm.as_numpy(Y)))
def assert_close(X,Y):
    X = sm.as_numpy(X).ravel()
    Y = sm.as_numpy(Y).ravel()
    if   X.dtype == np.float32: np.testing.assert_allclose(X,Y,rtol=1e-3,atol=np.inf)
    elif X.dtype == np.float64: np.testing.assert_allclose(X,Y,rtol=1e-6,atol=np.inf)
    else: assert_all(X == Y)

int_ranges = {np.bool:  (0,2), 
              np.int8:  (-7,8), np.uint8:  (0,8),
              np.int16: (-63,64), np.uint16: (0,64),
              np.int32: (-255,256), np.uint32: (0,256),
              np.int64: (-128**2+1,128**2), np.uint64: (0,128**2)}

dtype_short_name={np.bool: "b8",
                  np.int8:  "i8",  np.uint8:  "u8",
                  np.int16: "i16", np.uint16: "u16",
                  np.int32: "i32", np.uint32: "u32",
                  np.int64: "i64", np.uint64: "u64",
                  np.float32:"f32",np.float64:"f64",
                  }

def make_rand(n,m,dt):
    if dt in dtypes_float:
        return npr.randn(n,m) # deliberately don't cast to dtype "dt" yet
    else:
        return npr.randint(int_ranges[dt][0],int_ranges[dt][1],size=(n,m))

