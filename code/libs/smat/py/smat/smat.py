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
from smat_dll import *
from ctypes import *
from exceptions import *
import numpy as np
import string,atexit,__builtin__
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import cPickle as pickle
import util
import gc

bool   = np.bool
int8   = np.int8
uint8  = np.uint8
int16  = np.int16
uint16 = np.uint16
int32  = np.int32
uint32 = np.uint32
int64  = np.int64
uint64 = np.uint64
float32= np.float32
float64= np.float64
index   = int32   # should be same as c_index_t
uindex  = uint32  # should be same as c_uindex_t

tic = util.tic
toc = util.toc

_int2dtype = util._int2dtype
_dtype2int = util._dtype2int
_arg2dtype = util._arg2dtype

def int2dtype(i):
    return _int2dtype[i]

def dtype2int(dt):
    if type(dt) == type(None):return -1
    return _dtype2int[arg2dtype(dt)]

def dtype2str(dtype):
    dtype = _int2dtype(dll.actual_dtype(_dtype2int[dtype]))
    return dtype.__name__

def arg2dtype(s):
    # Can't just test s == None due to some weird bug where "np.dtype('float64') == None" returns true
    if type(s) == type(None):   return _int2dtype[dll.api_get_default_dtype()]
    return _arg2dtype[s]

_integral_types = (__builtin__.chr,__builtin__.int,__builtin__.long,int8,int16,int32,int64)
_py_max = __builtin__.max
_py_min = __builtin__.min
_py_sum = __builtin__.sum
_py_all = __builtin__.all
_py_any = __builtin__.any

_smat_exiting = False
_span = c_slice_t(0,c_slice_end)

_axis2int = { None: -1, 0:1, 1:0 }  # Only x and y axes supported right now

##############################################################

class sarray(object):
    '''
    A numpy-like wrapper around a streaming-mode matrix (smat) object.
    '''
    def __init__(self,ptr):
        self._ptr = ptr   # Just wrap a pointer to the C++ smat instance.
        self._attr = None # List of "extra" attributes that have been attached to this smat instance, if any

    def __del__(self):
        if dll != None and not _smat_exiting:  # dll may have already been unloaded if exiting application
            dll.api_delete(self._ptr)

    @property
    def size(self): return dll.api_size(self._ptr)

    @property
    def shape(self):
        s = c_shape_t(0,0,0)
        dll.api_shape(self._ptr,byref(s))
        return (s.y,s.x)

    @property
    def nrow(self): return dll.api_nrow(self._ptr) # number of rows

    @property
    def ncol(self): return dll.api_ncol(self._ptr) # number of cols

    @property
    def ndim(self): return 2   # TODO: support variable dimension arrays

    @property
    def dtype(self): return np.dtype(_int2dtype[dll.api_dtype(self._ptr)])

    @property
    def T(self): return transpose(self)

    def setattr(self,name,val):
        if self._attr is None:
            self._attr = set()
        self._attr.add(name)
        self.__setattr__(name,val)

    def getattr(self,name,default=None):
        return getattr(self,name) if self.hasattr(name) else default
    
    def hasattr(self,name):
        return self._attr is not None and name in self._attr
    
    def clearattr(self):
        if self._attr:
            for attr in _copy(self._attr):
                delattr(self,attr)

    def copyattr(self,A,deep=False):
        if A is self:
            return
        self.clearattr()
        if A._attr:
            for attr in A._attr:
                val = A.getattr(attr)
                if deep:
                    val = _deepcopy(val)
                self.setattr(attr,val)

    def copy(self): return _deepcopy(self)

    def __delattr__(self,name):
        object.__delattr__(self,name)
        if self._attr is not None:
            self._attr.remove(name)

    # These provided for the standard library copy.copy() and copy.deepcopy() functions
    def __deepcopy__(self,memo):
        A = empty(self.shape,self.dtype)  # deep copy, including all data and extra attributes
        A[:] = self
        sync()
        if self._attr is not None:
            for attr in self._attr:
                A.setattr(attr,_deepcopy(getattr(self,attr),memo))
        return A 

    def __copy__(self):  # shallow copy, where things like shape can be modified separately from the original, but shares the same underlying data array
        A = self[:]
        sync()
        if self._attr is not None:
            for attr in self._attr:
                A.setattr(attr,getattr(self,attr))
        return A

    def astype(self,dtype,copy=False): 
        return as_sarray(self,dtype,copy)

    def asnumpy(self,async=False,out=None):
        if   out == None:                    out = np.empty(self.shape,dtype=self.dtype,order='C')
        elif not isinstance(out,np.ndarray): raise ValueError("Keyword argument 'out' must be a numpy array.\n")
        elif out.shape != A.shape:           raise ValueError("Keyword argument 'out' must have matching dimensions.\n")
        if out.ndim > 1:
            rstride = out.strides[0]
            cstride = out.strides[1]
        else:
            cstride = out.strides[0]
            rstride = cstride*A.size
        # Looks good, so issue the copy operation.
        dll.api_copy_to(self._ptr,out.ctypes.data_as(c_void_p),rstride,cstride)
        if not async:
            sync()  # Wait for writes to finish.
        return out

    def isscalar(self):
        return self.size == 1
    
    # Do NOT provide len() and iteration, because it gives numpy a way to 
    # silently cause terrible performance problems when mixed with sarrays.
    
    def __len__(self):
        #raise NotImplementedError("An sarray does not support __len__, to prevent accidental operations mixing numpy ndarrays.")
        return self.nrow
    
    def __iter__(self): 
        #raise NotImplementedError("An sarray does not support __iter__, to prevent accidental operations mixing numpy ndarrays.")
        for row in xrange(self.shape[0]):  # Iterate over rows
            yield self[row]

    ############################### PICKLING ##############################

    def __getstate__(self):
        data = self.asnumpy()
        attrdict = {name : pickle.dumps(self.getattr(name)) for name in self._attr} if self._attr is not None else None
        return (data,attrdict)

    def __setstate__(self,state):
        data,attrdict = state
        self._ptr = dll.api_empty(_make_shape_p(data.shape),dtype2int(data.dtype))
        dll.api_copy_from(self._ptr,data.ctypes.data_as(c_void_p),data.strides[0],data.strides[1]) # Copy from data
        self._attr = None
        if attrdict:
            for name,value in attrdict.items():
                self.setattr(name,pickle.loads(value))
    
    ############################### RESHAPING ##############################

    def reshape(self,shape):
        return sarray(dll.api_reshape(self._ptr,_make_shape_p(shape)))

    def ravel(self):
        return sarray(dll.api_reshape(self._ptr,_make_shape_p((-1,1))))

    ############################### SLICING ################################

    def __getitem__(self,i):
        ti = type(i)
        if ti != tuple:
            # One slice dimension
            if _is_full_slice(i):
                return self
            rows = _make_gettable_slice_p(i)
            cols = _span
        else:
            # Two slice dimensions
            if len(i) != 2:
                raise IndexError("Too many indices.\n")
            rows = _make_gettable_slice_p(i[0])
            cols = _make_gettable_slice_p(i[1]) if i[1] != slice(None) else _make_gettable_slice_p(slice(0,self.ncol))

        if type(rows) == c_slice_t and type(cols) == c_slice_t:
            # Contiguous row indices, contiguous col indices
            return sarray(dll.api_slice(self._ptr,byref(rows),byref(cols)))
        elif type(rows) == c_slice_t and type(cols) == np.ndarray:
            # Contiguous row indices, list of individual col indices
            raise NotImplementedError("List-based slicing not implemented")
        else:
            raise NotImplementedError("Gettable slicing only supports slice-, integer-, or list-based indexing.")


    def __setitem__(self,i,val):
        ti = type(i)
        if ti != tuple:
            # One slice dimension
            if _is_full_slice(i):
                self.assign(val)
                return
            rows = _make_settable_slice_p(i)
            cols = _span
        else:
            # Two slice dimensions
            if len(i) != 2:
                raise IndexError("Too many indices.\n")
            rows = _make_settable_slice_p(i[0])
            cols = _make_settable_slice_p(i[1]) if i[1] != slice(None) else _make_settable_slice_p(slice(0,self.ncol))
        
        if type(rows) == c_slice_t and type(cols) == c_slice_t:
            # Contiguous row indices, contiguous col indices
            view = sarray(dll.api_slice(self._ptr,byref(rows),byref(cols)))
            view.assign(val)  # use .assign to avoid recursion
        else:
            raise NotImplementedError("Settable slicing only supports slice- or integer-based indexing.")

    def assign(self,val):
        val = as_sarray(val)
        dll.api_assign(self._ptr,val._ptr)
        return self

    ######################## COMPARISON OPERATIONS #########################

    def __eq__(self,other): 
        if isscalar(other):       other = _scalar2smat(other)
        if type(other) == sarray: return sarray(dll.api_eq(self._ptr,other._ptr))
        return False

    def __ne__(self,other): return self.__eq__(other).__invert__()
    def __lt__(self,other): other = _scalar2smat(other); return sarray(dll.api_lt(self._ptr,other._ptr))
    def __le__(self,other): other = _scalar2smat(other); return sarray(dll.api_le(self._ptr,other._ptr))
    def __gt__(self,other): other = _scalar2smat(other); return sarray(dll.api_gt(self._ptr,other._ptr))
    def __ge__(self,other): other = _scalar2smat(other); return sarray(dll.api_ge(self._ptr,other._ptr))

    ######################## LOGICAL/BITWISE OPERATIONS #########################

    def __or__(self,other):  other = _scalar2smat(other); return sarray(dll.api_or (self._ptr,other._ptr))
    def __xor__(self,other): other = _scalar2smat(other); return sarray(dll.api_xor(self._ptr,other._ptr))
    def __and__(self,other): other = _scalar2smat(other); return sarray(dll.api_and(self._ptr,other._ptr))

    def __ror__(self,other):  other = _scalar2smat(other); return sarray(dll.api_or (other._ptr,self._ptr))
    def __rxor__(self,other): other = _scalar2smat(other); return sarray(dll.api_xor(other._ptr,self._ptr))
    def __rand__(self,other): other = _scalar2smat(other); return sarray(dll.api_and(other._ptr,self._ptr))

    def __ior__(self,other):  other = _scalar2smat(other); dll.api_ior (self._ptr,other._ptr); return self
    def __ixor__(self,other): other = _scalar2smat(other); dll.api_ixor(self._ptr,other._ptr); return self
    def __iand__(self,other): other = _scalar2smat(other); dll.api_iand(self._ptr,other._ptr); return self

    ########################## UNARY OPERATORS ##########################

    def __neg__(self):   return sarray(dll.api_neg(self._ptr))
    def __abs__(self):   return sarray(dll.api_abs(self._ptr))
    def __invert__(self):return sarray(dll.api_not(self._ptr))
    def __nonzero__(self):
        if self.size != 1:
            raise ValueError("Truth value of matrix is ambiguous; use all() or any().")
        return self.asnumpy().__nonzero__()  # must pull back from device
    def __int__(self):   return int(self.asnumpy())
    def __long__(self):  return long(self.asnumpy())
    def __float__(self): return float(self.asnumpy())

    ########################## ARITHMETIC OPERATORS #########################

    def __add__(self,other): other = _scalar2smat(other); return sarray(dll.api_add(self._ptr,other._ptr))
    def __sub__(self,other): other = _scalar2smat(other); return sarray(dll.api_sub(self._ptr,other._ptr))
    def __mul__(self,other): other = _scalar2smat(other); return sarray(dll.api_mul(self._ptr,other._ptr))
    def __div__(self,other): other = _scalar2smat(other); return sarray(dll.api_div(self._ptr,other._ptr))
    def __mod__(self,other): other = _scalar2smat(other); return sarray(dll.api_mod(self._ptr,other._ptr))
    def __pow__(self,other): other = _scalar2smat(other); return sarray(dll.api_pow(self._ptr,other._ptr))

    def __radd__(self,other): other = _scalar2smat(other); return sarray(dll.api_add(other._ptr,self._ptr))
    def __rsub__(self,other): other = _scalar2smat(other); return sarray(dll.api_sub(other._ptr,self._ptr))
    def __rmul__(self,other): other = _scalar2smat(other); return sarray(dll.api_mul(other._ptr,self._ptr))
    def __rdiv__(self,other): other = _scalar2smat(other); return sarray(dll.api_div(other._ptr,self._ptr))
    def __rmod__(self,other): other = _scalar2smat(other); return sarray(dll.api_mod(other._ptr,self._ptr))
    def __rpow__(self,other): other = _scalar2smat(other); return sarray(dll.api_pow(other._ptr,self._ptr))

    def __iadd__(self,other): other = _scalar2smat(other); dll.api_iadd(self._ptr,other._ptr); return self
    def __isub__(self,other): other = _scalar2smat(other); dll.api_isub(self._ptr,other._ptr); return self
    def __imul__(self,other): other = _scalar2smat(other); dll.api_imul(self._ptr,other._ptr); return self
    def __idiv__(self,other): other = _scalar2smat(other); dll.api_idiv(self._ptr,other._ptr); return self
    def __imod__(self,other): other = _scalar2smat(other); dll.api_imod(self._ptr,other._ptr); return self
    def __ipow__(self,other): other = _scalar2smat(other); dll.api_ipow(self._ptr,other._ptr); return self

    ########################## REDUCE OPERATIONS ##########################

    def max(self,axis=None):  return sarray(dll.api_max( self._ptr,_axis2int[axis]))  #.reshape((1,-1))   # mimick numpy's conversion to 1d row vector ? Naaaah, it's too annoying; pretend keep_dim is on by default
    def min(self,axis=None):  return sarray(dll.api_min( self._ptr,_axis2int[axis]))
    def sum(self,axis=None):  return sarray(dll.api_sum( self._ptr,_axis2int[axis]))
    def mean(self,axis=None): return sarray(dll.api_mean(self._ptr,_axis2int[axis]))
    def nnz(self,axis=None):  return sarray(dll.api_nnz( self._ptr,_axis2int[axis]))
    def any(self,axis=None):  return sarray(dll.api_any( self._ptr,_axis2int[axis]))
    def all(self,axis=None):  return sarray(dll.api_all( self._ptr,_axis2int[axis]))

    ########################## REPEAT OPERATORS ##########################
    
    def _rep_op(self,n,axis,op):
        if axis not in (None,0,1): raise ValueError("Axis must be None, 0 or 1.")
        A = self
        if isinstance(n,(tuple,list)):
            if axis is not None: raise ValueError("Axis must be None if n is a tuple")
            if len(n) == 1:
                n = (n[0],1) if axis == 0 else (1,n[0])
        else:
            if axis is None:
                A = self.ravel()  # emulate numpy flattening on axis=None
            n = (n,1) if axis == 0 else (1,n)
        B = sarray(op(A._ptr,_make_shape_p(n)))
        return B if axis is not None else B.reshape((-1,1))
        
    def repeat(self,n,axis=None): return self._rep_op(n,axis,dll.api_repeat)
    def tile(self,n,axis=None):   return self._rep_op(n,axis,dll.api_tile)

    ########################## OTHER OPERATORS ##########################

    def __repr__(self):
        max_device_rows = 512 if self.shape[1] > 16 else 2048
        if True or self.shape[0] <= max_device_rows:
            A = self.asnumpy()
        else:
            # If this is a huge matrix, only copy the start and end of the matrix to the host,
            # so that printing is faster, and so that interactive debuggers like Visual Studio
            # are faster (otherwise have to wait for huge memory transfers at each breakpoint,
            # to update the variable values in Visual Studio's Locals window).
            # For now, just handle the case when there are many rows.
            A = np.empty((max_device_rows,)+self.shape[1:],self.dtype)
            A[:max_device_rows/2] = self[:max_device_rows/2].asnumpy()
            A[max_device_rows/2:] = self[-max_device_rows/2:].asnumpy()
        txt = A.__repr__().replace('array(', 'sarray(').replace(' [','  [')
        if txt.find("dtype=") == -1:
            txt = txt[:-1] + (",dtype=%s)" % A.dtype)
        return txt
    
    def __str__(self):  return self.asnumpy().__str__()

##############################################################

def empty(shape,dtype=None): return sarray(dll.api_empty(_make_shape_p(shape),dtype2int(dtype)))
def zeros(shape,dtype=None): return sarray(dll.api_zeros(_make_shape_p(shape),dtype2int(dtype)))
def ones (shape,dtype=None): return sarray(dll.api_ones (_make_shape_p(shape),dtype2int(dtype)))
def empty_like(A,dtype=None):return sarray(dll.api_empty_like(A._ptr,dtype2int(dtype)))
def zeros_like(A,dtype=None):return sarray(dll.api_zeros_like(A._ptr,dtype2int(dtype)))
def ones_like (A,dtype=None):return sarray(dll.api_ones_like (A._ptr,dtype2int(dtype)))

def eye  (n,m=None,k=0,dtype=None): 
    if _dtype2int.has_key(m): dtype = m; m = None
    if m != None and n != m: raise NotImplementedError("Non-square identity matrices not supported.\n")
    if k != 0: raise NotImplementedError("Off-diagonal identity matrices not supported.\n")
    return sarray(dll.api_eye(n,dtype2int(dtype)))

def identity(n,dtype=None): 
    return sarray(dll.api_eye(n,dtype2int(dtype)))

def arange(*args,**kwargs):
    if len(args) == 0: raise ValueError("Not enough arguments.\n")
    if len(args) == 1: start = 0;       stop = args[0]
    if len(args) == 2: start = args[0]; stop = args[1]
    return sarray(dll.api_arange(start,stop,dtype2int(kwargs.get("dtype",None))))

def rand(n,m=1,dtype=None):        return sarray(dll.api_rand(     _make_shape_p((int(n),int(m))),dtype2int(dtype)))
def randn(n,m=1,dtype=None):       return sarray(dll.api_randn(    _make_shape_p((int(n),int(m))),dtype2int(dtype)))
def bernoulli(shape,p,dtype=None): return sarray(dll.api_bernoulli(_make_shape_p(shape),p,dtype2int(dtype)))
def rand_seed(seed):           dll.api_set_rand_seed(seed)
def sync():  dll.api_sync()

###############################################################

# Maps python/numpy type to corresponding smat scalar constructor.
_smat_const_lookup = {
    __builtin__.bool:   (dll.api_const_b8, c_bool),
                bool:   (dll.api_const_b8, c_bool),
    __builtin__.chr:    (dll.api_const_i8, c_byte),
                int8:   (dll.api_const_i8, c_byte),
                uint8:  (dll.api_const_u8, c_ubyte),
                int16:  (dll.api_const_i16,c_short),
                uint16: (dll.api_const_u16,c_ushort),
    __builtin__.int:    (dll.api_const_i32,c_int),
                int32:  (dll.api_const_i32,c_int), 
                uint32: (dll.api_const_u32,c_uint),
    __builtin__.long:   (dll.api_const_i64,c_longlong),
                int64:  (dll.api_const_i64,c_longlong),
                uint64: (dll.api_const_u64,c_ulonglong),
                float32:(dll.api_const_f32,c_float),
                float64:(dll.api_const_f64,c_double),
    __builtin__.float:  (dll.api_const_f64,c_double)
    }

def as_sarray(A,dtype=None,copy=False,force=False):
    if isinstance(A,sarray):
        # Type SARRAY (smat)
        if dtype is None or A.dtype == dtype:
            return A if not copy else A.copy()  # Return a reference to A, or a direct copy
        B = empty(A.shape,dtype)
        B[:] = A       # Return type-converted copy of A
        return B
    if isinstance(A,list):
        if dtype is None:
            if type(A[0]) == float or (isinstance(A[0],(list,tuple)) and type(A[0][0]) == float):
                dtype = get_default_dtypef() # Convert to "float32" or "float64" depending on current default for floats
        A = np.asarray(A,dtype=dtype)        # Let numpy do the dirty work of rearranging the data, and fall through to the next if statement.
    if isinstance(A,np.ndarray):
        # Type NDARRAY (numpy)
        if dtype != None and dtype != A.dtype:
            A = np.require(A,dtype=dtype)  # Implicit conversion first, since simultaneous copy-and-convert is not supported by smat.
        if A.ndim > 2:
            raise NotImplementedError("Only 1- or 2-D sarrays are supported.")
        if not A.flags['C_CONTIGUOUS']:
            if not force:
                raise TypeError("Expected C-contiguous ndarray, but received F-contiguous; use force=True to allow automatic conversion.")
            A = np.require(A,requirements=["C_CONTIGUOUS"])
        if A.ndim > 1:
            rstride = A.strides[0]
            cstride = A.strides[1]
        else:
            rstride = A.strides[0]
            cstride = rstride
        B = empty(A.shape,A.dtype)
        dll.api_copy_from(B._ptr,A.ctypes.data_as(c_void_p),rstride,cstride) # Return copy of A
        return B
    if np.isscalar(A):
        # Type SCALAR; convert to a 1x1 sarray of the appropriate type
        func,ctype = _smat_const_lookup[type(A) if dtype is None else arg2dtype(dtype)]
        b = sarray(func(ctype(A)))               # Return scalar wrapped in an smat
        return b
    raise TypeError("Unrecognized type '%s'.\n" % str(type(A)))

asarray = as_sarray
array = as_sarray

def index_array(A):  return as_sarray(A,dtype=index)
def uindex_array(A): return as_sarray(A,dtype=uindex)

def asnumpy(A,async=False,out=None):
    try:
        if isinstance(A,list):   return list(as_numpy(item) for item in A)
        if isinstance(A,tuple):  return tuple(as_numpy(item) for item in A)
        if isinstance(A,sarray): return A.asnumpy(async,out)
        if out != None: raise ValueError("Keyword argument 'out' only supported when input is of type sarray.")
        # If not an SARRAY, pass it along to the regular numpy asarray() function
        return np.asarray(A) if A is not None else None
    except MemoryError as mem:
        print ("OUT OF MEMORY in asnumpy() with A=%s (%d bytes)" % (str(A.shape),A.size))
        raise

as_numpy = asnumpy

def as_numpy_array(A): return as_numpy(A)  # gnumpy calls it as_numpy_array 

def isarray(x): return type(x) == sarray

def isscalar(x):
    if type(x) == sarray:  return x.isscalar()
    if type(x) == str: return False  # for some reason np.isscalar returns true for strings
    return np.isscalar(x)

def sign(A):      return sarray(dll.api_sign(A._ptr))       if isinstance(A,sarray) else np.sign(A)
def signbit(A):   return sarray(dll.api_signbit(A._ptr))    if isinstance(A,sarray) else np.signbit(A,out=np.empty(A.shape,A.dtype)) # force numpy to use input dtype instead of bool
def sqrt(A):      return sarray(dll.api_sqrt(A._ptr))       if isinstance(A,sarray) else np.sqrt(A)
def square(A):    return sarray(dll.api_square(A._ptr))     if isinstance(A,sarray) else np.square(A)
def sin(A):       return sarray(dll.api_sin(A._ptr))        if isinstance(A,sarray) else np.sin(A)
def cos(A):       return sarray(dll.api_cos(A._ptr))        if isinstance(A,sarray) else np.cos(A)
def tan(A):       return sarray(dll.api_tan(A._ptr))        if isinstance(A,sarray) else np.tan(A)
def arcsin(A):    return sarray(dll.api_arcsin(A._ptr))     if isinstance(A,sarray) else np.arcsin(A)
def arccos(A):    return sarray(dll.api_arccos(A._ptr))     if isinstance(A,sarray) else np.arccos(A)
def arctan(A):    return sarray(dll.api_arctan(A._ptr))     if isinstance(A,sarray) else np.arctan(A)
def sinh(A):      return sarray(dll.api_sinh(A._ptr))       if isinstance(A,sarray) else np.sinh(A)
def cosh(A):      return sarray(dll.api_cosh(A._ptr))       if isinstance(A,sarray) else np.cosh(A)
def tanh(A):      return sarray(dll.api_tanh(A._ptr))       if isinstance(A,sarray) else np.tanh(A)
def arcsinh(A):   return sarray(dll.api_arcsinh(A._ptr))    if isinstance(A,sarray) else np.arcsinh(A)
def arccosh(A):   return sarray(dll.api_arccosh(A._ptr))    if isinstance(A,sarray) else np.arccosh(A)
def arctanh(A):   return sarray(dll.api_arctanh(A._ptr))    if isinstance(A,sarray) else np.arctanh(A)
def exp(A):       return sarray(dll.api_exp(A._ptr))        if isinstance(A,sarray) else np.exp(A)
def exp2(A):      return sarray(dll.api_exp2(A._ptr))       if isinstance(A,sarray) else np.exp2(A)
def log(A):       return sarray(dll.api_log(A._ptr))        if isinstance(A,sarray) else np.log(A)
def log2(A):      return sarray(dll.api_log2(A._ptr))       if isinstance(A,sarray) else np.log2(A)
def logistic(A):  return sarray(dll.api_logistic(A._ptr))   if isinstance(A,sarray) else 1/(1+np.exp(-A))
def round(A):     return sarray(dll.api_round(A._ptr))      if isinstance(A,sarray) else np.round(A)
def floor(A):     return sarray(dll.api_floor(A._ptr))      if isinstance(A,sarray) else np.floor(A)
def ceil(A):      return sarray(dll.api_ceil(A._ptr))       if isinstance(A,sarray) else np.ceil(A)
def clip(A,lo=0.,hi=1.):return sarray(dll.api_clip(A._ptr,lo,hi)) if isinstance(A,sarray) else np.clip(A,lo,hi)
def isinf(A):     return sarray(dll.api_isinf(A._ptr))      if isinstance(A,sarray) else np.isinf(A)
def isnan(A):     return sarray(dll.api_isnan(A._ptr))      if isinstance(A,sarray) else np.isnan(A)
def transpose(A): return sarray(dll.api_trans(A._ptr))      if isinstance(A,sarray) else np.transpose(A)

def dot(A,B,out=None):
    if not isinstance(A,sarray) or not isinstance(B,sarray):
        assert not isinstance(A,sarray) and not isinstance(B,sarray), "Cannot perform product on sarray and numpy array."
        return np.dot(A,B,out=out)
    if out is None:
        return sarray(dll.api_dot(A._ptr,B._ptr))
    assert isinstance(out,sarray), "Output must be an sarray."
    dll.api_dot_out(A._ptr,B._ptr,out._ptr)

def dot_tn(A,B,out=None):
    if not isinstance(A,sarray) or not isinstance(B,sarray):
        assert not isinstance(A,sarray) and not isinstance(B,sarray), "Cannot perform product on sarray and numpy array."
        return np.dot(A.T,B,out=out)
    if out is None:
        return sarray(dll.api_dot_tn(A._ptr,B._ptr))
    assert isinstance(out,sarray), "Output must be an sarray."
    dll.api_dot_tn_out(A._ptr,B._ptr,out._ptr)

def dot_nt(A,B,out=None):
    if not isinstance(A,sarray) or not isinstance(B,sarray):
        assert not isinstance(A,sarray) and not isinstance(B,sarray), "Cannot perform product on sarray and numpy array."
        return np.dot(A,B.T,out=out)
    if out is None:
        return sarray(dll.api_dot_nt(A._ptr,B._ptr))
    assert isinstance(out,sarray), "Output must be an sarray."
    dll.api_dot_nt_out(A._ptr,B._ptr,out._ptr)
    
def dot_tt(A,B,out=None):
    if not isinstance(A,sarray) or not isinstance(B,sarray):
        assert not isinstance(A,sarray) and not isinstance(B,sarray), "Cannot perform product on sarray and numpy array."
        return np.dot(A.T,B.T,out=out)
    if out is None:
        return sarray(dll.api_dot_tt(A._ptr,B._ptr))
    assert isinstance(out,sarray), "Output must be an sarray."
    dll.api_dot_tt_out(A._ptr,B._ptr,out._ptr)

def relu(A):      return maximum(0,A)       if isinstance(A,sarray) else np.maximum(0,A)  # Returns Z = relu(A)
def relu_grad(Z): return sign(Z)            if isinstance(Z,sarray) else np.sign(Z)       # Returns d/dA(relu)(A) = sign(Z) where Z = relu(A)

def _binary_elemwise(sop,nop,A,B,*args):
    if type(A) == sarray and np.isscalar(B): B = as_sarray(B,dtype=A.dtype)
    if type(B) == sarray and np.isscalar(A): A = as_sarray(A,dtype=B.dtype)
    if type(A) == sarray and type(B) == sarray: return sarray(sop(A._ptr,B._ptr))
    if nop is not None: return nop(A,B,*args)
    raise RuntimeException("Both arguments should be of type sarray.")

def maximum(A,B): return _binary_elemwise(dll.api_maximum,np.maximum,A,B)
def minimum(A,B): return _binary_elemwise(dll.api_minimum,np.minimum,A,B)

def isclose(A,B,rtol=None,atol=None):
    if rtol == None: rtol = _default_rtol(A.dtype)
    if atol == None: atol = _default_atol(A.dtype)
    return _binary_elemwise(dll.api_isclose,None,A,B,rtol,atol)

def allclose(A,B,rtol=None,atol=None):
    if rtol == None: rtol = _default_rtol(A.dtype)
    if atol == None: atol = _default_atol(A.dtype)
    return _binary_elemwise(dll.api_allclose,np.allclose,A,B,rtol,atol)

def _reduce_op(A,axis,sop,nop,pyop):
    if isinstance(A,sarray):     return sop(A,axis)
    if isinstance(A,np.ndarray): return nop(A,axis)
    if pyop == None:             raise TypeError("Invalid type for reduce operation.")
    if isinstance(A,list) and axis==None: return pyop(A)
    return pyop(A,axis) # A is first item, axis is second item (e.g. call __builtin__.min(A,axis))

def max(A,axis=None):  return _reduce_op(A,axis,sarray.max,np.ndarray.max,_py_max)
def min(A,axis=None):  return _reduce_op(A,axis,sarray.min,np.ndarray.min,_py_min)
def sum(A,axis=None):  return _reduce_op(A,axis,sarray.sum,np.ndarray.sum,_py_sum)
def mean(A,axis=None): return _reduce_op(A,axis,sarray.mean,np.ndarray.mean,None)
def nnz(A,axis=None):  return A.nnz(axis)  if isinstance(A,sarray) else (np.count_nonzero(A) if axis == None else np.sum(A!=0,axis))
def all(A,axis=None):  return _reduce_op(A,axis,sarray.all,np.ndarray.all,_py_all)
def any(A,axis=None):  return _reduce_op(A,axis,sarray.any,np.ndarray.any,_py_any)
def count_nonzero(A):  return A.nnz() if isinstance(A,sarray) else np.count_nonzero(A)

def repeat(A,n,axis=None):
    if isinstance(A,sarray):
        return A.repeat(n,axis)
    return np.repeat(A,n,axis)

def tile(A,n,axis=None):
    if isinstance(A,sarray):
        return A.tile(n,axis)
    assert axis is None
    return np.tile(A,n)

def diff(A,n=1,axis=1):
    if not isinstance(A,sarray): return np.diff(A,n,axis)
    if n <= 0:  return A
    B = diff(A,n-1,axis)
    return sarray(dll.api_diff(B._ptr,_axis2int[axis]))

def softmax(A,axis=1): return sarray(dll.api_softmax(A._ptr,_axis2int[axis]))
def apply_mask(A,mask): dll.api_apply_mask(A._ptr,mask._ptr)

def logical_not(A):     return sarray(dll.api_lnot(A._ptr))        if isinstance(A,sarray) else np.logical_not(A)
def logical_or(A,B):    return sarray(dll.api_lor(A._ptr,B._ptr))  if isinstance(A,sarray) and isinstance(B,sarray) else np.logical_or(A,B)
def logical_and(A,B):   return sarray(dll.api_land(A._ptr,B._ptr)) if isinstance(A,sarray) and isinstance(B,sarray) else np.logical_and(A,B)

###############################################################
# These extra global functions are provided so that there's an
# easy, named function available for all smat operations.

def eq(A,B): return A == B
def ne(A,B): return A != B
def lt(A,B): return A <  B
def le(A,B): return A <= B
def gt(A,B): return A >  B
def ge(A,B): return A >= B
def _or(A,B):  return A | B
def _xor(A,B): return A ^ B
def _and(A,B): return A & B
def _abs(A):   return abs(A)
def invert(A):     return ~A
def reciprocal(A): return 1./A
def negative(A):   return -A
def add(A,B):      return A+B
def subtract(A,B): return A-B
def multiply(A,B): return A*B
def divide(A,B):   return A/B
def mod(A,B):      return A%B
def power(A,B):    return A**B
def max_x(A):  return A.max(axis=1)
def max_y(A):  return A.max(axis=0)
def min_x(A):  return A.min(axis=1)
def min_y(A):  return A.min(axis=0)
def sum_x(A):  return A.sum(axis=1)
def sum_y(A):  return A.sum(axis=0)
def mean_x(A): return A.mean(axis=1)
def mean_y(A): return A.mean(axis=0)
def nnz_x(A):  return A.nnz(axis=1)
def nnz_y(A):  return A.nnz(axis=0)
def any_x(A):  return A.any(axis=1)
def any_y(A):  return A.any(axis=0)
def all_x(A):  return A.all(axis=1)
def all_y(A):  return A.all(axis=0)
def diff_x(A): return A.diff(axis=1)
def diff_y(A): return A.diff(axis=0)
def repeat_x(A,n): return A.repeat(n,axis=1)
def repeat_y(A,n): return A.repeat(n,axis=0)
def tile_x(A,n):   return A.tile(n,axis=1)
def tile_y(A,n):   return A.tile(n,axis=0)
def softmax_x(A):      return softmax(A,axis=1)
def softmax_y(A):      return softmax(A,axis=0)

###############################################################

def _as_tuple(x):      return x if type(x) != tuple else (x,)
def _is_full_slice(x): return type(x) == slice and x.start == None and x.stop == None
def _scalar2smat(x):
    if type(x) == sarray:  return x
    if not np.isscalar(x): raise TypeError("Type %s not directly supported in this operation.\n" % str(type(x)))
    func,ctype = _smat_const_lookup[type(x)]
    b = sarray(func(ctype(x)))                    # Return scalar wrapped in an smat
    return b

def _make_settable_slice_p(x):
    tx = type(x)
    if tx == slice:
        if not x.step in (None, 1):
            raise NotImplementedError("Settable slicing is only supported for contiguous ranges.\n")
        return c_slice_t(x.start or 0L, x.stop if x.stop != None else c_slice_end)
    if tx in _integral_types:
        return c_slice_t(x,x+1)
    if tx == sarray:
        if x.dtype == bool: raise NotImplementedError("Logical slicing not yet implemented.\n")
        else:               raise NotImplementedError("Settable list-based slicing not yet implemented.\n")
    raise NotImplementedError("Settable index must be integral or contiguous slice.\n")

def _make_gettable_slice_p(x):
    tx = type(x)
    if tx == slice:
        if not x.step in (None, 1):
            return np.arange(x.start,x.stop,x.step,dtype=index)
        return c_slice_t(x.start or 0L, x.stop if x.stop != None else c_slice_end)
    if tx in _integral_types:
        return c_slice_t(x,x+1)
    if tx == list or tx == tuple:
        x = np.asarray(x)
        tx = np.ndarray
    if tx == np.ndarray:
        x = as_sarray(x)
        tx = sarray
    if tx == sarray:
        if x.dtype == bool: raise NotImplementedError("Logical slicing not yet implemented.\n")
        if x.ndim != 1:     raise NotImplementedError("List-based slicing must use 1-dimensional vector.")
        return x
    raise NotImplementedError("Gettable index must be integral, slice, or list.\n")


def _make_shape_p(shape):
    if isinstance(shape,int): return byref(c_shape_t(1,shape,1))
    if not isinstance(shape,tuple) or not len(shape) in [1,2]:
        raise ValueError("Shape must be a tuple of length 1 or 2.\n")
    if len(shape) == 1: return byref(c_shape_t(1,shape[0],1))
    return byref(c_shape_t(shape[1],shape[0],1))

def _kwargs2argv(kwargs):
    as_str = lambda v: str(val) if not isinstance(val,list) else string.join([str(v) for v in val],",")
    args = [key + '=' + as_str(val) for key,val in kwargs.items()]
    argv = (c_char_p * len(args))()  # convert list into ctype array of char*
    argv[:] = args                   # make each char* item point to the corresponding string in 'args'
    return argv

###############################################################

def set_backend(name,**kwargs):
    gc.collect()
    argv = _kwargs2argv(kwargs)
    return dll.api_set_backend(c_char_p(name),len(argv),argv)

def set_backend_options(**kwargs):
    gc.collect()
    argv = _kwargs2argv(kwargs)
    return dll.api_set_backend_options(len(argv),argv)

def reset_backend(**kwargs):
    gc.collect()
    argv = _kwargs2argv(kwargs)
    return dll.api_reset_backend(len(argv),argv)


def get_backend_name():     return str(dll.api_get_backend_info().name)
def get_supported_dtypes():   return [_int2dtype[dt] for dt in _dtype2int.values() if dll.api_is_dtype_supported(dt) == True]
def set_default_dtype(dt):  dll.api_set_default_dtype(dtype2int(dt))
def set_default_dtypef(dt): dll.api_set_default_dtypef(dtype2int(dt))
def get_default_dtype():    return int2dtype(dll.api_get_default_dtype())
def get_default_dtypef():   return int2dtype(dll.api_get_default_dtypef())
def get_dtype_size(dt):     return int(dll.api_dtype_size(dtype2int(dt)))

def get_backend_info():
    info = c_backend_info()
    dll.api_get_backend_info(byref(info))
    return backend_info(info)

class backend_info(object):
    def __init__(self,info):
        self.uuid    = int(info.uuid)
        self.name    = str(info.name)
        self.version = str(info.version)
        self.device  = str(info.device)

    def __repr__(self):
        return "%s (v%s) using %s\n" % (self.name,self.version,self.device)

def get_heap_status():
    info = c_heap_status()
    dll.api_get_heap_status(byref(info))
    return heap_status(info)

class heap_status(object):
    def __init__(self,info):
        self.host_total   = long(info.host_total)
        self.host_avail   = long(info.host_avail)
        self.host_used    = long(info.host_used)
        self.device_total = long(info.device_total)
        self.device_avail = long(info.device_avail)
        self.device_used  = long(info.device_used)
        self.device_committed = long(info.device_committed)

    def __repr__(self):
        string = ''
        for name in ['host_total','host_avail','host_used','device_total','device_avail','device_used','device_committed']:
            string += '%s: %s\n' % (name,util.format_bytecount(self.__dict__[name],fmt="2.2cM"))
        return string

def autotune():
    dll.api_autotune_backend()

def destroy_backend(force=False):
    """
    Destroys the backend, including any device resources associated with the current thread.
    If there are outstanding handles to memory alloations (e.g. an sarray instance still
    holding on to memory used by the backend) then the call will fail; use force=True to override,
    though the program may later crash due to those objects holding invalid pointers.
    """
    gc.collect()
    dll.api_destroy_backend(force)



#########################################################################
# dropout functions

def dropout(X, rate, test_mode=False):
    """
    Applies dropout to input matrix X using the given dropout rate [0,1).
    
    If test_mode is false, returns pair (Z, M) where Z = M * X and M is a 
    matrix of bernoulli trials (M.dtype is bool).

    If test_mode is True, returns pair (Z, None) where Z = (1-rate)*X.
    """
    if test_mode:
        Z = (1-rate)*X
        return Z, None
    else:
        Z = empty_like(X)
        M = empty(X.shape,dtype=bool)
        dll.api_dropout_fp_tr(X._ptr, rate, Z._ptr, M._ptr)
        return Z, M


def dropout_grad(dZ, M=None, rate=None):
    """
    Backpropagates differentials dZ at the output of dropout operation, 
    returning the differentials dX at the input of that operation.

    If M is specified, the return value is dX = M * dZ.
    
    If rate is specified, then it is assumed you are trying to back-propagate
    through the mean network (e.g. for backprop-to-input on an already trained model)
    and so it returns dX = (1-rate)*dZ.
    """
    if M is not None:
        dX = empty_like(dZ)
        dll.api_dropout_bp_tr(dZ._ptr, M._ptr, dX._ptr)
        return dX
    else:
        dX = (1-rate)*dZ
        return dX


def maskout(X, M):
    """
    Inplace masking of A with mask M
    Equivalent to X[i] = M[i] ? X[i] : 0 where M is of dtype bool.
    Notice that this replaces NaN with zero, unlike X *= M
    """
    dll.api_maskout(X._ptr, A._ptr)

#########################################################################
# CUDNN functions

def featuremap_bias(fmaps, dims, bias, cpu_check=False):
    """
    Adds a separate bias to each featuremap generated by a convolution,
    where 'dims' is the size of each feature map, either (wd, ht) or wd*ht.
    The operation is in-place, so it modifies the existing fmaps array.

    Let 
      n = number of images
      c = number of feature maps
      d = number of elements in each feature map (e.g. width * height)

    Then
      fmaps must be (n) x (c*d)
      bias  must be (c) x (1)

    The computation adds bias[i] to all elements of featuremap i, 
    across all images:
      for i in range(c):
        fmaps[:][i*d:(i+1)*d] += bias[i]

    If cpu_check is True, then the result of conv2 will be compared to a
    simple CPU implementation to make sure absolute and relative error is
    below an internal certain threshold; used for unit tests.
    """

    cfg = c_featuremap_bias_cfg_t(int(np.prod(dims)), True, cpu_check)
    cudnn_dll().api_featuremap_bias(fmaps._ptr, bias._ptr, byref(cfg))


def featuremap_bias_grad(fmapsgrad, dims, biasgrad=None, accumulate=False, cpu_check=False):
    """
    Computes the bias gradient (biasgrad) from the given source featuremap 
    gradients (fmapsgrad).

    Let 
      n = number of images
      c = number of feature maps
      d = number of elements in each feature map (e.g. width * height)

    Then
      fmapsgrad must be (n) x (c*d)
      biasgrad  must be (c) x (1)

    The computation accumulates all elements of fmapsgrad stored in in featuremap i 
    into scalar stored in bias[i]:
      for i in range(c):
        biasgrad[i] = sum( fmapsgrad[:][i*d:(i+1)*d] )

    If biasgrad is None, a new array of the correct size will be created and returned.
    """

    if biasgrad is None:
        c = fmapsgrad.shape[1] // int(np.prod(dims))
        biasgrad = (zeros if accumulate else empty)((c, 1), fmapsgrad.dtype)

    cfg = c_featuremap_bias_cfg_t(int(np.prod(dims)), accumulate, cpu_check)
    cudnn_dll().api_featuremap_bias_grad(fmapsgrad._ptr, biasgrad._ptr, byref(cfg))

    return biasgrad


def conv2(src, src_w, src_h, filters, filter_w, filter_h, dst=None, bias=None, stride=1, accumulate=False, cpu_check=False):
    """
    Convolves a set of 2D filters (filters) across a mini-batch of 2D images (src),
    to generate a batch of 2D feature maps (dst). If a bias is given (bias != None),
    also adds a separate bias for each feature map.

    Let 
      n = number of src images
      c = number of channels per src image
      k = number of filters

    Then 
      src     must be (n) x (c*src_h*src_w)
      dst     must be (n) x (k*dst_h*dst_w)
      filters must be (k) x (c*filter_h*filter_w)
      bias    must be (k) x (1)

      where dst_w = (src_w-filter_w)//stride + 1
            dst_h = (src_h-filter_h)//stride + 1

    The memory layouts of each array are, in C-order notation,
      src[image][in_channel][pixel_y][pixel_x]
      dst[image][out_channel][pixel_y][pixel_x]
      filters[out_channel][in_channel][filter_y][filter_x]
      bias[out_channel]

    If dst is None, a new array of the correct size will be created and returned.

    If accumulate is True, the output will be added to the current value of dst.

    If cpu_check is True, then the result of conv2 will be compared to a
    simple CPU implementation to make sure absolute and relative error is
    below an internal certain threshold; used for unit tests.

    For the user's convenience, the "dst" instance will contain 
    attributes dst.w = dst_w and dst.h = dst_h so that the output
    size can be returned, rather than re-computed by the user.
    """

    dst_h = (src_h-filter_h)//stride + 1
    dst_w = (src_w-filter_w)//stride + 1
    if dst is None:
        k = len(filters);
        n = len(src);
        dst = (zeros if accumulate else empty)((n, k*dst_h*dst_w), src.dtype)

    cfg = c_conv2cfg_t(src_w, src_h, filter_w, filter_h, stride, accumulate, cpu_check)
    cudnn_dll().api_conv2(src._ptr, filters._ptr, dst._ptr, byref(cfg))

    # If bias was specified, add it to our final feature maps
    if bias is not None:
        featuremap_bias(dst, dst_w*dst_h, bias, cpu_check)
    
    # For convenience return the width and height
    setattr(dst, "w", dst_w)
    setattr(dst, "h", dst_h)
    return dst


def conv2_srcgrad(src_w, src_h, filters, filter_w, filter_h, dstgrad, srcgrad=None, stride=1, accumulate=False, cpu_check=False):
    """
    Computes gradient differentials at src (srcgrad) of a convolution 
    using gradient differentials given at dst (dstgrad). The shape and
    memory layout of each array correspond to conv2.

    If srcgrad is None, a new array of the correct size will be created and returned.

    If accumulate is True, the output will be added to the current value of srcgrad.
    """

    if srcgrad is None:
        c = filters.shape[1] // (filter_h*filter_w);
        n = len(dstgrad);
        srcgrad = (zeros if accumulate else empty)((n, c*src_h*src_w), dstgrad.dtype)

    cfg = c_conv2cfg_t(src_w, src_h, filter_w, filter_h, stride, accumulate, cpu_check)
    cudnn_dll().api_conv2_srcgrad(srcgrad._ptr, filters._ptr, dstgrad._ptr, byref(cfg))
    return srcgrad


def conv2_filtersgrad(src, src_w, src_h, filter_w, filter_h, dstgrad, filtersgrad=None, stride=1, accumulate=False, cpu_check=False):
    """
    Computes gradient differentials for filters (filtersgrad) of a convolution 
    using gradient differentials given at dst (dstgrad) and original inputs (src).
    The shape and memory layout of each array correspond to conv2.

    If filtersgrad is None, a new array of the correct size will be created and returned.

    If accumulate is True, the output will be added to the current value of filtersgrad.
    """

    if filtersgrad is None:
        dst_h = (src_h-filter_h)//stride+1
        dst_w = (src_w-filter_w)//stride+1
        k = dstgrad.shape[1] // (dst_h*dst_w);
        c = src.shape[1]     // (src_h*src_w);
        filtersgrad = (zeros if accumulate else empty)((k, c*filter_h*filter_w), src.dtype)

    cfg = c_conv2cfg_t(src_w, src_h, filter_w, filter_h, stride, accumulate, cpu_check)
    cudnn_dll().api_conv2_filtersgrad(src._ptr, filtersgrad._ptr, dstgrad._ptr, byref(cfg))
    return filtersgrad


def conv2_biasgrad(bias, dstgrad, biasgrad=None, accumulate=False, cpu_check=False):
    """
    Computes gradient differentials at bias (biasgrad) of a convolution 
    using gradient differentials given at dst (dstgrad). The shape and
    memory layout of each array correspond to conv2.

    If biasgrad is None, a new array of the correct size will be created and returned.

    If accumulate is True, the output will be added to the current value of biasgrad.
    """

    dims = dstgrad.shape[1] // bias.size
    return featuremap_bias_grad(dstgrad, dims, biasgrad, accumulate, cpu_check)


def pool2(mode, src, src_w, src_h, window_w, window_h, dst=None, stride=1, accumulate=False, cpu_check=False):
    """
    Pools a set of 2D regions across a batch of 2D images (src) 
    to generate a batch of 2D smaller feature maps (dst).

    Let 
      n = number of src images
      c = number of channels (feature maps) in src and dst

    Then 
      src must be (n) x (c*src_h*src_w)
      dst must be (n) x (c*dst_h*dst_w)

      where dst_w = (src_w-window_w)//stride + 1
            dst_h = (src_h-window_h)//stride + 1

    The memory layouts of each array are, in C-order notation,
      src[image][channel][pixel_y][pixel_x]
      dst[image][channel][pixel_y][pixel_x]

    If mode is "max", the pooling will take the maximum value over the region.
    If mode is "avg", the pooling will compute the average over the region.

    If dst is None, a new array of the correct size will be created and returned.

    If accumulate is True, the output will be added to the current value of dst.

    If cpu_check is True, then the result of pool2 will be compared to a
    simple CPU implementation to make sure absolute and relative error is
    below an internal certain threshold; used for unit tests.

    For the user's convenience, the "dst" instance will contain 
    attributes dst.w = dst_w and dst.h = dst_h so that the output
    size can be returned, rather than re-computed by the user.
    """

    if   mode == "max":   mode_int = 0
    elif mode == "avg":   mode_int = 1
    else: raise ValueError("Unrecognized mode '%s'" % mode)
    
    dst_h = (src_h-window_h)//stride + 1
    dst_w = (src_w-window_w)//stride + 1
    if dst is None:
        c = src.shape[1] // (src_w*src_h);
        n = len(src);
        dst = (zeros if accumulate else empty)((n, c*dst_h*dst_w), src.dtype)

    cfg = c_pool2cfg_t(mode_int, src_w, src_h, window_w, window_h, stride, accumulate, cpu_check)
    cudnn_dll().api_pool2(src._ptr, dst._ptr, byref(cfg))
    
    # For convenience return the width and height
    setattr(dst, "w", dst_w)
    setattr(dst, "h", dst_h)
    return dst



def pool2_grad(mode, src, src_w, src_h, window_w, window_h, dst, dstgrad, srcgrad=None, stride=1, accumulate=False, cpu_check=False):
    """
    Backpropagates a gradient through a pool2 operation. The original src and dst for the forward pool2 operation
    are needed, as well as the incoming gradients for the pooling outputs (dstgrad).
    The function sets the outgoing gradients for the pooling inputs (srcgrad).    

    Let 
      n = number of images in all arrays
      c = number of channels (feature maps) in all arrays

    Then 
      src and srcgrad must be (n) x (c*src_h*src_w)
      dst and dstgrad must be (n) x (c*dst_h*dst_w)

      where dst_w = (src_w-window_w)//stride + 1
            dst_h = (src_h-window_h)//stride + 1

    The memory layouts of each array are, in C-order notation,
      src    [image][channel][pixel_y][pixel_x]
      srcgrad[image][channel][pixel_y][pixel_x]
      dst    [image][channel][pixel_y][pixel_x]
      dstgrad[image][channel][pixel_y][pixel_x]

    If mode is "max", each dstgrad value will be backpropagated to (accumulated to) srcgrad
    at the position corresponding to the maximum element of its corresponding window in 'src'.
    
    If mode is "avg", each dstgrad value will be backpropagated to (accumulated to) srcgrad
    at all the positions in the corresponding window in 'src'.

    If srcgrad is None, a new array of the correct size will be created and returned.

    If accumulate is True, the output will be added to the current value of dstgrad.

    If cpu_check is True, then the result of pool2_grad will be compared to a
    simple CPU implementation to make sure absolute and relative error is
    below an internal certain threshold; used for unit tests.
    """

    if   mode == "max":   mode_int = 0
    elif mode == "avg":   mode_int = 1
    else: raise ValueError("Unrecognized mode '%s'" % mode)

    if srcgrad is None:
        srcgrad = (zeros_like if accumulate else empty_like)(src)

    cfg = c_pool2cfg_t(mode_int, src_w, src_h, window_w, window_h, stride, accumulate, cpu_check)
    cudnn_dll().api_pool2_grad(src._ptr, srcgrad._ptr, dst._ptr, dstgrad._ptr, byref(cfg))
    
    return srcgrad

