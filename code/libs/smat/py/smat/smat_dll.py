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
import os,platform,warnings,sys
from os.path import abspath,join,dirname
from ctypes import *

dllconfig = "release"
#dllconfig = "debug"

__all__ = ["dll","safe_dll","load_cdll","cudnn_dll",
           "load_extension", "unload_extension",
           "c_shape_t","c_index_t","c_isize_t","c_slice_t","c_axis_t","c_slice_end",
           "c_dtype_t","c_char_p_p",
           "c_heap_status","c_backend_info",
           "c_smat_p",
           "c_conv2cfg_t","c_conv2cfg_p",
           "c_pool2cfg_t","c_pool2cfg_p",
           "c_featuremap_bias_cfg_t","c_featuremap_bias_cfg_p",
           ]

###################################################################
# Declare some useful ctypes based on the C++ types

c_char_p_p = POINTER(c_char_p)
c_index_t  = c_int
c_uindex_t = c_uint
c_isize_t  = c_int
c_usize_t  = c_uint
c_dtype_t  = c_int
c_vtype_t  = c_int
c_axis_t   = c_int

class c_shape_t(Structure):
    _fields_ = [("x", c_isize_t),
                ("y", c_isize_t),
                ("z", c_isize_t)]
c_shape_p = POINTER(c_shape_t)

class c_coord_t(Structure):
    _fields_ = [("x", c_index_t),
                ("y", c_index_t),
                ("z", c_index_t)]
c_coord_p = POINTER(c_coord_t)
c_strides_t = c_coord_t
c_strides_p = c_coord_p

class c_slice_t(Structure):
    _fields_ = [("first", c_index_t),
                ("last", c_index_t)]
c_slice_p = POINTER(c_slice_t)
c_slice_end = c_isize_t(0x7f000000)  # this constant from types.h

class c_smat(Structure):
    _fields_ = [("vtype", c_vtype_t),
                ("dtype", c_dtype_t),
                ("shape", c_shape_t),
                ("strides", c_strides_t),
                ]  # NOT a complete list of members within an smat object
                   # -- do NOT allocate c_smat instances directly in Python!
c_smat_p = POINTER(c_smat)

class c_heap_status(Structure):
    _fields_ = [("host_total", c_size_t),
                ("host_avail", c_size_t),
                ("host_used" , c_size_t),
                ("device_total", c_size_t),
                ("device_avail", c_size_t),
                ("device_used", c_size_t),
                ("device_committed", c_size_t)]
c_heap_status_p = POINTER(c_heap_status)

class c_backend_info(Structure):
    _fields_ = [("uuid", c_int),
                ("name", c_char*32),
                ("version", c_char*32),
                ("device", c_char*128)]
c_backend_info_p = POINTER(c_backend_info)

###################################################################

###################################################################
# Load smat.dll

def load_cdll(dllname,search_dirs=None):
    module_dir  = dirname(abspath(__file__))                      # .../smat/py/smat
    parent_dir  = dirname(dirname(dirname(module_dir)))           # .../smat
    devbin_dir  = join(parent_dir,"smat","build",dllconfig,"bin") # .../smat/build/{release|debug}/bin (C++ development smat, standalone)
    instbin_dir = join(module_dir,"bin")                          # site-packages/smat/bin            (C++ installed smat, as python package)
    if search_dirs is None:
        search_dirs = []
    search_dirs += [devbin_dir,instbin_dir]

    # First determine the platform-specific file name of the dll
    dllfiles = { "Windows": "%s.dll"%dllname,
                 "Linux"  : "lib%s.so"%dllname,
                 "Unix"   : "lib%s.so"%dllname }
    dllfile = dllfiles.get(platform.system(),None)
    if dllfile is None:
        raise NotImplementedError("Platform not yet supported by smat")

    # Then try to find it in one of the standard search paths.
    for search_dir in search_dirs:
        dllpath = join(search_dir,dllfile)                             # deepity/build/{release|debug}/bin/smat.dll
        try:
            os.environ["PATH"] += os.pathsep + search_dir
            if os.environ.get("LD_LIBRARY_PATH","") == "":
                os.environ["LD_LIBRARY_PATH"] = search_dir
            else:
                os.environ["LD_LIBRARY_PATH"] += os.pathsep + search_dir
            dll = cdll.LoadLibrary(dllpath)
            _smat_load_err = None
            break
        except OSError as err:
            _smat_load_err = err

    if _smat_load_err is not None:
        print "**** Failed to load %s from:" % dllfile
        for search_dir in search_dirs:
            print '  ',search_dir
        raise _smat_load_err

    return dll

class SmatException(Exception):
    pass

# safe_dll_func
#   Each insance of safe_dll_func wraps a callable function 
#   on a ctypes.CDLL or WinDLL object.
#   The difference is that safe_dll_func will catch exceptions 
#   and call _get_last_error to retrieve an error message from 
#   the DLL before raising the exception further.
#
class safe_dll_func(object):
    def __init__(self,name,func,get_last_error,clear_last_error):
        self._name = name
        self._func = func
        self._get_last_error = get_last_error
        self._clear_last_error = clear_last_error

    def __call__(self,*args):
        rval = self._func(*args)

        # If the dll threw an exception, _get_last_error() should get the error message
        msg = self._get_last_error()
        if msg is None:
            return rval

        self._clear_last_error()
        msg = "%s(...) raised an exception\n%s" % (str(self._name.replace("api_","")), msg)
        raise SmatException(msg)

    def declare(self,restype,argtypes):
        self._func.restype  = restype
        self._func.argtypes = argtypes

    def __repr__(self): return "smat_dll.%s(...)" % self._name

# safe_dll
#   Simply wraps a ctypes.CDLL so that function calls to that DLL
#   are all called through safe_dll_func objects.
#
class safe_dll(object):
    def __init__(self,dll,get_last_error,clear_last_error):
        self._dll = dll
        self._get_last_error = get_last_error
        self._clear_last_error = clear_last_error
        self._funcs = {}

    def __getattr__(self,name):
        if not self._funcs.has_key(name):
            func = safe_dll_func(name,self._dll.__getattr__(name),self._get_last_error,self._clear_last_error)
            self._funcs[name] = func
            return func
        return self._funcs[name]


def load_extension(dllname,search_dirs=None):
    dll.api_sync()
    handle = dll.api_load_extension(dllname)
    ext_dll = CDLL(dllname,handle=handle)
    ext_dll = safe_dll(ext_dll,dll.api_get_last_error,dll.api_clear_last_error)
    return ext_dll

def unload_extension(ext_dll):
    dll.api_sync()
    dll.api_unload_extension(ext_dll._dll._handle)
    del ext_dll._dll


# Now create the public 'dll' object exposed to smat.py, with all the methods
# exported by the DLL available for calling
#
smat_cdll = load_cdll('smat')
smat_cdll.api_get_last_error.restype = c_char_p
dll = safe_dll(smat_cdll,smat_cdll.api_get_last_error,smat_cdll.api_clear_last_error)

###################################################################
# Configure function prototypes exported from smat.dll

# dtypes.cpp
dll.api_set_default_dtype.declare(  None,     [c_dtype_t])
dll.api_set_default_dtypef.declare( None,     [c_dtype_t])
dll.api_get_default_dtype.declare(  c_dtype_t,[])
dll.api_get_default_dtypef.declare( c_dtype_t,[])
dll.api_dtype_size.declare(         c_int,    [c_dtype_t])

# context.cpp
dll.api_set_backend.declare(      c_bool, [c_char_p,c_int,c_char_p_p])
dll.api_set_backend_options.declare(None, [c_int,c_char_p_p])
dll.api_get_backend_info.declare(   None, [c_backend_info_p])
dll.api_reset_backend.declare(      None, [c_int,c_char_p_p])
dll.api_destroy_backend.declare(    None, [c_bool])
dll.api_get_heap_status.declare(    None, [c_heap_status_p])
dll.api_is_dtype_supported.declare(c_bool,[c_dtype_t])
dll.api_load_extension.declare( c_size_t, [c_char_p])
dll.api_unload_extension.declare(   None, [c_size_t])
dll.api_set_rand_seed.declare(      None, [c_size_t])

# smat.cpp
dll.api_get_last_error.declare(c_char_p, [])
dll.api_clear_last_error.declare(None,   [])
dll.api_set_debug_break.declare(None,    [c_bool])

dll.api_empty_like.declare(c_smat_p,   [c_smat_p,c_dtype_t])
dll.api_zeros_like.declare(c_smat_p,   [c_smat_p,c_dtype_t])
dll.api_ones_like.declare( c_smat_p,   [c_smat_p,c_dtype_t])
dll.api_empty.declare(     c_smat_p,   [c_shape_p,c_dtype_t])
dll.api_zeros.declare(     c_smat_p,   [c_shape_p,c_dtype_t])
dll.api_ones.declare(      c_smat_p,   [c_shape_p,c_dtype_t])
dll.api_eye.declare(       c_smat_p,   [c_isize_t,c_dtype_t])
dll.api_arange.declare(    c_smat_p,   [c_index_t,c_index_t,c_dtype_t])
dll.api_rand.declare(      c_smat_p,   [c_shape_p,c_dtype_t])
dll.api_randn.declare(     c_smat_p,   [c_shape_p,c_dtype_t])
dll.api_bernoulli.declare( c_smat_p,   [c_shape_p,c_float,c_dtype_t])
dll.api_const_b8.declare(  c_smat_p,   [c_bool])
dll.api_const_i8.declare(  c_smat_p,   [c_byte])
dll.api_const_u8.declare(  c_smat_p,   [c_ubyte])
dll.api_const_i16.declare( c_smat_p,   [c_short])
dll.api_const_u16.declare( c_smat_p,   [c_ushort])
dll.api_const_i32.declare( c_smat_p,   [c_int])
dll.api_const_u32.declare( c_smat_p,   [c_uint])
dll.api_const_i64.declare( c_smat_p,   [c_longlong])
dll.api_const_u64.declare( c_smat_p,   [c_ulonglong])
dll.api_const_f32.declare( c_smat_p,   [c_float])
dll.api_const_f64.declare( c_smat_p,   [c_double])

dll.api_delete.declare(   None,       [c_smat_p])
dll.api_nrow.declare(     c_isize_t,  [c_smat_p])
dll.api_ncol.declare(     c_isize_t,  [c_smat_p])
dll.api_size.declare(     c_size_t,   [c_smat_p])
dll.api_shape.declare(    None,       [c_smat_p,c_shape_p])
dll.api_reshape.declare(  c_smat_p,   [c_smat_p,c_shape_p])
dll.api_dtype.declare(    c_int,      [c_smat_p])
dll.api_slice.declare(    c_smat_p,   [c_smat_p,c_slice_p,c_slice_p])
dll.api_assign.declare(   None,       [c_smat_p,c_smat_p])
dll.api_copy_from.declare(None,       [c_smat_p,c_void_p,c_isize_t,c_isize_t])
dll.api_copy_to.declare(  None,       [c_smat_p,c_void_p,c_isize_t,c_isize_t])
dll.api_sync.declare(     None,       [])

dll.api_add.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_sub.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_mul.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_div.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_mod.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_pow.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_iadd.declare(     None,       [c_smat_p,c_smat_p])
dll.api_isub.declare(     None,       [c_smat_p,c_smat_p])
dll.api_imul.declare(     None,       [c_smat_p,c_smat_p])
dll.api_idiv.declare(     None,       [c_smat_p,c_smat_p])
dll.api_imod.declare(     None,       [c_smat_p,c_smat_p])
dll.api_ipow.declare(     None,       [c_smat_p,c_smat_p])
dll.api_dot.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_dot_tn.declare(   c_smat_p,   [c_smat_p,c_smat_p])
dll.api_dot_nt.declare(   c_smat_p,   [c_smat_p,c_smat_p])
dll.api_dot_tt.declare(   c_smat_p,   [c_smat_p,c_smat_p])
dll.api_dot_out.declare(      None,   [c_smat_p,c_smat_p,c_smat_p])
dll.api_dot_tn_out.declare(   None,   [c_smat_p,c_smat_p,c_smat_p])
dll.api_dot_nt_out.declare(   None,   [c_smat_p,c_smat_p,c_smat_p])
dll.api_dot_tt_out.declare(   None,   [c_smat_p,c_smat_p,c_smat_p])
dll.api_eq.declare(       c_smat_p,   [c_smat_p,c_smat_p])
dll.api_ne.declare(       c_smat_p,   [c_smat_p,c_smat_p])
dll.api_lt.declare(       c_smat_p,   [c_smat_p,c_smat_p])
dll.api_le.declare(       c_smat_p,   [c_smat_p,c_smat_p])
dll.api_gt.declare(       c_smat_p,   [c_smat_p,c_smat_p])
dll.api_ge.declare(       c_smat_p,   [c_smat_p,c_smat_p])
dll.api_not.declare(      c_smat_p,   [c_smat_p])
dll.api_or.declare(       c_smat_p,   [c_smat_p,c_smat_p])
dll.api_xor.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_and.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_lnot.declare(     c_smat_p,   [c_smat_p])
dll.api_lor.declare(      c_smat_p,   [c_smat_p,c_smat_p])
dll.api_land.declare(     c_smat_p,   [c_smat_p,c_smat_p])
dll.api_ior.declare(      None,       [c_smat_p,c_smat_p])
dll.api_ixor.declare(     None,       [c_smat_p,c_smat_p])
dll.api_iand.declare(     None,       [c_smat_p,c_smat_p])
dll.api_neg.declare(      c_smat_p,   [c_smat_p])
dll.api_abs.declare(      c_smat_p,   [c_smat_p])
dll.api_sign.declare(     c_smat_p,   [c_smat_p])
dll.api_signbit.declare(  c_smat_p,   [c_smat_p])
dll.api_sin.declare(      c_smat_p,   [c_smat_p])
dll.api_cos.declare(      c_smat_p,   [c_smat_p])
dll.api_tan.declare(      c_smat_p,   [c_smat_p])
dll.api_arcsin.declare(   c_smat_p,   [c_smat_p])
dll.api_arccos.declare(   c_smat_p,   [c_smat_p])
dll.api_arctan.declare(   c_smat_p,   [c_smat_p])
dll.api_sinh.declare(     c_smat_p,   [c_smat_p])
dll.api_cosh.declare(     c_smat_p,   [c_smat_p])
dll.api_tanh.declare(     c_smat_p,   [c_smat_p])
dll.api_arcsinh.declare(  c_smat_p,   [c_smat_p])
dll.api_arccosh.declare(  c_smat_p,   [c_smat_p])
dll.api_arctanh.declare(  c_smat_p,   [c_smat_p])
dll.api_exp.declare(      c_smat_p,   [c_smat_p])
dll.api_exp2.declare(     c_smat_p,   [c_smat_p])
dll.api_log.declare(      c_smat_p,   [c_smat_p])
dll.api_log2.declare(     c_smat_p,   [c_smat_p])
dll.api_logistic.declare( c_smat_p,   [c_smat_p])
dll.api_sqrt.declare(     c_smat_p,   [c_smat_p])
dll.api_square.declare(   c_smat_p,   [c_smat_p])
dll.api_round.declare(    c_smat_p,   [c_smat_p])
dll.api_floor.declare(    c_smat_p,   [c_smat_p])
dll.api_ceil.declare(     c_smat_p,   [c_smat_p])
dll.api_clip.declare(     c_smat_p,   [c_smat_p,c_double,c_double])
dll.api_isinf.declare(    c_smat_p,   [c_smat_p])
dll.api_isnan.declare(    c_smat_p,   [c_smat_p])
dll.api_isclose.declare(  c_smat_p,   [c_smat_p,c_smat_p,c_double,c_double])
dll.api_allclose.declare( c_smat_p,   [c_smat_p,c_smat_p,c_double,c_double])
dll.api_maximum.declare(  c_smat_p,   [c_smat_p,c_smat_p])
dll.api_minimum.declare(  c_smat_p,   [c_smat_p,c_smat_p])
dll.api_max.declare(      c_smat_p,   [c_smat_p,c_axis_t])
dll.api_min.declare(      c_smat_p,   [c_smat_p,c_axis_t])
dll.api_sum.declare(      c_smat_p,   [c_smat_p,c_axis_t])
dll.api_mean.declare(     c_smat_p,   [c_smat_p,c_axis_t])
dll.api_nnz.declare(      c_smat_p,   [c_smat_p,c_axis_t])
dll.api_all.declare(      c_smat_p,   [c_smat_p,c_axis_t])
dll.api_any.declare(      c_smat_p,   [c_smat_p,c_axis_t])
dll.api_diff.declare(     c_smat_p,   [c_smat_p,c_axis_t])
dll.api_repeat.declare(   c_smat_p,   [c_smat_p,c_shape_p])
dll.api_tile.declare(     c_smat_p,   [c_smat_p,c_shape_p])
dll.api_trace.declare(    c_smat_p,   [c_smat_p])
dll.api_trans.declare(    c_smat_p,   [c_smat_p])

dll.api_softmax.declare(  c_smat_p,   [c_smat_p,c_axis_t])
dll.api_apply_mask.declare(None,      [c_smat_p,c_smat_p])

dll.api_dropout_fp_tr.declare( None, [c_smat_p,c_double,c_smat_p,c_smat_p])
dll.api_dropout_bp_tr.declare( None, [c_smat_p,c_smat_p,c_smat_p])

####################################################################

dll.api_set_debug_break(False)   # disable debug break events so that, within an integrated debugger,
                                 # the error of interest can immediately propagate up to Python rather
                                 # than stopping the debugger at the C++ breakpoint.



###################################################################
#          CUDNN extension
###################################################################
# Declare some useful ctypes based on the C++ types


class c_conv2cfg_t(Structure):
    _fields_ = [("src_w",    c_int), ("src_h",    c_int),
                ("filter_w", c_int), ("filter_h", c_int),
                ("stride",   c_int),
                ("accumulate", c_int),
                ("cpu_check", c_int),]
c_conv2cfg_p = POINTER(c_conv2cfg_t)

class c_featuremap_bias_cfg_t(Structure):
    _fields_ = [("dims",       c_int),
                ("accumulate", c_int),
                ("cpu_check",  c_int),]
c_featuremap_bias_cfg_p = POINTER(c_featuremap_bias_cfg_t)

class c_pool2cfg_t(Structure):
    _fields_ = [("mode", c_int),
                ("src_w",    c_int), ("src_h",    c_int),
                ("window_w", c_int), ("window_h", c_int),
                ("stride",   c_int),
                ("accumulate", c_int),
                ("cpu_check", c_int),]
c_pool2cfg_p = POINTER(c_pool2cfg_t)


_cudnn_dll = None
def cudnn_dll():
    """Get handle to smat_cudnn.dll, loading it if necessary."""
    global _cudnn_dll
    if _cudnn_dll is None:
        path_sep = ";" if platform.system()=="Windows" else ":"
        path_vars = os.environ['PATH'].split(path_sep)
        cudnn_path = os.environ.get('CUDNN_PATH')
        if cudnn_path and cudnn_path not in path_vars:
            path_vars += [cudnn_path]
            os.environ['PATH'] = path_sep.join(path_vars)
        _cudnn_dll = load_extension("smat_cudnn")
        _cudnn_dll.api_conv2.declare(            None, [c_smat_p, c_smat_p, c_smat_p, c_conv2cfg_p])
        _cudnn_dll.api_conv2_srcgrad.declare(    None, [c_smat_p, c_smat_p, c_smat_p, c_conv2cfg_p])
        _cudnn_dll.api_conv2_filtersgrad.declare(None, [c_smat_p, c_smat_p, c_smat_p, c_conv2cfg_p])
        _cudnn_dll.api_featuremap_bias.declare(        None, [c_smat_p, c_smat_p, c_featuremap_bias_cfg_p])
        _cudnn_dll.api_featuremap_bias_grad.declare(   None, [c_smat_p, c_smat_p, c_featuremap_bias_cfg_p])
        _cudnn_dll.api_pool2.declare(         None, [c_smat_p, c_smat_p, c_pool2cfg_p])
        _cudnn_dll.api_pool2_grad.declare(    None, [c_smat_p, c_smat_p, c_smat_p, c_smat_p, c_pool2cfg_p])
    return _cudnn_dll


