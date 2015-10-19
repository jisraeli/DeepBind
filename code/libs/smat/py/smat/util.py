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
import math,string
import numpy as np
import time
import __builtin__

#
# MATLAB-like tic/toc for convenience
#
_tics = {None: 0.0}

def tic(id=None):
    global _tics
    now = time.time()
    _tics[id] = now
    return now

def toc(id=None):
    global _tics
    now = time.time()
    return now - _tics[id]

##################################################

_int2dtype = { 0 : np.bool,
               1 : np.int8,
               2 : np.uint8,
               3 : np.int16,
               4 : np.uint16,
               5 : np.int32,
               6 : np.uint32,
               7 : np.int64,
               8 : np.uint64,
               9 : np.float32,
               10: np.float64 }

_dtype2int = { np.bool    : 0,
               np.int8    : 1,
               np.uint8   : 2,
               np.int16   : 3,
               np.uint16  : 4,
               np.int32   : 5,
               np.uint32  : 6,
               np.int64   : 7,
               np.uint64  : 8,
               np.float32 : 9,
               np.float64 : 10 }

_arg2dtype = { "bool"   : np.bool,    np.dtype('bool')   : np.bool,    np.bool   : np.bool,    __builtin__.bool : np.bool,
               "int8"   : np.int8,    np.dtype('int8')   : np.int8,    np.int8   : np.int8,    __builtin__.chr  : np.int8,
               "uint8"  : np.uint8,   np.dtype('uint8')  : np.uint8,   np.uint8  : np.uint8,
               "int16"  : np.int16,   np.dtype('int16')  : np.int16,   np.int16  : np.int16,
               "uint16" : np.uint16,  np.dtype('uint16') : np.uint16,  np.uint16 : np.uint16,
               "int32"  : np.int32,   np.dtype('int32')  : np.int32,   np.int32  : np.int32,   __builtin__.int  : np.int32,
               "uint32" : np.uint32,  np.dtype('uint32') : np.uint32,  np.uint32 : np.uint32,
               "int64"  : np.int64,   np.dtype('int64')  : np.int64,   np.int64  : np.int64,   __builtin__.long : np.int64,
               "uint64" : np.uint64,  np.dtype('uint64') : np.uint64,  np.uint64 : np.uint64,
               "float32": np.float32, np.dtype('float32'): np.float32, np.float32: np.float32, 
               "float64": np.float64, np.dtype('float64'): np.float64, np.float64: np.float64, __builtin__.float: np.float64 }


# copied from http://code.activestate.com/recipes/578323-human-readable-filememory-sizes-v2/

def format_bytecount(val,fmt=".2cM"):
    """ define a size class to allow custom formatting
        format specifiers supported : 
            em : formats the size as bits in IEC format i.e. 1024 bits (128 bytes) = 1Kib 
            eM : formats the size as Bytes in IEC format i.e. 1024 bytes = 1KiB
            sm : formats the size as bits in SI format i.e. 1000 bits = 1kb
            sM : formats the size as bytes in SI format i.e. 1000 bytes = 1KB
            cm : format the size as bit in the common format i.e. 1024 bits (128 bytes) = 1Kb
            cM : format the size as bytes in the common format i.e. 1024 bytes = 1KB
    """
    if val == 0:
        return "0"

    # work out the scale, suffix and base        
    factor, suffix = (8, "b") if fmt[-1] in string.lowercase else (1,"B")
    base = 1024 if fmt[-2] in ["e","c"] else 1000

    # Add the i for the IEC format
    suffix = "i"+ suffix if fmt[-2] == "e" else suffix

    mult = ["","K","M","G","T","P"]

    val = float(val) * factor
    i = 0 if val < 1 else int(math.log(val, base))+1
    v = val / math.pow(base,i)
    v,i = (v,i) if v > 0.5 else (v*base,i-1)

    # Identify if there is a width and extract it
    width = "" if fmt.find(".") == -1 else fmt[:fmt.index(".")]        
    precis = fmt[:-2] if width == "" else fmt[fmt.index("."):-2]

    # do the precision bit first, so width/alignment works with the suffix
    t = ("{0:{1}f}"+mult[i]+suffix).format(v, precis) 

    return "{0:{1}}".format(t,width) if width != "" else t

