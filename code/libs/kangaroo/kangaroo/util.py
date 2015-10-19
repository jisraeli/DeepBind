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
import time
import os
import os.path

#################################################
# GATC string <-> 0123 array conversion

_ord2acgt = ['N']*256;    # lookup table for str.translate, so that 0123 => GATC
_ord2acgt[0] = 'A';
_ord2acgt[1] = 'C';
_ord2acgt[2] = 'G';
_ord2acgt[3] = 'T';
_ord2acgt = "".join(_ord2acgt)

_acgt2ord = ['\xff']*256;    # lookup table for str.translate, so that GATC => 0123
_acgt2ord[ord('a')] = _acgt2ord[ord('A')] = '\x00';
_acgt2ord[ord('c')] = _acgt2ord[ord('C')] = '\x01';
_acgt2ord[ord('g')] = _acgt2ord[ord('G')] = '\x02';
_acgt2ord[ord('t')] = _acgt2ord[ord('T')] = '\x03';
_acgt2ord[ord('u')] = _acgt2ord[ord('U')] = '\x03';
_acgt2ord = "".join(_acgt2ord)

_acgtcomplement = ['\xff']*256;    # lookup table for str.translate, so that GATC => CTAG
_acgtcomplement[ord('a')] = _acgtcomplement[ord('A')] = 'T';
_acgtcomplement[ord('c')] = _acgtcomplement[ord('C')] = 'G';
_acgtcomplement[ord('g')] = _acgtcomplement[ord('G')] = 'C';
_acgtcomplement[ord('t')] = _acgtcomplement[ord('T')] = 'A';
_acgtcomplement[ord('u')] = _acgtcomplement[ord('U')] = 'A';
_acgtcomplement[ord('n')] = _acgtcomplement[ord('N')] = 'N';
_acgtcomplement = "".join(_acgtcomplement)


def acgt2ord(s):
    """
    Convert an RNA string ("ACGT") into a numpy row-vector 
    of ordinals in range {0,1,2,3,255} where 255 indicates "padding".
    """
    x = s.translate(_acgt2ord)
    return np.ndarray(shape=(1,len(x)),buffer=x,dtype=np.uint8)


def ord2acgt(x):
    """
    Convert a vector of integral values in range {0,1,2,3} 
    to an RNA string ("ACGT"). Integers outside that range will
    be translated to a "padding" character (".").
    """
    s = str(np.asarray(x,dtype=np.uint8).data).translate(_ord2acgt)
    return s


def ord2mask(x):
    """
    Convert a vector of length N with integral values in range {0,1,2,3} 
    into an Nx4 numpy array, where for example "2" is represented by 
    row [0,0,1,0].
    """
    mask = np.zeros((x.size,4))
    mask[np.arange(x.size),x] = 1
    return mask


def acgt2mask(s):
    """
    Convert an RNA string ("ACGT") of length N into an Nx4 numpy 
    array, where for example "G" is represented by row [0,0,1,0].
    """
    return ord2mask(acgt2ord(s))


def acgtcomplement(s):
    """
    Complement a DNA string ("ACGT" to "TGCA").
    """
    return s.translate(_acgtcomplement)

def revcomp(s):
    """
    Reverse complement a DNA string ("ATTGC" to "GCAAT").
    """
    return s.translate(_acgtcomplement)[::-1]

##########################################

def str2intlist(arg):
    if arg is None:
        return None
    ints = []
    for part in arg.split(","):
        if '-' in part: id0,id1 = part.split('-')
        else:           id0 = id1 = part
        assert int(id0) >= 0 and int(id0) <= int(id1)
        ints.extend(range(int(id0), int(id1)+1))
    return sorted(set(ints))

def str2time(arg):
    if arg.endswith("s"): return float(arg.rstrip("s"))
    if arg.endswith("m"): return float(arg.rstrip("m"))*60.
    if arg.endswith("h"): return float(arg.rstrip("h"))*60.*60.
    if arg.endswith("d"): return float(arg.rstrip("d"))*60.*60.*24
    raise ValueError("Could not parse time argument \"%s\". Must end in 's','m','h', or 'd'." % arg)


def makepath(dir):
    if os.path.exists(dir):
        return dir
    retries = 8
    while retries >= 0:
        try:
            time.sleep(0.001)
            os.makedirs(dir)
            retries = -1
        except Exception, e:
            if retries == 0:
                raise
        retries -= 1
    return dir

