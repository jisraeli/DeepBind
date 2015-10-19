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
import os
import time
import tempfile

#################################################

def splitlist(x,n):
    m = len(x)
    return [x[i:min(m,i+m//n)] for i in range(0,m,m//n)]

#################################################

def makepath(dir):
    """
    Makes a complete path if it does not exist already. 
    Does not remove any existing files.
    Fixes periodic failure of os.makedirs on Windows.
    """
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
        else:
            retries -= 1
    return dir

#################################################

def hashed_filename(filename,**kwargs):
    # The cache is a temporary file that is hashed based on args+variant, 
    # i.e. if we only want cols 1,2,3 for example, that will go into a different
    # temporary file than if we were asked for cols 2,3,4.
    filebase = os.path.basename(os.path.splitext(filename)[0])
    args_hash = abs(hash(";".join(["%s=%s" % (key,val) for key,val in kwargs.iteritems()])))
    cachepath = os.path.join(tempfile.gettempdir(),"%s.hash%s" % (filebase,args_hash))
    return cachepath

#################################################
# MATLAB-like tic/toc

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


