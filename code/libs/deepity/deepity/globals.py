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
import sys
import logging
import util
import copy

####################################

class global_flags(object):
    def __init__(self):
        self._flags = {}

    def copy_from(self, other):
        self._flags = copy.deepcopy(other._flags)

    def __contains__(self, name):
        if name in self._flags:
            return len(self._flags[name]) > 0
        return False

    def __getitem__(self, name):
        return self.get(name)

    def get(self, name, default_value=None):
        if name in self:
            return self._flags[name][-1]
        return default_value

    def push(self, name, value):
        if name in self._flags:
            self._flags[name].append(value)
        else:
            self._flags[name] = [value]

    def pop(self, name):
        assert name in self._flags
        val = self._flags[name].pop()
        if len(self._flags[name]) == 0:
            del self._flags[name]
        return val

flags = global_flags()

_allow_multiprocessing = True

def set_multiprocessing(enabled):
    global _allow_multiprocessing
    _allow_multiprocessing = enabled


def reset_backend(**kwargs):
    import smat
    smat.reset_backend(**kwargs)

def set_logging(file = None, level = None, echo = True, logger = None):
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []   # Clear any existing handlers

    if echo:
        # The stdout handler should only print what the user has requested
        shandler = logging.StreamHandler()
        shandler.setFormatter(logging.Formatter("%(message)s"))
        if   level == 0: shandler.setLevel(logging.ERROR)
        elif level == 1: shandler.setLevel(logging.INFO)
        elif level == 2: shandler.setLevel(logging.DEBUG)
        logger.addHandler(shandler)

    if file is not None:
        # The file handler should always write full debug logging
        util.makepath(os.path.dirname(file))
        fhandler = logging.FileHandler(file,'w')
        fhandler.setFormatter(logging.Formatter("%(message)s"))
        fhandler.setLevel(logging.DEBUG)
        logger.addHandler(fhandler)

