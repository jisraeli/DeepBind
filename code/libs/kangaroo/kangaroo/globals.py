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
import time
import logging

import os
import os.path
import copy
import time
import logging
import deepity

_logfile = None
_randseed = None
_devices = [0]
_allow_multiprocessing = True

flags = deepity.globals.flags

def set_devices(devices):
    """List of device IDs that can be used by worker processes."""
    global _devices
    _devices = devices


def set_randseed(seed):
    """Random seed used each time a training session begins."""
    global _randseed
    _randseed = seed


def set_multiprocessing(enabled):
    global _allow_multiprocessing
    _allow_multiprocessing = enabled
    deepity.globals.set_multiprocessing(enabled)


def set_logging(outdir):
    global _logfile
    _logfile = os.path.join(outdir, "kangaroo.log")
    deepity.set_logging(_logfile, level=1)
    logging.debug("\n----------------------------- %s -----------------------------" % time.strftime("%y-%m-%d %H-%M-%S",time.localtime()))


def _set_default_logging(outdir):
    global _logfile
    if _logfile is None:
        set_logging(outdir)
