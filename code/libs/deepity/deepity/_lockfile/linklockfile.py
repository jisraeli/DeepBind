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
from __future__ import absolute_import

import time
import os

from . import (LockBase, LockFailed, NotLocked, NotMyLock, LockTimeout,
               AlreadyLocked)

class LinkLockFile(LockBase):
    """Lock access to a file using atomic property of link(2).

    >>> lock = LinkLockFile('somefile')
    >>> lock = LinkLockFile('somefile', threaded=False)
    """

    def acquire(self, timeout=None):
        try:
            open(self.unique_name, "wb").close()
        except IOError:
            raise LockFailed("failed to create %s" % self.unique_name)

        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        while True:
            # Try and create a hard link to it.
            try:
                os.link(self.unique_name, self.lock_file)
            except OSError:
                # Link creation failed.  Maybe we've double-locked?
                nlinks = os.stat(self.unique_name).st_nlink
                if nlinks == 2:
                    # The original link plus the one I created == 2.  We're
                    # good to go.
                    return
                else:
                    # Otherwise the lock creation failed.
                    if timeout is not None and time.time() > end_time:
                        os.unlink(self.unique_name)
                        if timeout > 0:
                            raise LockTimeout
                        else:
                            raise AlreadyLocked
                    time.sleep(timeout is not None and timeout/10 or 0.1)
            else:
                # Link creation succeeded.  We're good to go.
                return

    def release(self):
        if not self.is_locked():
            raise NotLocked
        elif not os.path.exists(self.unique_name):
            raise NotMyLock
        os.unlink(self.unique_name)
        os.unlink(self.lock_file)

    def is_locked(self):
        return os.path.exists(self.lock_file)

    def i_am_locking(self):
        return (self.is_locked() and
                os.path.exists(self.unique_name) and
                os.stat(self.unique_name).st_nlink == 2)

    def break_lock(self):
        if os.path.exists(self.lock_file):
            os.unlink(self.lock_file)

