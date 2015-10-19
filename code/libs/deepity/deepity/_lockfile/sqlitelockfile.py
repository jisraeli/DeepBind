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
from __future__ import absolute_import, division

import time
import os

from . import LockBase, NotLocked, NotMyLock, LockTimeout, AlreadyLocked

class SQLiteLockFile(LockBase):
    "Demonstrate SQL-based locking."

    testdb = None

    def __init__(self, path, threaded=True):
        """
        >>> lock = SQLiteLockFile('somefile')
        >>> lock = SQLiteLockFile('somefile', threaded=False)
        """
        LockBase.__init__(self, path, threaded)
        self.lock_file = unicode(self.lock_file)
        self.unique_name = unicode(self.unique_name)

        if SQLiteLockFile.testdb is None:
            import tempfile
            _fd, testdb = tempfile.mkstemp()
            os.close(_fd)
            os.unlink(testdb)
            del _fd, tempfile
            SQLiteLockFile.testdb = testdb

        import sqlite3
        self.connection = sqlite3.connect(SQLiteLockFile.testdb)
        
        c = self.connection.cursor()
        try:
            c.execute("create table locks"
                      "("
                      "   lock_file varchar(32),"
                      "   unique_name varchar(32)"
                      ")")
        except sqlite3.OperationalError:
            pass
        else:
            self.connection.commit()
            import atexit
            atexit.register(os.unlink, SQLiteLockFile.testdb)

    def acquire(self, timeout=None):
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        if timeout is None:
            wait = 0.1
        elif timeout <= 0:
            wait = 0
        else:
            wait = timeout / 10

        cursor = self.connection.cursor()

        while True:
            if not self.is_locked():
                # Not locked.  Try to lock it.
                cursor.execute("insert into locks"
                               "  (lock_file, unique_name)"
                               "  values"
                               "  (?, ?)",
                               (self.lock_file, self.unique_name))
                self.connection.commit()

                # Check to see if we are the only lock holder.
                cursor.execute("select * from locks"
                               "  where unique_name = ?",
                               (self.unique_name,))
                rows = cursor.fetchall()
                if len(rows) > 1:
                    # Nope.  Someone else got there.  Remove our lock.
                    cursor.execute("delete from locks"
                                   "  where unique_name = ?",
                                   (self.unique_name,))
                    self.connection.commit()
                else:
                    # Yup.  We're done, so go home.
                    return
            else:
                # Check to see if we are the only lock holder.
                cursor.execute("select * from locks"
                               "  where unique_name = ?",
                               (self.unique_name,))
                rows = cursor.fetchall()
                if len(rows) == 1:
                    # We're the locker, so go home.
                    return
                    
            # Maybe we should wait a bit longer.
            if timeout is not None and time.time() > end_time:
                if timeout > 0:
                    # No more waiting.
                    raise LockTimeout
                else:
                    # Someone else has the lock and we are impatient..
                    raise AlreadyLocked

            # Well, okay.  We'll give it a bit longer.
            time.sleep(wait)

    def release(self):
        if not self.is_locked():
            raise NotLocked
        if not self.i_am_locking():
            raise NotMyLock((self._who_is_locking(), self.unique_name))
        cursor = self.connection.cursor()
        cursor.execute("delete from locks"
                       "  where unique_name = ?",
                       (self.unique_name,))
        self.connection.commit()

    def _who_is_locking(self):
        cursor = self.connection.cursor()
        cursor.execute("select unique_name from locks"
                       "  where lock_file = ?",
                       (self.lock_file,))
        return cursor.fetchone()[0]
        
    def is_locked(self):
        cursor = self.connection.cursor()
        cursor.execute("select * from locks"
                       "  where lock_file = ?",
                       (self.lock_file,))
        rows = cursor.fetchall()
        return not not rows

    def i_am_locking(self):
        cursor = self.connection.cursor()
        cursor.execute("select * from locks"
                       "  where lock_file = ?"
                       "    and unique_name = ?",
                       (self.lock_file, self.unique_name))
        return not not cursor.fetchall()

    def break_lock(self):
        cursor = self.connection.cursor()
        cursor.execute("delete from locks"
                       "  where lock_file = ?",
                       (self.lock_file,))
        self.connection.commit()
