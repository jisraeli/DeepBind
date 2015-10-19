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
import node  # circular import ref ok because we only access node module inside functions

##########################################################################
# Plugs take value 'null' when they are uninitialized, rather than None, 
# so that there is no confusion in cases where a node sets an output plug to None
class plug_null(object):
    def __nonzero__(self):
        return False  # Behave like None during tests


class plug(object):
    """
    A plug on a dependency graph node. See iplug and oplug.
    """

    def __init__(self,node,name=None,shape=None):
        self._shape = shape
        self.name   = name
        self.node   = node        # Node this plug belongs to. 
        self.srcs   = []          # Upstream plugs
        self.dsts   = []          # Downstream plugs
        self._fpval = plug_null   # Last value forward-propagated through this plug.
        self._bpval = plug_null   # Last value backward-propagated through this plug.
        self._shape = None       # Shape constraints on this plug's potential values
        self.inst_axis = 1        # Which axis of _shape grows when there are multiple 'instances' on the plug; by default, it's columns
        self.trainable = True
        
    def has_src(self): return len(self.srcs) > 0
    def has_dst(self): return len(self.dsts) > 0

    def has_upstream(self):
        """Returns whether there's an upstream plug with a computed value."""
        if len(self.srcs) == 0:
            return self.is_oplug()
        for src in self.srcs:
            if src.has_upstream():
                return True
        return False

    @property
    def path(self):
        return self.node.path + "."+self.name

    def rename(self,newname):
        del self.node.__dict__[self.name]
        self.name = newname
        self.node.__dict__[newname] = self

    def origin(self):
        """A supernode plug's origin is its corresponding 'real' plug (on a non-supernode)"""
        # If we're attached to an actual node, then return ourselves
        if not isinstance(self.node,node.supernode):
            return self

        # If we're an iplug on a supernode, ask our dst plug for its origin
        if self.is_iplug():
            return self.dsts[0].origin()

        # Otherwise, we're an oplug on a supernode, so ask our src plug for it's origin
        return self.srcs[0].origin()

    @property
    def shape(self):
        """
        The shape of a plug's value, as a tuple (nrow,ncol), even if no plug value
        has been assigned yet. If nrow or ncol is None, then that dimension's size
        is undefined (e.g. nrow = None when nrow varies with minibatch size).
        """
        if self._shape is None:
            self._calc_shape([])  # Try to calculate shape
        return self._shape

    @shape.setter
    def shape(self,shape):
        self._shape = shape

    @property
    def fpval(self):
        """Pull _fpval from all src plugs."""
        if self._fpval is plug_null:
            if self.has_src():
                for src in self.srcs:
                    if self._fpval is plug_null: self._fpval = src.fpval                # 1. First time, get a reference (no copy)
                    elif src is self.srcs[1]:    self._fpval = self._fpval + src.fpval  # 2. Second time, make a new sum
                    else:                        self._fpval += src.fpval               # 3. Third+ time, add directly to new value
            elif self.is_oplug():
                self.node.fprop()
            else:
                #raise AssertionError("unreachable code")
                self._fpval = None
        return self._fpval

    @property
    def bpval(self):
        """Pull _bpval from all dst plugs."""
        if self._bpval is plug_null:
            if self.has_dst():
                for dst in self.dsts:
                    if self._bpval is plug_null: self._bpval = dst.bpval                # 1. First time, get a reference (no copy)
                    elif dst is self.dsts[1]:    self._bpval = self._bpval + dst.bpval  # 2. Second time, make a new sum
                    else:                        self._bpval += dst.bpval               # 3. Third+ time, add directly to new value
            elif self.is_iplug():
                self.node.bprop()
            else:
                raise AssertionError("unreachable code")
        return self._bpval

    @fpval.setter
    def fpval(self,value):
        """Directly set the value forward propagated by this input plug (e.g. int, float, numpy array, etc.)"""
        self._fpval = value;
        self._check_shape()

    @bpval.setter
    def bpval(self,value):
        """Directly set the value backward propagated by this output plug"""
        assert not self.has_dst(), "Cannot directly set bpval on plug with downstream connection."
        assert self._bpval is plug_null, "Must call node.clear() before setting new value on a plug."
        self._bpval = value;
        self._check_shape()

    def _check_shape(self):
        if self.shape and hasattr(self._fpval,"shape"):
            assert len(self._fpval.shape) == len(self.shape), "Value dimension does not match plug dimension."
            for ss,vs in zip(self.shape,self._fpval.shape):
                if ss:   # If shape[i] is None, don't enforce any shape
                    assert vs >= ss , "Value shape is too small for plug shape."
                    assert vs % ss == 0 , "Value shape does not broadcast to plug shape."

    def _calc_shape(self,visited):
        if self._shape is not None:
            return True

        # Don't backtrack.
        if self in visited:
            return False

        # If any of our immediate srcs/dsts have a well-defined shape, then
        # our shape must match theirs.
        for p in self.srcs+self.dsts:
            if p._calc_shape(visited+[self]):
                self._shape = p._shape
                assert self.inst_axis == p.inst_axis
                #print "%s._calc_shape() succeeded: %s" % (self.path,str(self._shape))
                return True

        # Otherwise, ask the node to propagate shapes and see if it managed
        # to determine a shape for us.
        self.node.calc_shapes(visited+[self])
        if self._shape:
            return True

    def _add_src(self,src):
        self.srcs.append(src)
        self._validate()

    def _add_dst(self,dst):
        self.dsts.append(dst)
        self._validate()

    def is_iplug(self): return self in self.node.iplugs
    def is_oplug(self): return self in self.node.oplugs

    def _validate(self):
        for p in self.srcs+self.dsts:
            assert isinstance(p,plug), "Src/dst must be of type plug, not \"%s\"" % type(p)
        if isinstance(self.node,node.supernode):
            if self.is_iplug(): assert len(self.dsts) <= 1, "Supernode iplug can only have one dst"
            else:               assert len(self.srcs) <= 1, "Supernode oplug can only have one src"
        else:
            if self.is_iplug(): assert len(self.dsts) == 0, "Node iplug cannot have dst"
            else:               assert len(self.srcs) == 0, "Node oplug cannot have src"

    # Functions to allow easier connect(X,Y) syntax.
    # Allows things like X >> Y,  Y << X where Y is a plug
    # and X is a plug or a literal.
    def __lshift__(self,other):  connect(other,self); return self
    def __rshift__(self,other):  connect(self,other); return other
    def __rlshift__(self,other): connect(self,other); return other
    def __rrshift__(self,other): connect(other,self); return self

##########################################################################

def connect(src,dst):
    """
    Connects src to dst.
       - src can be an iplug, a node, or a value (float, ndarray, sarray)
       - dst can be an oplug, or a node
    If src is a node, its first output plug is used.
    If dst is a node, its first input plug is used.
    
    Once connected, any change to src's value will be propagated 
    to dst and thereby to all dst's downstream dependents.

    Note that operators "src >> dst" and "dst << src" are equivalent
    to calling this function.
    """
    # If src/dst is a node, then assume the user wants to connect 
    # the first src.oplugs[0] to dst.iplugs[0] as convenient shorthand
    if isinstance(src,node.node): src = src.oplugs[0]
    if isinstance(dst,node.node): dst = dst.iplugs[0]
    
    # Connect the two plugs.
    src._add_dst(dst)         
    dst._add_src(src)

def disconnect(src,dst):
    if isinstance(src,node.node): src = src.oplugs[0]
    if isinstance(dst,node.node): dst = dst.iplugs[0]
    src.dsts.remove(dst)
    dst.srcs.remove(src)