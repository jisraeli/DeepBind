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
from .plug import plug,connect,plug_null
from .data import datasource
from . import globals
import re
import smat as sm
import numpy as np
try:
    from hpsearch import paramdef
except:
    class dummy_type(object): pass
    paramdef = dummy_type

class node(object):
    """
    A dependency graph node. 
    Each node must provide two functions:

        _fprop: compute the value of each output plug, using the input plugs
        _bprop: compute the delta of each input  plug, using the input plugs and delta of each output plug

    For example, a "Z = tanh(X)" node would be implemented as:

       class tanh_node(node):
           def __init__(self):    node.__init__(["X"],["Z"])   # Create iplug X and oplug Z
           def _fprop(self,X):    return tanh(X)               # Compute Z from X
           def _bprop(self,Z,dZ): return dZ*(1-Z**2)           # Compute dX from Z and dZ
                                                               # (Note: tanh'(X) = 1-tanh(X)^2 = 1-Z^2)
    """

    def __init__(self,iplugs,oplugs,name=None):
        self.name = name
        self.ninst    = 1
        self._parent  = None    # Weakref to parent (the @parent property automatically derefs)

        # Add plugs directly specified by subclass
        self.iplugs = [plug(self,pname) for pname in iplugs]
        self.oplugs = [plug(self,pname) for pname in oplugs]
        
        # For convenience, add each plug as a named attribute, to facilitate "self.X"
        self.__dict__.update({ plug.name : plug for plug in self.iplugs })
        self.__dict__.update({ plug.name : plug for plug in self.oplugs })

    @property
    def parent(self):
        return self._parent

    @property
    def path(self):
        if not self._parent:
            return ""

        # Iterate over our other parent's children and identify our path relative to that parent
        for i,child in enumerate(self.parent):
            if child is self:
                # The path to a child is, by default, "path[i]" where i is the child's index
                relpath = "["+str(i)+"]"

                # However, if the child is also an attribute, like .foo, then path is "path.foo"
                for name,attr in self.parent.__dict__.iteritems():
                    if self is attr:
                        relpath = "."+name
                        break

                break

        return self.parent.path + relpath

    def _visit(self,path,callback):
        # Visit ourself first.
        callback(path,self)

        # Visit all our own attributes, *except* any nodes (let supernode deal with them if need be)
        for name,attr in self.__dict__.iteritems():
            if isinstance(attr,(plug,paramdef)):
                newval = callback(path+"."+name,attr)
                if newval is not None:
                    setattr(self,name,newval)

    def visit(self,callback):
        """
        Visits each plug of the node, and each child of the node, 
        invoking callback(path,object) for each one. For example, 
        if 'node' is a "dropout" node, then node.visit would call:

            callback(".X",node.X)        # visit the input plug X
            callback(".Z",node.Z)        # visit the output plug Z
            callback(".rate",node.rate)  # visit the dropout rate (a scalar)

        If 'node' were a supernode, such as a chain([bias,dropout]),
        then node.visit would call

            callback(".X",node.X)                  # visit supernode's input plug X (which will have dst bias.X)
            callback(".Z",node.Z)                  # visit supernode's output plug X (which will have src dropout.Z)
            callback("[0]",node.children[0])       # visit first child node
            callback("[0].X",node.children[0].X)   # visit first child's input plug X
            callback("[0].b",node.children[0].b)   # visit first child's bias plug b (i.e. trainable weights)
            callback("[0].Z",node.children[0].Z)   # visit first child's output plug Z
            callback("[1]",node.children[1])       # ...
            callback("[1].X",node.children[1].X)
            callback("[1].Z",node.children[1].Z)
            callback("[1].rate",node.children[1].rate)
        """
        self._visit("",callback)

    def set_ninstance(self,ninst):
        def set_ninst(path,obj):
            if isinstance(obj,node):
                obj.ninst = ninst
        self.visit(set_ninst)

    def _set_ninstance(self,ninst):
        raise NotImplementedError("Subclass should implement _set_ninstance.")

    def slice_inst(self,inst):
        # First, ask each internal node to pull its _fpval and slice out
        # the parameters for instance number 'inst'.
        # The internal nodes may pull their _fpval from supernode plugs.
        def do_slice_inst(path,obj):
            if isinstance(obj,node):
                obj._slice_inst(inst)
                obj.ninst = 1
        self.visit(do_slice_inst)

        # Second, a supernode parameter plug may still have 
        def replace_with_origin(path,obj):
            if isinstance(obj,plug) and isinstance(obj.node,supernode):
                if obj.origin() is not obj and obj.origin()._fpval is not None:
                    obj.fpval = obj.origin()._fpval
        self.visit(replace_with_origin)

    def _slice_inst(self,inst):
        return # by default, do nothing

    def find(self,path):
        """Returns the node or attribute specified by the path, relative to this node"""
        if path == "":
            return self
        assert path.startswith(".")
        return getattr(self,path[1:])

    def find_and_set(self,path,val):
        """Sets the node or attribute specified by the path, relative to this node"""
        if "." not in path:
            setattr(path, val)
        assert path.startswith(".")
        return getattr(self, path[1:])

    def requirements(self):
        # Return a dictionary of "requirements" that the external program should
        # make sure that the data accomodates. Each node has the opportunity to list
        # some requirement for the input data, such as data being padded.
        # Each requirement is in the form { "a.b.requirement_name" : requirement_value }
        reqs = {}
        def add_reqs(path,obj):
            if isinstance(obj,node):
                reqs.update({ (path+"."+key) : val  for key,val in obj._requirements().iteritems() })
        self.visit(add_reqs)
        return reqs

    def _requirements(self):
        return {}

    def data_requirements(self):
        reqs = self.requirements()
        padding = max([0]+[val for key,val in reqs.items() if key.endswith("sequence_padding")])
        target = [val for key,val in reqs.items() if key.endswith("target")]
        assert len(target) <= 1
        if target:
            target = target[0]
        return { 'target' : target, 'padding' : padding }

    def calc_shapes(self,visited):
        if self in visited:
            return
        argnames = self._calc_shapes.__code__.co_varnames[1:self._calc_shapes.__code__.co_argcount]
        inputs = { name : getattr(self,name) for name in argnames }
        for iplug in inputs.itervalues():
            iplug._calc_shape(visited+[self])

        self._calc_shapes(**inputs)

        #for iplug in inputs.itervalues():
            #if iplug._shape is not None:
                #print "%s._calc_shape() succeeded: %s" % (iplug.path,str(iplug._shape))


    def _calc_shapes(self): raise NotImplementedError("Subclass should implement _calc_shapes.")

    def clear(self):
        def clear_plug(path,obj):
            if isinstance(obj,plug):
                if obj.has_src() or obj.is_oplug():
                    obj._fpval = plug_null  # Clear all fprop values except input plugs with a fixed (external) value set on them
                obj._bpval = plug_null
        self.visit(clear_plug)

    def fprop(self):
        """
        Calculate the value of all output plugs based on the current iplugs.
        """
        for oplug in self.oplugs:
            assert oplug._fpval is plug_null, "fprop on a node can only be called while it's output fpvals are undefined"

        # If the subclass of this object defined
        #  def _fprop(self,X,Y):
        #     ...
        # then we want to automatically pull the value stored by plugs X and Y 
        # and pass those values as _fprop's arguments.
        argnames = self._fprop.__code__.co_varnames[1:self._fprop.__code__.co_argcount]
        inputs = { }
        for argname in argnames:
            p = getattr(self,argname)
            assert p in self.iplugs, "_fprop cannot request parameter %s, as its value has not been determined yet" % argname
            inputs[argname] = p.fpval

        # Call subclass _fprop implementation and get a list of outputs
        outputs = self._fprop(**inputs)
        if not isinstance(outputs,tuple):
            outputs = (outputs,)

        # Move each output to the corresponding oplug
        for oplug,fpval in zip(self.oplugs,outputs):
            oplug._fpval = fpval

    def bprop(self):
        """
        Calculate 'delta' for each input plug.
        For example, if iplugs=[X] and oplugs=[Z], 
        then bprop should compute dX from X,Z,dZ.
        """
        for iplug in self.iplugs:
            assert iplug._bpval is plug_null, "bprop on a node can only be called while it's input bpvals are undefined"

        # If the subclass of this object defined
        #     def _bprop(self,X,Z,dZ):
        #        ...
        # then we want to automatically call 
        #     self._bprop(X._fpval,Z._fpval,Z._bpval)
        #
        argnames = self._bprop.__code__.co_varnames[1:self._bprop.__code__.co_argcount]
        inputs = {}
        for argname in argnames:
            if argname.startswith("d") and hasattr(self,argname[1:]):
                # If the plug is dZ, then substitute with Z.bpval
                p = getattr(self,argname[1:])
                assert p in self.oplugs, "_bprop cannot request parameter %s, as its value has not been determined yet" % argname
                inputs[argname] = p.bpval
            else:
                # Otherwise, simply use Z.fpval
                p = getattr(self,argname)
                inputs[argname] = p.fpval

        # Call subclass _bprop implementation and get a list of deltas
        outputs = self._bprop(**inputs)
        if not isinstance(outputs,tuple):
            outputs = (outputs,)

        # Copy each output to the corresponding iplug's delta value
        for iplug,bpval in zip(self.iplugs,outputs):
            iplug._bpval = bpval

    def _fprop(self): raise NotImplementedError("Cannot propagate through node.")
    def _bprop(self): raise NotImplementedError("Cannot back-propagate through node.")

    def eval(self,**kwargs):

        want_clear = kwargs.pop('clear',True)
        want_bprop_inputs = kwargs.pop('want_bprop_inputs',False)
        bprop_inputs_loss = kwargs.pop('bprop_inputs_loss',None)
        self.clear()

        globals.flags.push("want_bprop_inputs",want_bprop_inputs)
        globals.flags.push("bprop_mode",want_bprop_inputs)
        
        # For each keyword, set the corresponding plug's value
        for key,val in kwargs.iteritems():
            getattr(self,key).fpval = val

        # Pull the final loss value for this minibatch
        result = { p.name : p.fpval for p in self.oplugs }

        # If needed, also pull the backprop'd input deltas and include them in the result
        if want_bprop_inputs or bprop_inputs_loss:
            # First set up special backprop values: "Z" (the prediction) backpropagates a negative value
            # All other output plugs (e.g. costs) backpropagate zero.
            for p in self.oplugs:
                if p.name == "Z":
                    bprop_inputs_loss.batchmean = False # Disable scaling gradient by minibatch size
                    bprop_inputs_loss.Z.fpval = result["Z"]
                    bprop_inputs_loss.Y.fpval = sm.zeros_like(result["Z"])
                    #bprop_inputs_loss.Y.fpval = -1.*sm.ones_like(result["Z"])
                    p._bpval = bprop_inputs_loss.Z.bpval
                    result['Zmask'] = bprop_inputs_loss.Zmask._fpval

                    # Only backprop gradient of target #0, not the other targets
                    if p._bpval.shape[1] > 1:
                        p._bpval[:,1:] = sm.zeros_like(p._bpval[:,1:])
                    #p._bpval = -result["Z"]
                    #p._bpval = -sm.ones_like(result["Z"])
                else:
                    p._bpval = sm.zeros((0,0))

            # Now backpropagate to each input, and store the result
            if want_bprop_inputs:
                result.update({ "d"+p.name : p.bpval for p in self.iplugs if p.name in kwargs})

        globals.flags.pop("want_bprop_inputs")
        globals.flags.pop("bprop_mode")

        # Clear all stored values in the dependency graph, effectively resetting it
        if want_clear:
            self.clear()
        for key in kwargs.iterkeys():
            getattr(self,key).fpval = plug_null

        return result


    # Functions to allow easier connect(X,Y) syntax.
    # Allows things like X >> Y,  Y << X where Y is a node
    # and X is a node or a literal.
    def __lshift__(self,other):  connect(other,self); return self
    def __rshift__(self,other):  connect(self,other); return other
    def __rlshift__(self,other): connect(self,other); return other
    def __rrshift__(self,other): connect(other,self); return self


#####################################################################


class supernode(node):
    """
    A supernode wraps a subgraph with a simple interface.
    All of this node's inputs are forwarded internal inputs, 
    likewise all its outputs are and forwarded from internal outputs.
    """
    def __init__(self,children,name=None):
        self._children = children
        for child in children:
            child._parent = self

        # Collect all unconnected plugs so that we can expose 
        # corresponding external plugs on the supernode
        iplugs = [plug for child in children
                       for plug  in child.iplugs
                       if not plug.has_src()]
        oplugs = [plug for child in children
                       for plug  in child.oplugs
                       if not plug.has_dst()]

        # Figure out what the external name for each internal plug should be,
        # e.g. two internal plugs named "W" get renamed "W0" and "W1".
        iplug_names = self._calc_plug_names(iplugs)
        oplug_names = self._calc_plug_names(oplugs)

        # Call node constructor to initialize our own plugs.
        super(supernode,self).__init__(iplug_names,oplug_names,name)

        # Connect each external plug to its internal counterpart
        for src,dst in zip(self.iplugs,iplugs): src >> dst; src.inst_axis = dst.inst_axis
        for src,dst in zip(oplugs,self.oplugs): src >> dst; dst.inst_axis = src.inst_axis

    @staticmethod
    def _calc_plug_names(plugs):
        # If two internal plugs are named R, for example, then their external
        # counterparts get named R0,R1
        raw_names = [plug.name for plug in plugs]
        new_names = []
        for i,name in enumerate(raw_names):
            while (raw_names+new_names).count(name) > 1:
                name = name + str(raw_names[:i].count(name))  # R,R becomes R0,R1
            new_names.append(name)
        return new_names

    def _visit(self,path,callback):
        # First visit all our own attributes
        super(supernode,self)._visit(path,callback)

        # Next, iterate over our other immediate children and recursively call _visit
        for i,child in enumerate(self):
            # The path to a child is, by default, "path[i]" where i is the child's index
            childpath = "["+str(i)+"]"

            # However, if the child is also an attribute, like .foo, then path is "path.foo"
            for name,attr in self.__dict__.iteritems():
                if attr is child:
                    childpath = "."+name
                    break

            child._visit(path+childpath,callback)

    def _set_ninstance(self,ninst):
        return  # Do nothing on the supernode itself. Let set_ninstance visit all our children automatically.

    def find(self,path):
        """Returns the node or attribute specified by the path, relative to this node"""
        if path == "":
            return self

        # If path is "[i]...", then call children[i].find(...)
        match = re.search("^\[(\d+)\](.*)",path)
        if match:
            return self._children[int(match.group(1))].find(match.group(2))

        # If path is ".name", then return getattr(self,name)
        match = re.search("^\.(\w+)$",path)
        if match:
            return getattr(self,match.group(1))

        # If path is ".name..." then 'name' must be a child node, and we return getattr(self,name).find(...)
        match = re.search("^\.(\w+)(.+)",path)
        if match:
            return getattr(self,match.group(1)).find(match.group(2))

    def __iter__(self):      return self._children.__iter__()
    def __getitem__(self,i): return self._children[i]
    def __len__(self):       return len(self._children)

    def calc_shapes(self,visited): return # Do nothing
    def fprop(self): return # Do nothing
    def bprop(self): return # Do nothing
    def _fprop(self): return # Do nothing
    def _bprop(self): return # Do nothing

    def slice_inst(self,inst):
        super(supernode,self).slice_inst(inst)
