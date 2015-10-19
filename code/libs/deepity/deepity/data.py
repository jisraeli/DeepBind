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
import copy
import logging
import tempfile
import cPickle as pickle
import numpy as np
import numpy.random as npr
import smat as sm
from .util import tic,toc
from os.path import join,basename,splitext,exists

def _default_augmented(data):
    return data

class datasource(object):
    """
    A datasource provides sliced views into a dataset.
    For example, it might be used like
        >>> chunk = my_datasource[first:last]
        >>> chunk.Y
            array([[...]])   # array of training targets
    A subclass may store the data itself in host memory, 
    in device memory, synthesized, or even paged from disk.

    If the subclass items contain attribute "Y", it is assumed
    to be be training targets, stored as an N x (M*numtargets) numpy
    array where M is the dimensionality 
    """
    def __init__(self, input_attrs, output_attrs, extra_attrs=None):
        self._input_attrs  = input_attrs
        self._output_attrs = output_attrs
        self._extra_attrs  = extra_attrs
        self._data_attrs   = input_attrs + output_attrs
        self._all_attrs    = input_attrs + output_attrs
        if extra_attrs:
            self._all_attrs = self._all_attrs + extra_attrs
        self.__dict__.update({ name : None for name in self._data_attrs })
        if not hasattr(self,"targetnames"):
            self.targetnames = ["Target"]
        self.augmented = _default_augmented


    def input_attrs(self):  return self._input_attrs
    def output_attrs(self): return self._output_attrs
    def data_attrs(self):   return self._data_attrs
    def input_data(self):   return { key : getattr(self,key) for key in self._input_attrs }
    def output_data(self):  return { key : getattr(self,key) for key in self._output_attrs }
    def data(self):         return { key : getattr(self,key) for key in self._data_attrs }

    def attrdim(self,attrname):
        raise NotImplementedError("Subclass should implement attrdim().")
    
    def open(self):
        # Subclass should treat this function as a request to 'open' 
        # the database and get it ready for use.
        raise NotImplementedError("Subclass should implement open().")

    def __len__(self):
        # Subclass should return the number of individual training examples.
        raise NotImplementedError("Subclass should implement __len__.")
    
    def __getitem__(self, indices):
        # Subclass should return a datasource instance that contains one attribute for each input.
        # If the object has an attribute "Y", then that will be used as targets during
        # any training task.
        raise NotImplementedError("Subclass should implement __getitem__.")

    def dump_preprocessors(self, outdir, cols=None):
        return

    def shuffle(self):
        raise NotImplementedError("Subclass did not implement shuffle.")

    def asbatches(self, batchsize=256):
        # Subclass should return a new object, where that object's __getitem__[i] 
        # returns minibatch #i, rather than an individual item.
        # Needed for use with batch-based trainers, like SGD.
        # If use_smat is True, then the resulting batches should be 
        # converted to an smat.sarray intance, rather than numpy.ndarrays.
        raise NotImplementedError("Subclass did not implement asbatches.")

    def astargets(self, targetnames):
        # For multi-task learning, this 
        raise NotImplementedError("Subclass did not implement astargets.")
    
    def split(self, index=0, nsplit=1):
        # Returns two datasources, A and B, split in a way
        # that is useful for k-fold cross validation (k = nsplit+1).
        # Suppose we had 10 samples:
        #    0,1,2,3,4,5,6,7,8,9
        # then split() produces
        #    A = 0,1,2,3,4
        #    B = 5,6,7,8,9
        # whereas split(1) produces the reverse
        #    A = 5,6,7,8,9
        #    B = 0,1,2,3,4
        # and, similarly, split(1,3) produces an approximate 
        # 3-way split
        #    A = 0,1,2,6,7,8,9
        #    B = 3,4,5
        # Split A is always the larger of the two splits.
        assert index <= nsplit
        if nsplit > 0:
            size = len(self)
            at = [int(x) for x in np.linspace(0, size, nsplit+1+1)]
            rindex = nsplit-index
            sliceB = slice(at[rindex],at[rindex+1])
            sliceA1 = slice(at[0],at[rindex])
            sliceA2 = slice(at[rindex+1],at[-1])
            #print sliceB, sliceA1, sliceA2
            #splitsize = size // (nsplit+1)
            #sliceA1 = slice(0,(nsplit-index)*splitsize)
            #sliceA2 = slice((nsplit-index+1)*splitsize,size)
            #sliceB  = slice(sliceA1.stop,sliceA2.start)
            return self._split(sliceA1,sliceA2,sliceB)
        return self,None

    def _split(self, sliceA1, sliceA2, sliceB):
        # Subclass should implement the rest of split()
        raise NotImplementedError("Subclass did not implement split.")

######################################################################

class resident_datasource(datasource):
    """
    A datasource that can be stored completely in host memory.
    That means it is straight-forward to support arbitrary 
    slicing, or loading/saving, of the dataset.
    """

    def __init__(self, input_attrs, output_attrs, extra_attrs=None, join=None):
        super(resident_datasource,self).__init__(input_attrs, output_attrs, extra_attrs)
        self.join = join

    def attrdim(self,attrname):
        attr = getattr(self,attrname)
        if isinstance(attr,np.ndarray):
            dim = attr.shape[1] if attr.ndim > 1 else 1
        elif isinstance(attr,list):
            dim = 1
        else:
            raise NotImplementedError("Cannot determine dimension of datasource attribute %s" %attrname)

        if attrname in self._output_attrs:
            assert dim % self.ntask() == 0
            dim /= self.ntask()
        return dim

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        newsrc = copy.copy(self)
        for name in newsrc._all_attrs:
            oldattr = getattr(newsrc, name)
            if isinstance(oldattr, np.ndarray):
                newval  = oldattr[i,:]
                setattr(newsrc, name, newval)   # Replace newsrc.X with newsrc.X[i]
            elif isinstance(oldattr, list):
                if isinstance(i,slice):
                    newval = oldattr[i]
                else:
                    newval = [oldattr[x] for x in i]
                setattr(newsrc, name, newval)

        return newsrc
    
    def shuffle(self, perm=None):
        if perm is None:
            perm = npr.permutation(len(self))
        for name in self._all_attrs:
            attr = getattr(self, name)
            if isinstance(attr,np.ndarray):
                attr[:] = attr[perm]  # Permute the rows
            elif isinstance(attr,list):
                setattr(self, name, [attr[i] for i in perm])
            else:
                raise NotImplementedError("Unrecognized attribute type %s for attribute %s" % (type(attr), name))

    def _split(self, sliceA1, sliceA2, sliceB):
        # For A, we must vertically stack the A1 and A2 portions of each attribute
        A = copy.copy(self)
        for name in A._all_attrs:
            oldattr = getattr(A,name)
            if isinstance(oldattr,np.ndarray):
                newval  = np.vstack((oldattr[sliceA1,:],oldattr[sliceA2,:]))
                setattr(A,name,newval)       # Replace A.X with vstack(A.X[A1],A.X[A2])
            elif isinstance(oldattr,list):
                newval  = oldattr[sliceA1] + oldattr[sliceA2]
                setattr(A,name,newval)       # Replace A.X with A.X[A1]+A.X[A2]
            else:
                raise NotImplementedError("Unrecognized attribute type %s for attribute %s" % (type(oldattr), name))

        # For B, we can just directly return contiguous sliceB
        B = self[sliceB]

        return A,B

    def astargets(self, targetnames):
        # Returns a *shallow* copy of the datasource, where the original targets
        # are now swapped out for some new, derived targets. This is used
        # during hyperparameter search, when there are several models being
        # trained in parallel, each on an arbitrary column slice of the original
        # matrix of targets Y.
        # Create a shallow copy, and replace the OUTPUT attribute with our new arrangement of columns

        # First, compute the number of targets we currently have, and the number of columns for each target
        newsrc = copy.copy(self)
        ntarget = len(self.targetnames)
        if ntarget == 0:
            return newsrc
        assert self.Y.shape[1] % ntarget == 0, "Number of target columns was not divisible by number of targets!"
        targetsize = self.Y.shape[1] // ntarget

        # Then, compute a list of column indices that will replicate/slice the columns of the current targets
        targetidx = [self.targetnames.index(name) for name in targetnames]
        cols =  np.repeat(targetidx, targetsize)*targetsize      # If targetidx=[0,1,3,4] and targetsize=3 this gives [0,0,0,3,3,3,9,9,9,12,12,12] 
        cols += np.tile(np.arange(targetsize),len(targetnames))  # ... which then gets turned into [0,1,2,3,4,5,9,10,11,12,13,14] 

        # Create a shallow copy where the 
        newsrc.targetnames = targetnames
        if hasattr(newsrc,"targets"):
            newsrc.targets = self.targets[:,cols].copy()
        if hasattr(newsrc,"Ymask"):
            newsrc.Ymask = self.Ymask[:,cols].copy()
        newsrc.Y = self.Y[:,cols].copy()

        return newsrc

    def convert_to_sarray(self):
        # Upload the data to a GPU device
        for name in self._data_attrs:
            oldattr = getattr(self,name)
            setattr(self,name,sm.asarray(oldattr))

    def asbatches(self, batchsize=64, reshuffle=False):
        # By default, create a bunch of tiny resident_datasource instances,
        # and upload each to the GPU as we go.
        n = len(self)
        assert n > 0
        nbatch = (n + batchsize - 1) // batchsize
        batches = []
        for i in range(nbatch):
            idx = np.arange(i*batchsize,min(n,(i+1)*batchsize))
            batch = self[idx]
            batch.convert_to_sarray()
            batches.append(batch)
        return shuffled_repeat_iter(batches, reshuffle)

    def close(self):
        #for attr in self._all_attrs:
        #    delattr(self,attr)
        return

# Return a wrapper around our list of batches.
# The wrapper will keep serving batches forever,
# cycling through the list, and shuffling the order
# each time it starts a new pass.
class shuffled_repeat_iter(object):
    """
    Repeated traversal over a list of items, where the order
    is re-shuffled for each pass through the items.
    """
    def __init__(self, items, reshuffle):
        self._items = items
        self._reshuffle = reshuffle
        self._order = np.arange(len(self._items))
        self._index = len(items)

    def __len__(self):       return len(self._items)
    def __getitem__(self,i): return self._items[i]
    def __iter__(self):      return self._items.__iter__()

    def shuffle(self):
        np.random.shuffle(self._order)

    def curr(self):
        if self._index == len(self._items):
            if self._reshuffle:
                np.random.shuffle(self._order)
            self._index = 0
        return self._items[self._order[self._index]]

    def next(self):
        item = self.curr()
        self._index += 1
        return item


##############################################

def make_predictions(model,datasrc):
    if isinstance(datasrc,datasource):
        datasrc = datasrc.asbatches(128)
    results = {}
    for batch in datasrc:
        args = batch.input_data()
        result = model.eval(**args)
        for key,val in result.iteritems():
            if not key in results:
                results[key] = []
            results[key].append(result[key].asnumpy())
    for key,val in results.iteritems():
        results[key] = np.vstack(results[key])
    return results


def count_errors(model,datasrc):
    # If model.ninst > datasrc.ntask, then that means multiple models 
    # were trained on the original data, so we need to replicate the task.
    # For each task, we generate model.ninst/data.ntask copies 
    if isinstance(datasrc,datasource):
        datasrc = datasrc.asbatches(128)
    Z = make_predictions(model,datasrc)["Z"]
    Y = np.vstack([sm.asnumpy(batch.Y) for batch in datasrc])

    error_counts = []
    for i in range(model.ninst):
        s = slice(i*Z.shape[1]//model.ninst,(i+1)*Z.shape[1]//model.ninst)
        count = np.sum(np.argmax(Z[:,s],axis=1) != np.argmax(Y[:,s],axis=1))
        error_counts.append(count)
    error_counts = np.asarray(error_counts)
    return error_counts
