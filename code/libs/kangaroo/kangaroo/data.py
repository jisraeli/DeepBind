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
import re
import os
import csv
import copy
import time
import logging
import tempfile
import itertools
import smat as sm
import numpy as np
import numpy.random as npr
import deepity
import scipy
import scipy.stats
import gzip
import cPickle
from .util import acgt2ord,acgtcomplement,ord2acgt
from .     import globals
from deepity.util import tic,toc
from os.path import join,basename,splitext,exists
from math import tanh

_dinucs = ["".join(dinuc) for dinuc in itertools.product(['A','C','G','T'],['A','C','G','T'])]

def dinuc_enrichment_features(s):
    # Assumption: all kmers have same length
    n = len(s)
    k = len(_dinucs[0])
    expected = float(n-k+1) / (4.**k)
    feats = []
    for dinuc in _dinucs:
        count = sum(1 for _ in re.finditer('(?=%s)'%dinuc, s))  # count all occurrances of kmer; str.count doesn't count overlapping kmers
        #feats.append(count/expected-1.0)
        feats.append(0)
    return feats

#########################################################################

class datasource(deepity.resident_datasource):
    """
    A kangaroo datasource that serves input attributes:
       - X_Sequence0...X_Sequencek: a list of "sequence columns", 
         where each column has the same size, and 
         is provided under the name X_SequenceName (where SequenceName
         was taken from the column header in the sequencefile)
       - F: a single table of features F, taken from the featuresfile.
    and output attributes:
       - Y: the targets, with one column per target
       - Ymask: the mask of non-NaN elements in Y
    """

    @staticmethod
    def fromtxt(sequencefile, featurefile=None, targetfile=None, foldfilter=None, maxrows=None, targetcols=None, sequencenames=None, featurenames=None, targetnames=None, dinucfeatures=True, **kwargs):

        # Load each text file, possible from cache
        sequencenames, sequences = loadtxt(sequencefile, maxrows=maxrows, colnames=sequencenames)
        featurenames,  features  = loadtxt(featurefile,  maxrows=maxrows, colnames=featurenames)
        targetnames,   targets   = loadtxt(targetfile,   maxrows=maxrows, colnames=targetnames, usecols=targetcols)

        # If the sequence file contained the targets, then split off that extra column
        if targets is None and sequencenames[-1].lower() == "bound":
            targetnames = [sequencenames.pop()]
            targets   = [row[-1]  for row in sequences]
            sequences = [row[:-1] for row in sequences]

        rowidx = np.arange(len(sequences)).astype(np.uint32).reshape((-1,1))

        # Filter out rows that are not actually sequences
        if foldfilter:
            idx = [i for i in range(len(sequences)) if sequences[i][0] in foldfilter]
            sequences = [sequences[i] for i in idx]
            rowidx = rowidx[idx]
            if features is not None:
                features  = [features[i] for i in idx]
            if targets is not None:
                targets   = [targets[i] for i in idx]

        # Strip out the Fold ID and Event ID columns of the sequence array.
        if sequencenames and sequencenames[0].lower() in ("fold","foldid","fold id"):
            sequencenames = sequencenames[2:]
            foldids   = [row[0]  for row in sequences]
            sequences = [row[2:] for row in sequences]
        else:
            foldids   = ["A" for i in range(len(sequences))]

        # Automatically add dinucleotide frequency features for each input sequence
        if dinucfeatures:
            if not featurenames:
                featurenames = []
                features = [[] for row in sequences]
            for seqname in sequencenames:
                featurenames += [seqname+"."+dinuc for dinuc in _dinucs]
            for rowfeats, rowseqs in zip(features, sequences):
                for s in rowseqs:
                    rowfeats += dinuc_enrichment_features(s)

        return datasource(sequencenames, sequences, featurenames, features, targetnames, targets, foldids, rowidx, **kwargs)

    @staticmethod
    def _generate_dinuc_featurevec(X):
        return dinuc_enrichment_features(ord2acgt(X))

    def __init__(self, sequencenames, sequences, featurenames, features, targetnames, targets, foldids, rowidx):

        self.sequencenames = sequencenames
        self.featurenames  = featurenames if features is not None else []
        self.targetnames   = targetnames  if targets  is not None else []

        nsequence = len(self.sequencenames)
        seqattrs  = sum([self._seqattrnames(i) for i in range(nsequence)],())
        featattr  = [("F",),("features",)]       if features is not None else [(),()]
        targattrs = [("Y","Ymask"),("targets",)] if targets is not None else [(),()]
        foldattr  = ("foldids",)                 if foldids is not None else ()

        # Initialize the datasource superclass by telling it how many
        # input attributes to expect, based on 
        super(datasource,self).__init__(input_attrs  = seqattrs + featattr[0],
                                        output_attrs = targattrs[0],
                                        extra_attrs  = ("rowidx","sequences") + featattr[1] + targattrs[1] + foldattr,  # Attributes not batched or sent to the GPU
                                        )

        nrow = len(sequences)

        self.rowidx    = rowidx
        self.sequences = sequences
        self.features  = np.asarray(features, dtype=np.float32).reshape((nrow,-1)) if features is not None else None
        self.targets   = np.asarray(targets, dtype=np.float32).reshape((nrow,-1))  if targets  is not None else None
        self.foldids   = foldids

        self._task_ids = sorted(self.targetnames)
        self.preprocessors = {"features" : [], "targets" : []}
        self.requirements = {}
        self._create_attributes()
    
            
    def extract_fold(self, foldid):
        idx = np.asarray([i for i in range(len(self)) if self.foldids[i] == foldid])
        return self[idx]




    def add_requirements(self, reqs):
        self.requirements.update(reqs)


    def clamp_extremes(self, lo, hi):
        self.Y = self.Y.copy()  # Make a copy in case we're looking at a row-slice of a larger datasource
        self.Ymask = self.Ymask.copy()
        self.preprocessors = copy.deepcopy(self.preprocessors)
        pp = _clamp_extremes_preprocessor(self.Y, lo, hi)
        self.targets = self.Y.copy()
        self.Ymask = ~np.isnan(self.Y)
        self.preprocessors["targets"].append(pp)


    def logtransform_targets(self):
        self.Y = self.Y.copy()  # Make a copy in case we're looking at a row-slice of a larger datasource
        self.Ymask = self.Ymask.copy()
        self.preprocessors = copy.deepcopy(self.preprocessors)
        pp = _logtransform_preprocessor(self.Y)
        self.preprocessors["targets"].append(pp)
        

    def arcsinhtransform_targets(self):
        self.Y = self.Y.copy()  # Make a copy in case we're looking at a row-slice of a larger datasource
        self.Ymask = self.Ymask.copy()
        self.preprocessors = copy.deepcopy(self.preprocessors)
        pp = _arcsinhtransform_preprocessor(self.Y)
        self.preprocessors["targets"].append(pp)
        

    def normalize_targets(self, **requirements):
        requirements.update(self.requirements)
        if any([value == 'logistic' for value in requirements.values()]):
            intercept_mode = "min"
        else:
            intercept_mode = "mean"
        self.Y = self.Y.copy()  # Make a copy in case we're looking at a row-slice of a larger datasource
        self.Ymask = self.Ymask.copy()
        self.preprocessors = copy.deepcopy(self.preprocessors)
        pp = _normalize_preprocessor(self.Y, intercept_mode)
        self.preprocessors["targets"].append(pp)


    def normalize_features(self):
        if hasattr(self,"F"):
            self.F = self.F.copy()  # Make a copy in case we're looking at a row-slice of a larger datasource
            self.preprocessors = copy.deepcopy(self.preprocessors)
            pp = _normalize_preprocessor(self.F, "mean")
            self.preprocessors["features"].append(pp)


    def _create_attributes(self):
        # Adds public attributes with names matching 
        nrow = len(self)
        nseq = len(self.sequencenames)
        for i in range(nseq):
            Xname,Rname = self._seqattrnames(i)
            self.__dict__[Xname] = [row[i] for row in self.sequences]
            self.__dict__[Rname] = np.zeros((nrow,1), np.uint32) # empty until set during asbatches()
        if self.features is not None:
            self.__dict__['F'] = self.features.copy()
        if self.targets is not None:
            self.__dict__['Y'] = self.targets.copy()
            self.__dict__['Ymask'] = ~np.isnan(self.targets)

    def _seqattrnames(self, index):
        return ('X_%s'%self.sequencenames[index], 'R_%s'%self.sequencenames[index])

    def __len__(self):
        return len(self.rowidx)

    def open(self):
        return

    def load_preprocessors(self, indir):
        if not os.path.exists(join(indir, 'preprocessors.pkl')):
            return
        with open(join(indir, 'preprocessors.pkl'),'rb') as f:
            assert not self.preprocessors['features'], "Cannot load preprocessors for a datasource with already-preprocessed features."
            assert not self.preprocessors['targets'],  "Cannot load preprocessors for a datasource with already-preprocessed targets."
            self.preprocessors = cPickle.load(f)
            for pp in self.preprocessors['features']:
                self.F = self.F.copy()
                pp.apply(self.F)
            for pp in self.preprocessors['targets']:
                self.Y     = self.Y.copy()
                self.Ymask = self.Ymask.copy()
                pp.apply(self.Y)

    def dump_preprocessors(self, outdir, cols=None):
        if cols is None:
            cols = slice(None)
        preprocessors_sliced = { 'features' : self.preprocessors['features'],
                                 'targets'  : [pp.slice(cols) for pp in self.preprocessors['targets']] }
        with open(join(outdir, 'preprocessors.pkl'), 'wb') as f:
            cPickle.dump(preprocessors_sliced, f)

    def _insert_reversecomplements(self):
        if "reverse_complement" not in globals.flags:
            return

        nseq = len(self.sequencenames)
        for i in range(nseq):
            Xname,Rname = self._seqattrnames(i)
            X = getattr(self, Xname)
            rows = range(len(X))
            Xrev = [acgtcomplement(x[::-1]) for x in X]
            newX = [Xrev[i] if j else X[i] for i in rows for j in (0,1)]
            setattr(self, Xname, newX)
        
        # For all the other attributes, simply duplicate their rows.
        duprows = np.repeat(np.arange(len(self)), 2)
        if hasattr(self, "rowidx"):self.rowidx = self.rowidx[duprows,:]
        if hasattr(self, "Y"):     self.Y      = self.Y[duprows,:]
        if hasattr(self, "Ymask"): self.Ymask  = self.Ymask[duprows,:]
        if hasattr(self, "F"):
            self.F = self.F[duprows,:]

            # HACK: For dinuc statistic features, adjust columns.
            fwdrows = np.arange(0,len(self.F),2)
            revrows = np.arange(1,len(self.F),2)
            for j in range(len(self.featurenames)):
                fname = self.featurenames[j]
                if "." in fname:
                    prefix, suffix = fname.rsplit(".",1)
                    if suffix in _dinucs:
                        rcsuffix = acgtcomplement(suffix[::-1])
                        k = self.featurenames.index(prefix+"."+rcsuffix)
                        self.F[revrows,k] = self.F[fwdrows,j]
        return 


    def asbatches(self, batchsize=64, reshuffle=False):
        n = len(self)
        assert n > 0
        nbatch = (n + batchsize - 1) // batchsize
        nseq = len(self.sequencenames)
        padding = self.requirements.get('padding',0)
        batches = []
        for i in range(nbatch):

            # Slice a our data attributes row-wise, according to batch index
            batch = self[np.arange(i*batchsize,min(n,(i+1)*batchsize))]
            batch._insert_reversecomplements()

            # Convert each sequence attribute from a list of strings ("GATC") to a 
            # single contiguous numpy array X (0..3), along with a list of
            # regions R that identify the batch-relative offsets to the start/end 
            # of each individual sequence
            for i in range(nseq):
                Xname,Rname = self._seqattrnames(i)
                batchX = getattr(batch, Xname)
                batchR = np.asarray(np.cumsum([0]+[padding+len(x) for x in batchX]),np.uint32).reshape((-1,1))
                batchR = np.hstack([batchR[:-1],batchR[1:]])

                # Convert list of strings to giant contiguous array of integers 0..3, 
                # with padding values of 255 put between the individual sequences
                batchX = acgt2ord(("."*padding).join([""]+[x for x in batchX]+[""])).reshape((-1,1))

                # Convert each batch from numpy array to sarray, 
                # and then quickly forget about the numpy batch
                batchX = sm.asarray(batchX)
                batchR = sm.asarray(batchR)

                setattr(batch, Xname, batchX)
                setattr(batch, Rname, batchR)
                setattr(batch, "regions", batchR)
                batch._data_attrs = batch._data_attrs + ("regions",)

            if hasattr(batch,"F") and batch.F is not None: 
                batch.F = sm.asarray(batch.F,sm.get_default_dtype())
            if hasattr(batch,"Y") and batch.Y is not None:
                batch.Y = sm.asarray(batch.Y,sm.get_default_dtype())
                if isinstance(batch.Ymask,np.ndarray):
                    batch.Ymask = sm.asarray(batch.Ymask)

            batches.append(batch)
        return deepity.shuffled_repeat_iter(batches, reshuffle)


###################################################################################

class _preprocessor(object):

    def apply(self, data): raise NotImplementedError("Subclass should implement this.")
    def undo(self, data):  raise NotImplementedError("Subclass should implement this.")
    def slice(self, cols): return self # Do nothing by default



class _normalize_preprocessor(_preprocessor):
    def __init__(self, data, intercept_mode):
        self.scales = []
        self.biases = []
            
        # Preprocess each column to have unit variance and zero mean
        ncol = data.shape[1]
        for i in range(ncol):
            col = data[:,i:i+1]
            mask = ~np.isnan(col)
            if intercept_mode == "mean":
                bias  = np.mean(col[mask].ravel())
                scale = np.std(col[mask].ravel())
            elif intercept_mode == "min":
                bias  = np.min(col[mask].ravel())
                scale = np.max(col[mask].ravel()) - bias
            else:
                raise NotImplementedError()

            # Save the scales and biases for later, in case we're asked to undo this transformation
            self.scales.append(scale)
            self.biases.append(bias)

        self.scales = np.asarray(self.scales)
        self.biases = np.asarray(self.biases)

        self.apply(data)

    def apply(self, data):
        # Preprocess each column to have unit variance and zero mean
        ncol = data.shape[1]
        for i in range(ncol):
            col = data[:,i:i+1]
            mask = ~np.isnan(col)
            # Basically assigns col[:] = (col-bias) / scale
            col[mask] -= self.biases[i]
            if self.scales[i]:
                col[mask] /= self.scales[i]

    def undo(self, data, colindex=None):
        if colindex is None:
            colindex = slice(None)
        scales = self.scales[colindex]
        biases = self.biases[colindex]
        # Undo the preprocessing on each column of 'data', by scaling the variance back up and adding back the bias 
        ncol = data.shape[1]
        assert len(scales) == ncol
        assert len(biases) == ncol
        for i in range(ncol):
            col = data[:,i:i+1]
            mask = ~np.isnan(col)
            if scales[i]:
                col[mask] *= scales[i]
            col[mask] += biases[i]


    def slice(self, cols):
        other = copy.deepcopy(self)
        other.scales = other.scales[cols]
        other.biases = other.biases[cols]
        return other


class _clamp_extremes_preprocessor(_preprocessor):
    def __init__(self, data, lo, hi):
            
        # Preprocess each column by removing its highest/lowest values according to hi/lo percentiles.
        ncol = data.shape[1]
        for i in range(ncol):
            col = data[:,i:i+1]
            mask = ~np.isnan(col)
            lo_i,hi_i = np.percentile(col[mask], [lo, hi])

            # Convert everything below the "lo" threshold to lo
            tmp = col[mask]
            tmp[tmp < lo_i] = lo_i
            col[mask] = tmp

            # Convert everything above the "hi" threshold to hi
            tmp = col[mask]
            tmp[tmp > hi_i] = hi_i
            col[mask] = tmp

    def apply(self, data):
        return # Do nothing. Should not apply this to new data.

    def undo(self, data):
        return # Do nothing. Can't really undo this particular operation in any meaningful sense -- it only effects things like 'normalize'.


class _logtransform_preprocessor(_preprocessor):
    def __init__(self, data):
        mask = ~np.isnan(data)
        lo = np.min(data[mask])
        assert lo >= 0, "All data must be non-negative in order to apply log transform"
        self.bias = 1. if lo == 0 else 0. # If min is exactly 0, then assume user wants log(1+data) instead of log(data)

    def apply(self, data):
        mask = ~np.isnan(data)
        data[mask] = np.log(data[mask]+self.bias)

    def undo(self, data, colindex=None):
        mask = ~np.isnan(data)
        data[mask] = np.exp(data[mask])-self.bias

class _arcsinhtransform_preprocessor(_preprocessor):
    def __init__(self, data):
        mask = ~np.isnan(data)
        self.intercept = np.median(data[mask])

    def apply(self, data):
        mask = ~np.isnan(data)
        data[mask] = np.arcsinh(data[mask]-self.intercept)

    def undo(self, data, colindex=None):
        mask = ~np.isnan(data)
        data[mask] = np.sinh(data[mask])+self.intercept

###################################################################################

def _is_sequence(string):
    return all((c in "ACGTUNacgtun") for c in string)

def _is_numeric(string):
    return all((c in "+-0123456789.eE") for c in string) or string in ("nan", "NaN", "NAN")

def _dtype_of(colname, colvalue):
    if colname.lower() in ("fold", "foldid", "fold id"):
        return "a1"    # Fold ID column gets a single char identifier
    if colname.lower() in ("event", "eventid", "event id"):
        return "a20"   # Event ID column gets a fixed-length string
    if _is_sequence(colvalue):
        return object  # Sequence columns get a variable-length string (object)
    if _is_numeric(colvalue):
        return "f4"    # Numeric columns get a 32-bit float
    raise ValueError("Could not recognize data type of value \"%s\" in column \"%s\"" % (colvalue, colname))

# Helper function turns string "a.txt[:3]" into a pair ("a.txt", slice(None,3))
def _split_filename(s):
    if s is None:
        return None, None
    match = re.findall('(.+)\[(.+)\]',s)
    if len(match) != 1:
        return s, slice(None)
    filename, colslice = match[0]
    if colslice and ":" in colslice:
        class _slice_getter(object):
            def __getitem__(self, i):
                return i
        colslice = eval("_slice_getter()[%s]" % colslice)
    else:
        colslice = slice(int(colslice), int(colslice)+1)

    return filename, colslice


           
def loadtxt(txtfile, separator=None, usecols=None, maxrows=None, colnames=None):
    """
    Reads txtfile and returns a tuple (colnames, rows) where 
    values is a list with one entry per row. Each row is itself
    a list of strings that were found in the file.
    To convert "rows" into a large numpy array, simply use
       np.asarray(rows, dtype=np.float32)
    """
    if txtfile is None:
        return None, None

    if usecols is None:
        txtfile, usecols = _split_filename(txtfile)

    if not os.path.exists(txtfile):
        raise IOError("Could not open \"%s\"" % txtfile)

    openfunc = gzip.open if txtfile.endswith(".gz") else open
    with openfunc(txtfile,'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        if colnames is None:
            colnames = reader.next()
        if not usecols:
            usecols = range(len(colnames))
        colnames = colnames[usecols] if isinstance(usecols,slice) else [colnames[i] for i in usecols] 
        if maxrows:
            rows = []
            for row in reader:
                rows.append(row[usecols] if isinstance(usecols,slice) else [row[i] for i in usecols])
                if len(rows) >= maxrows:
                    break
        else:
            rows = [row[usecols] if isinstance(usecols,slice) else [row[i] for i in usecols] for row in reader]

    return colnames, rows
