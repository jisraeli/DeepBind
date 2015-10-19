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
import copy
import logging
import tempfile
import smat as sm
import numpy as np
import numpy.random as npr
import deepity
import scipy
import scipy.stats
import cPickle
from ..util import acgt2ord,acgtcomplement
from deepity.util import tic,toc
from os.path import join,basename,splitext,exists

def get_filename_and_cols(s):
    if s is None:
        return None, None

    match = re.findall('(.+)\[(.+)\]',s)
    if len(match) != 1:
        return s, None
    
    filename, colslice = match[0]

    if colslice and ":" in colslice:
        class _slice_getter(object):
            def __getitem__(self, i):
                return i
        colslice = eval("_slice_getter()[%s]" % colslice)
    else:
        colslice = slice(int(colslice), int(colslice)+1)

    return filename, colslice


#########################################################################

class basic_datasource(deepity.resident_datasource):
    """
    A datasource that serves input attributes:
       - X_Sequence0...X_Sequencek: a list of "sequence columns", 
         where each column has the same size, and 
         is provided under the name X_SequenceName (where SequenceName
         was taken from the column header in the sequencefile)
       - F: a single table of features F, taken from the featuresfile.
    and output attributes:
       - Y: the targets, with one column per target
       - Ymask: the mask of non-NaN elements in Y
    """

    def __init__(self, sequencefile, featurefile=None, targetfile=None, foldfilter=None, 
                 requirements=None, maxrows=None, usecols=None, preprocess=None, reverse_complement=False):

        # If sequencefile is matches filename[slice] 
        # then separate the filename itself from the column slice

        self._sequencefile, self._sequencefile_cols = get_filename_and_cols(sequencefile)
        self._featurefile,  self._featurefile_cols  = get_filename_and_cols(featurefile)
        self._targetfile,   self._targetfile_cols   = get_filename_and_cols(targetfile)

        self.sequence_names = self._read_sequencenames()
        self.feature_names  = self._read_featurenames()
        self.target_names   = self._read_targetnames()
        self._task_ids = sorted(self.target_names)
        self._preprocess = preprocess or []
        self._reverse_complement = reverse_complement
        
        if usecols is not None:
            for col in usecols:
                assert col in self.target_names, "Column name \"%s\" does not exist in Targets file header." % col
            self._task_ids = [name for name in self._task_ids if name in usecols]

        nseq = len(self.sequence_names)
        seq_attrs = sum([self._seqattrnames(i) for i in range(nseq)],())
        feat_attr = ("F",) if self._featurefile else ()

        # Initialize the datasource superclass by telling it how many
        # input attributes to expect, based on 
        super(basic_datasource,self).__init__(input_attrs  = seq_attrs + feat_attr,
                                              output_attrs = ("Y","Ymask"),
                                              extra_attrs  = ("rowidx",),  # Attributes not batched or sent to the GPU
                                              )


        # Add some attributes that will be initialized when we load the actual data.
        self._sequences = None
        self._features  = None
        self._features_preprocess = None
        self._targets   = None
        self._targets_preprocess = None
        self.rowidx = None
        self._requirements = requirements or {}
        self._maxrows = maxrows if maxrows else (1L << 40L)
        self._foldfilter = foldfilter if foldfilter is not None else "ABC"

    def _read_sequencenames(self):
        with open(self._sequencefile) as f:
            header   = f.readline().rstrip().split('\t') # read header
            firstrow = f.readline().rstrip().split('\t') # read first non-header line
        assert len(header) >= 3, "Sequences file must have at least 3 columns: \"Fold ID\", \"Event ID\", then at least one sequence column."
        assert header[0] in ("Fold", "FoldID", "Fold ID"), "Sequences file must have first column titled \"Fold ID\"."
        assert header[1] in ("Event", "EventID", "Event ID"), "Sequences file must have second column titled \"Event ID\"."
        assert len(firstrow) == len(header), "Sequences file must have rows with same number of columns as header."
        if self._sequencefile_cols:
            header = header[self._sequencefile_cols]
        return header[2:]

    def _read_featurenames(self):
        if self._featurefile is None:
            return []
        with open(self._featurefile) as f:
            header   = f.readline().rstrip().split('\t') # read header
            firstrow = f.readline().rstrip().split('\t') # read first non-header line
        assert len(header) >= 1, "Targets file must have at least 1 column."
        if self._featurefile_cols:
            header = header[self._featurefile_cols]
        return header

    def _read_targetnames(self):
        if self._targetfile is None:
            return []
        with open(self._targetfile) as f:
            header   = f.readline().rstrip().split('\t') # read header
            firstrow = f.readline().rstrip().split('\t') # read first non-header line
        assert len(header) >= 1, "Targets file must have at least 1 column."
        if self._targetfile_cols:
            header = header[self._targetfile_cols]
        return header
    
    def open(self):
        # Called by a process when it's about to actually start pulling data
        """
        Loads a collection of RNA sequences and protein binding targets.
        Returns a dictionary of deepity.datasource instances, with one entry
        for each fold A,B,C.
        """

        self._sequences    = None
        self._features     = None
        self._features_preprocess = []
        self._targets      = None
        self._targets_mask = None
        self._targets_preprocess = []
        self.rowidx  = None

        self._open_sequences()
        self._open_features()
        self._open_targets()
        self._create_attributes()
        
    def _open_sequences(self):
        """Loads the raw sequences, storing them as lists of strings."""
        
        logging.info("loading %s ..." % basename(self._sequencefile))
        tic()

        # Read entire data file into one big string.
        # Manually scanning the string for newlines and tabs is 3x faster than
        # using readlines() and then calling split() on each line.
        with open(self._sequencefile) as f:
            f.readline() # discard header
            txt = f.read()
            assert txt[-1] == '\n', "Sequence file must end with a newline."

        for name in self.sequence_names:
            logging.info("   %s" % name)

        foldfilter = self._foldfilter
        maxrows = self._maxrows
        seqloop = range(len(self.sequence_names)-1) # Used in innermost loop
        revcomp = self._reverse_complement

        # Store each column as its own list of sequences.
        # Scan through the txt string until we've hit the end.
        sequences = [[] for s in self.sequence_names]
        rowidx    = []
        i,j = 0,txt.find('\n')
        for row_index in xrange(len(txt)):
            if j == -1 or len(rowidx) >= maxrows:
                break
            
            # Check FoldID is wanted (first char of any new line)
            if txt[i] in foldfilter:

                # Add each sequence in this row to its corresponding list
                k = txt.find('\t', i+2) # k = index of first char of first sequence in this row
                for s in seqloop:
                    i, k = k+1, txt.find('\t', k+1)
                    sequences[s].append(txt[i:k])  # Pull out column 's' sequence
                i, k = k+1, txt.find('\t', k+1)
                if k == -1 or k > j:  # If the next tab is on the next line, then break at the newline
                     k = j
                sequences[-1].append(txt[i:k])   # Pull out the last column's sequence

                rowidx.append(row_index)  # Also remember the original row index of this example.

            # Advance so that txt[i:j] is the next line. The last character of the file must be a '\n'.
            i,j = j+1,txt.find('\n',j+1)

        txt = None  # Release memory for gigantic string immediately, for the stability of debugger

        # Convert row indices numpy array for faster indexing when loading features/targets
        self.rowidx = np.asarray(rowidx,np.uint32).reshape((-1,1))
        self._sequences   = sequences

        logging.info("... load took %.2fs" % toc())

    def _open_features(self):
        if self._featurefile is None:
            return

        logging.info("loading %s ..." % basename(self._featurefile))
        tic()

        # Read the entire features file, convert it to numpy array as a string, and slice
        # just the rows that we're using.
        # It turns out this strategy is MUCH faster than using numpy.loadtxt:
        #     features = np.loadtxt(self._featurefile, np.float32, 
        #                           delimiter='\t', skiprows=1, ndmin=2)
        with open(self._featurefile) as f:
            f.readline() # discard header
            txt = f.read()

        for name in self.feature_names:
            logging.info("   %s" % name)

        nfeature = len(self.feature_names)
        rowidx   = self.rowidx
        maxrows_to_read = rowidx[-1]+1

        if self._featurefile_cols:
            # np.fromstring is fast but doesn't support the presence of non-numeric columns
            raise NotImplementedError("This code should work but has not been tested.")
            features = np.asarray([[float(x) for x in line.split('\t')[self._featurefile_cols]]
                                             for line in txt.split('\n',maxrows_to_read)[:-1]])
        else:
            features = np.fromstring(txt, np.float32, sep='\t', 
                                     count=nfeature*maxrows_to_read).reshape(-1, nfeature)
        txt = None

        if len(features) > len(rowidx):
            features = features[rowidx.ravel(),:]

        # Preprocess each feature by normalizing and setting mean to 0
        a,b = [],[]
        for i in range(nfeature):
            col = features[:,i:i+1]
            mask = ~np.isnan(col)
            lo = np.min(col[mask], axis=0)  
            hi = np.max(col[mask], axis=0)
            if lo == hi:
                hi += 1 # Avoid divide by zero for degenerate targets
            meani = np.mean(col[mask])

            ai = 1./(hi-lo)
            bi = -meani*ai

            col[mask] = ai*col[mask] + bi
            a.append(ai)
            b.append(bi)

        self._feature_preprocess = [ ('normalize', np.asarray(a).reshape((1,-1)), 
                                                   np.asarray(b).reshape((1,-1))) ]

        nsequence = len(self._sequences[0])
        assert len(features) == nsequence, "Number of rows in Features file must match number of rows in Sequences file."
        self._features = features
        
        logging.info("... load took %.2fs" % toc())

    def _open_targets(self):
        if self._targetfile is None:
            return

        _log_xform_warned = False
        logging.info("loading %s ..." % basename(self._targetfile))
        tic()

        # Read the entire targets file, convert it to numpy array as a string, and slice
        # just the rows that we're using.
        # It turns out this strategy is MUCH faster than using numpy.loadtxt:
        #     features = np.loadtxt(self._featurefile, np.float32, 
        #                           delimiter='\t', skiprows=1, ndmin=2)
        with open(self._targetfile) as f:
            f.readline() # discard header
            txt = f.read()
        
        ntarget = len(self.target_names)
        ntask   = len(self._task_ids)
        rowidx  = self.rowidx
        maxrows_to_read = rowidx[-1]+1

        if self._targetfile_cols:
            # np.fromstring is fast but doesn't support the presence of non-numeric columns
            targets = np.asarray([[float(x) for x in line.split('\t')[self._targetfile_cols]]
                                            for line in txt.split('\n',maxrows_to_read)[:-1]])
        else:
            targets = np.fromstring(txt, np.float32, sep='\t', 
                                    count=ntarget*maxrows_to_read).reshape(-1, ntarget)
        txt = None

        if len(targets) > len(rowidx):
            targets = targets[rowidx.ravel(),:]

        # Select columns using '_task_ids' no matter what, since the order
        # might be different.
        usecols = np.asarray([self.target_names.index(name) for name in self._task_ids])  # nparray for faster indexing in 
        targets = targets[:,usecols]

        # Normalize targets by scaling min/max range to [0,1]
        if targets.size > 0:
            # OPTIONAL: clamp all originally negative values at zero
            #targets = np.maximum(0, targets)

            # For each individual column, get lo/hi percentile 
            # and then normalize the non-NaN values in that column
            a,b = [],[]
            for i in range(ntask):
                target_i = targets[:,i]
                mask_i = ~np.isnan(target_i)

                is_boolean = np.all(np.logical_or(target_i[mask_i] == 0, target_i[mask_i] == 1))
                if is_boolean:
                    # Automatically assume 0/1 classification target
                    logging.info("   %s \t(classification)" % self._task_ids[i])
                    ai,bi = 1,0
                else:
                    # Automatically assume regression target
                    logging.info("   %s \t(regression)" % self._task_ids[i])

                    if "log" in self._preprocess:
                        if (not np.all(target_i[mask_i] >= 0)):
                            if not _log_xform_warned:
                                _log_xform_warned = True
                                print "Warning: log transform requires all original targets to be non-negative; biasing the data and proceeding anyway."
                            target_i[mask_i] -= target_i[mask_i].min()
                        target_i[mask_i] = np.log(1+target_i[mask_i])
                    elif "sqrt" in self._preprocess:
                        if (not np.all(target_i[mask_i] >= 0)):
                            if not _log_xform_warned:
                                _log_xform_warned = True
                                print "Warning: sqrt transform requires all original targets to be non-negative; biasing the data and proceeding anyway."
                            target_i[mask_i] -= target_i[mask_i].min()
                        target_i[mask_i] = np.sqrt(target_i[mask_i])

                    lo_i,hi_i = np.percentile(target_i[mask_i], [0.0, 1.0])
                    #lo_i,hi_i = np.percentile(target_i[mask_i], [0.05, 99.95])
                    if lo_i == hi_i:
                        hi_i += 1 # Avoid divide by zero for degenerate targets

                    # Convert everything below the "lo" threshold to NaNs
                    tmp = target_i[mask_i]
                    tmp[tmp < lo_i] = np.nan
                    target_i[mask_i] = tmp
                    mask_i = ~np.isnan(target_i)

                    # Convert everything above the "hi" threshold to NaNs
                    tmp = target_i[mask_i]
                    tmp[tmp > hi_i] = np.nan
                    target_i[mask_i] = tmp
                    mask_i = ~np.isnan(target_i)

                    # Clamp everything to the range [lo,hi]
                    #target_i[mask_i] = np.maximum(lo_i, target_i[mask_i])
                    #target_i[mask_i] = np.minimum(hi_i, target_i[mask_i])  # Assume anything above hi_i is a "large" outlier

                    # Subtract the mean
                    if self._requirements.get('target',None) == 'logistic':
                        intercept_i = lo_i
                    else:
                        intercept_i = np.mean(target_i[mask_i])
                    ai = 1./(hi_i-lo_i)
                    bi = -intercept_i*ai
                    target_i[mask_i] = ai*target_i[mask_i] + bi
                    #mask_pos = target_i[mask_i] > 0
                    #target_i[mask_i][mask_pos] **= 0.5
                a.append(ai)
                b.append(bi)

            if "log" in self._preprocess:
                self._targets_preprocess.append(('log',))
            self._targets_preprocess.append(('normalize', np.asarray(a).reshape((1,-1)), 
                                                          np.asarray(b).reshape((1,-1))) )

            #targets[self._targets_mask] = np.maximum(0,targets[self._targets_mask])
            #targets[self._targets_mask] = np.minimum(1,targets[self._targets_mask])

        self._targets = targets
        self._targets_mask = ~np.isnan(targets)


        logging.info("... load took %.2fs" % toc())

    def _create_attributes(self):
        # Adds public attributes with names matching 
        for i in range(len(self._sequences)):
            Xname,Rname = self._seqattrnames(i)
            self.__dict__[Xname] = self._sequences[i]
            self.__dict__[Rname] = np.zeros((len(self._sequences[i]),1), np.uint32) # empty until set during asbatches()
        if self._features is not None:
            self.__dict__['F'] = self._features
        if self._targets is not None:
            self.__dict__['Y'] = self._targets
            self.__dict__['Ymask'] = self._targets_mask

    def _seqattrnames(self,index):
        return ('X_%s'%self.sequence_names[index], 'R_%s'%self.sequence_names[index])

    def __len__(self):
        return len(self.rowidx)

    def load_preprocessing(self, indir):
        with open(join(outdir, 'features_preprocess.pkl'), 'w') as f:
            ppsliced = []

    def dump_preprocessing(self, outdir, cols=None):
        if self._features_preprocess:
            if cols is None:
                cols = slice(None)
            with open(join(outdir, 'features_preprocess.pkl'), 'w') as f:
                ppsliced = []
                for pp in self._features_preprocess:
                    if pp[0] == 'normalize':
                        ppsliced.append((pp[0], pp[1][cols], pp[2][cols]))
                cPickle.dump(ppsliced, f)

    def _apply_reverse_complement(self):
        if not self._reverse_complement:
            return

        nseq = len(self.sequence_names)
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
        if hasattr(self, "F"):     self.F      = self.F[duprows,:]
        if hasattr(self, "Y"):     self.Y      = self.Y[duprows,:]
        if hasattr(self, "Ymask"): self.Ymask  = self.Ymask[duprows,:]


    def asbatches(self, batchsize=128, reshuffle=True):
        n = len(self)
        assert n > 0
        nbatch = (n + batchsize - 1) // batchsize
        nseq = len(self.sequence_names)
        padding = self._requirements['padding']
        batches = []
        for i in range(nbatch):

            # Slice a our data attributes row-wise, according to batch index
            batch = self[np.arange(i*batchsize,min(n,(i+1)*batchsize))]
            batch._apply_reverse_complement()

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

            if hasattr(batch,"F") and batch.F is not None: 
                batch.F = sm.asarray(batch.F,sm.get_default_dtype())
            if hasattr(batch,"Y") and batch.Y is not None:
                batch.Y = sm.asarray(batch.Y,sm.get_default_dtype())
                if isinstance(batch.Ymask,np.ndarray):
                    batch.Ymask = sm.asarray(batch.Ymask)

            batches.append(batch)
        return deepity.shuffled_repeat_iter(batches, reshuffle)

