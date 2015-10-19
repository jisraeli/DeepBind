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
import os.path
import copy
import time
import numpy as np
import numpy.random as npr
import smat as sm
import logging
import itertools
import deepity
import deepity.hpsearch
from . import _ext
from . import util
from . import globals
from .train import train, calibrate
from .predict import predict, load_modelinfos, load_model
import multiprocessing

class filtersampler(deepity.hpsearch.paramdef):
    def __init__(self, filters, samplesize, name=None):
        assert samplesize <= filters.shape[1]
        # Take only columns (filters) where the max weight magnitude is 
        # at least 0.05 times the max weight magnitude of the whole filterbank.
        colmax = np.max(abs(filters), axis=0)
        allmax = colmax.max()
        colmask = colmax >= allmax*0.05
        while sum(colmask) < samplesize:
            colmask[npr.randint(0,colmask.size)] = True

        self.samplesize = samplesize
        self.filters = filters[:, colmask].copy()
        #self.filters = npr.randn(self.filters.shape[0],self.filters.shape[1])*1.1
        self.subsets = list(itertools.combinations(range(self.filters.shape[1]), samplesize))
        
        super(filtersampler,self).__init__(name, filters.dtype)

    def sample(self, task_idx, sample_ids):
        return np.asarray([self.filters[:,self.subsets[npr.randint(0,len(self.subsets))]] 
                           for id in sample_ids])
    

def _bind_filterinit_hyperparam(path, obj, model):
    if not isinstance(obj, _ext.corr1ord):
        return

    convnode = getattr(model, "conv_" + obj.parent.name)[0]
    obj.init = filtersampler(convnode.W.fpval.asnumpy(), obj.nfilter)
    obj.fsize = convnode.fsize

def _bind_filterinit_hyperparam_main(args):
    simple_model, modelinfo = args
    model = load_model(modelinfo)
    simple_model.visit(lambda path, obj: _bind_filterinit_hyperparam(path, obj, model))
    return simple_model

def bind_filterinit_hyperparams(simple_model, modelinfo):
    # Have to load the model and do substitutions in a subprocess because otherwise a CUDA context will get created
    # in the MainProcess, which will screw everything up on the next fork() that follows
    pool = multiprocessing.Pool(1)
    bound_model = pool.map(_bind_filterinit_hyperparam_main, [(simple_model, modelinfo)])[0]
    pool.close()
    pool.join()
    return bound_model


def simplify(simple_model, trainer, data, modeldir, ncalibration=18, calibration_steps=None, nfold=2, nsample=1, outdir=None):
    # srcdir is the directory of previously trained models
    if not outdir:
        outdir = "out"
    globals._set_default_logging(outdir)

    # Generate predictions for each full model on the given input sequences.
    predictions = predict(modeldir, data, outdir=modeldir)

    # For each full model, train several simplified models and take the best
    for modelname, modelinfo in load_modelinfos(modeldir, include=data.targetnames).iteritems():

        # simple_model will be trained on *predictions* of the input data,
        # so here we replace the targets of that data with predictions
        simple_data = copy.copy(data)
        simple_data.targetnames = [modelname]
        simple_data.targets = predictions[modelname].copy()
        simple_data.Y       = predictions[modelname].copy()
        simple_data.Ymask[:] = True

        # Use the 'calibration' phase to select hyperparameters, where we
        # have made "which subset of filters to initialize with" one of 
        # the hyperparameters
        bound_model = bind_filterinit_hyperparams(simple_model, modelinfo)
        calibrate_trainer = copy.copy(trainer)
        if calibration_steps is not None:
            calibrate_trainer.max_steps = calibration_steps
        calibration = calibrate(bound_model, calibrate_trainer, simple_data, outdir=outdir, nfold=nfold, ncalibration=ncalibration)

        # Train final model
        train(bound_model, trainer, simple_data, calibration, outdir=outdir, nsample=nsample, nfold=nfold)

    return