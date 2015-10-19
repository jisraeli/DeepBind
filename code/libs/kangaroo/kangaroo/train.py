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
import glob
import time
import logging
import numpy as np
import shutil
import deepity
import deepity.hpsearch
from . import util
from . import globals

def getworkdir(outdir, targetname):
    dir = "%s/%s" % (outdir, targetname)
    return dir

def load_calib(filename):
    return deepity.load_hparams_result(filename)

def save_calib(filename, calib):
    deepity.save_hparams_result(filename, calib)

def load_calib_samples(filename):
    # Save one calibration sample per line
    samples = []
    with open(filename) as f:
        for line in f:
            args = eval(line,{ "nan" : np.nan })
            samples.append(deepity.hpsearch.sample(args[1], args[0]))
    return samples

def save_calib_samples(filename, mode, samples):
    # Save one calibration sample per line
    util.makepath(os.path.dirname(filename))
    with open(filename, mode) as f:
        for sample in samples:
            f.write(str([sample.metrics, sample.params])+"\n")

def calibrate(cfgs, data, outdir, nfold=1, allfolds=True, ncalib=10, auxfilter=None, append=False):
    globals._set_default_logging(outdir)
    if not append:
        for targetname in data.targetnames:
            workdir = getworkdir(outdir, targetname)
            if os.path.exists(workdir+"/calib.all.txt"):
                os.remove(workdir+"/calib.all.txt")

    best_hparams = {}
    for cfgname, cfg in cfgs.iteritems():
        samples = deepity.hypertrain(cfg["model"], 
                                     cfg["trainer"], 
                                     data,
                                     outdir=outdir,
                                     nfold=nfold, 
                                     allfolds=allfolds,
                                     nsample=ncalib/len(cfgs),
                                     devices=globals._devices,
                                     auxfilter=auxfilter,
                                     )
        for targetname in data.targetnames:
            workdir = getworkdir(outdir, targetname)
            target_samples = samples.get(targetname, [])  # Default is for case when model/trainer configs contained no hyperparameters at all!
            for sample in target_samples:
                sample.params[":cfgname"] = cfgname
            save_calib_samples(workdir+"/calib.all.txt", "a", target_samples)
            '''
            # Remember which hparams was best
            if (targetname not in best_hparams) or (target_hparams.result < best_hparams[targetname].result):
                best_hparams[targetname] = target_hparams
                best_hparams[targetname].params[":cfgname"] = cfgname
                
    for targetname in data.targetnames:
        workdir = getworkdir(outdir, targetname)
        save_calib(workdir, best_hparams[targetname])'''


def train(cfgs, data, calibdir, outdir, nfold=1, ntrial=1, auxfilter=None, metric_key="loss"):
    globals._set_default_logging(outdir)

    for targetname in sorted(data.targetnames):
        calib_workdir = getworkdir(calibdir, targetname)
        samples = load_calib_samples(calib_workdir+"/calib.all.txt")

        # Get the best sample for this specific model
        #   cfgbest = deepity.hpsearch.get_best_sample([_ for _ in samples if _.params[":cfgname"] == cfgname], "loss")
        cfgbest = deepity.hpsearch.get_best_sample(samples, metric_key, wantmax="loss" not in metric_key)
        cfgname = cfgbest.params[":cfgname"]
        cfg = cfgs[cfgname]

        outpattern = [outdir, ("target","%s")]
        outpattern += [("trial", "trial%s")]
        if nfold > 1:
            outpattern += [("fold","/fold%s")]
        
        deepity.train(cfg["model"], cfg["trainer"], 
                        data.astargets([targetname]), 
                        hparams={targetname : cfgbest}, hparams_metric=metric_key, 
                        outdir=outpattern,
                        nfold=nfold, 
                        nsample=ntrial,
                        devices=globals._devices,
                        auxfilter=auxfilter,
                        dumpviz=False,
                        )

        # Collect the performance of each trial
        performances = []
        for trial in range(ntrial):
            instdir = deepity.getinstdir(outpattern, targetname, trial, None)
            with open(instdir+"/metrics.txt") as f:
                header = f.readline().rstrip().split() # discard column headers
                for line in f:
                    line = line.rstrip().split()
                    metricname, trainvalue = line[:2]
                    if metricname == metric_key:
                        performances.append(float(trainvalue))
                        break
    
        # Find the trial with best performance
        besttrial = np.argmin(performances) if "loss" in metric_key else np.argmax(performances)
        print "trial metrics:", performances

        # Copy the best trial into the parent directory, and delete all other trials 
        # to save space and to not drive btsync so crazy.
        instdir = deepity.getinstdir(outpattern, targetname, besttrial, None)
        files = glob.glob(instdir+"/*")
        for file in files:
            dst = os.path.dirname(os.path.dirname(file))+"/"+os.path.basename(file)
            if os.path.isdir(file):
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(file, dst)
            else:
                shutil.copyfile(file, dst)
        time.sleep(0.1) # rmtree sometimes fails if the folder is scanned by btsync; this seems to help a bit
        for i in range(len(performances)):
            shutil.rmtree(deepity.getinstdir(outpattern, targetname, i, None), ignore_errors=True)
        deepity.call_dumpviz(os.path.dirname(instdir))
       