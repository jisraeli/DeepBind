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
import sys
import re
import os
import os.path
import argparse
import numpy as np
import numpy.random as npr
import deepbind_util as util

import cPickle
import scipy
import shutil

# Warning: The multiprocessing module may be used indirectly, so do not put any 
# unintentional statements outside of main().

datadir = "../data/encode"
seq_suffix = ".seq.gz"

def main():
    util.enable_reversecomplement()

    args   = loadargs()
    models = loadmodels(args)
    trdata = None
    tedata = None
    tfids  = load_tfids(args)

    for tfid in tfids:
        if "calib" in args.steps:
            print "-------------- calib:", tfid, "--------------"
            trdata = load_traindata(tfid, args)
            util.calibrate(models, trdata, args.calibdir, nfold=args.nfold, ncalib=args.ncalib)

        if "train" in args.steps:
            print "-------------- train:", tfid, "--------------"
            trdata = load_traindata(tfid, args)
            util.train(models, trdata, args.calibdir, args.finaldir, nfold=1,          ntrial=args.ntrial)

    if "test" in args.steps:
        tedata = load_testdata(tedata, tfids, args)
        util.save_metrics(tedata, "test", args.finaldir)

    if "report" in args.steps:
        tedata = load_testdata(tedata, tfids, args)
        util.save_featuremaps(tedata, args.finaldir, args.reportdir, maxrows=100000)
        util.save_report(args.finaldir, args.reportdir, tfids)


#########################################################################

def load_tfids(args):
    targetnames = sorted([filename.replace("_AC"+seq_suffix,"") for filename in os.listdir(datadir) if filename.endswith("_AC"+seq_suffix)])
    chunktargets = util.getchunktargets(args, targetnames)
    return chunktargets

#################################

def loadargs():
    args = argparse.ArgumentParser(description="Generate the ENCODE experiments.")
    args.add_argument("mode", type=str, help="Either \"top\" (train top 500 odd, test top 500 even) or \"all\" (train top 500 even + all after 1000, test top 500 even).")
    args = util.parseargs("encode", args)
    args.calibdir  = args.calibdir.replace(args.outdir, args.outdir+"/"+args.mode)
    args.finaldir  = args.finaldir.replace(args.outdir, args.outdir+"/"+args.mode)
    args.reportdir = args.reportdir.replace(args.outdir,args.outdir+"/"+args.mode)
    args.outdir    = args.outdir+"/"+args.mode
    return args

#################################

def loadmodels(args):
    models = util.loadmodels(args, "cfg/classification")
    if args.mode == "top":
        for name in models.keys():
            models[name]['trainer'].max_steps = 10000  # training on top 500 never seems to need more than 10000 steps, since it's a small training set
    for cfg in models.itervalues():
        cfg["model"].conv_seq[0].fsize = 24     # Override default max motif length
    return models

#################################

def load_traindata(tfid, args):
    print "load_traindata: %s"%tfid
    maxrows = 10000 if args.quick else None
    minrows = 100000
    trdata = util.load_seq("%s/%s_AC%s" % (datadir,tfid,seq_suffix), minrows=minrows, maxrows=maxrows)
    trdata.targetnames = [tfid]
    if args.mode == "top":
        trdata = trdata[:500] # Only top 500 even peaks (A); top 500 odd (B) are stored in the corresponding _B.seq.gz file
    elif args.mode == "all":
        pass # Top 500 even (A) plus everything else (C)
    else:
        quit("Unrecognized mode \"%s\". Expected \"top\" or \"all\".")
    return trdata

#################################

def load_testdata(tedata, tfids, args):
    if tedata is not None:
        return tedata
    if "encode" in datadir:
        maxrows = 10000
    elif "chip" in datadir:
        maxrows = 10000
    else:
        maxrows = None
    all_tedata = {}
    for tfid in tfids:
        print "load_testdata: %s ..."%tfid,
        tedata = util.load_seq("%s/%s_B%s" % (datadir,tfid,seq_suffix), minrows=10000, maxrows=maxrows)
        tedata.targetnames = [tfid]
        all_tedata[tfid] = tedata
        print "done"
    return all_tedata


if __name__=="__main__":
    #util.disable_multiprocessing()
    main()



