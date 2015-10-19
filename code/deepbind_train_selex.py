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

datadir = "../data/selex"

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
            set_motif_lengths(args, models, tfid)
            trdata = load_traindata(tfid, args)
            util.calibrate(models, trdata, args.calibdir, nfold=args.nfold, ncalib=args.ncalib, allfolds=False)

        if "train" in args.steps:
            print "-------------- train:", tfid, "--------------"
            set_motif_lengths(args, models, tfid)
            trdata = load_traindata(tfid, args)
            util.train(models, trdata, args.calibdir, args.finaldir, nfold=1,          ntrial=args.ntrial)

    if "test" in args.steps:
        tedata = load_testdata(tedata, tfids, args)
        util.save_metrics(tedata, "test", args.finaldir)

    if "report" in args.steps:
        tedata = load_testdata(tedata, tfids, args)
        util.save_featuremaps(tedata, args.finaldir, args.reportdir)
        util.save_report(args.finaldir, args.reportdir, tfids)


#########################################################################

def load_tfids(args):
    targetnames = sorted(list(set([filename.split(".")[0].rsplit("_",1)[0] for filename in os.listdir(datadir) if not os.path.isdir(datadir+"/"+filename)])))
    chunktargets = util.getchunktargets(args, targetnames)
    return chunktargets

#################################

def loadargs():
    global datadir
    args = argparse.ArgumentParser(description="Generate the HT-SELEX experiments.")
    args.add_argument("--jolma", action="store_true", default=False, help="Use this flag to train on the HT-SELEX cycles selected by Jolma et al., rather than the cycles selected by Alipanahi et al.")
    args = util.parseargs("selex", args)
    if args.jolma:
        args.calibdir  = args.outdir+"/jolma/calib"
        args.finaldir  = args.outdir+"/jolma/final"
        args.reportdir = args.outdir+"/jolma/report"
        args.testdir   = args.outdir+"/jolma/test"
        datadir = datadir + "/jolma"
    else:
        args.calibdir  = args.outdir+"/best/calib"
        args.finaldir  = args.outdir+"/best/final"
        args.reportdir = args.outdir+"/best/report"
        args.testdir   = args.outdir+"/best/test"
        datadir = datadir + "/best"
    return args

#################################

def loadmodels(args):
    models = util.loadmodels(args, "cfg/classification")
    return models

#################################

def set_motif_lengths(args, models, tfid):
    seqpatt = tfid.split("_")[-3]
    ligandlen = re.findall("(\d+)N", seqpatt)
    if not ligandlen:
        quit("Could not parse ligand length from tfid %s" % tfid)
    ligandlen = int(ligandlen[0])
    filterlens = { 14 : 14, 
                   20 : 20, 
                   30 : 24,
                   40 : 32, }
    filterlen = filterlens[ligandlen]
    print "Using filter size %d" % filterlen
    for cfg in models.itervalues():
        cfg["model"].conv_seq[0].fsize = filterlen


#################################

def load_traindata(tfid, args):
    print "load_traindata: %s"%tfid
    maxrows = 10000 if args.quick else None
    trdata = util.load_seq("%s/%s_A.seq.gz" % (datadir,tfid), minrows=100000, maxrows=maxrows)
    trdata.targetnames = [tfid]
    return trdata

#################################

def load_testdata(tedata, tfids, args, maxrows = None):
    if tedata is not None:
        return tedata
    if maxrows is None:
        maxrows = 10000 if args.quick else None
    tedata = {}
    for tfid in tfids:
        print "load_testdata: %s ..."%tfid,
        tedata[tfid] = util.datasource.fromtxt("%s/%s_B.seq.gz" % (datadir,tfid), None, None, maxrows=maxrows)
        tedata[tfid].targetnames = [tfid]
        print "done"
    return tedata

#################################




if __name__=="__main__":
    #util.disable_multiprocessing()
    main()



