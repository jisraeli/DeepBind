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
import os
import re
import os.path
sys.path.append("libs/kangaroo")
sys.path.append("libs/deepity")
sys.path.append("libs/smat/py")
import math
import time
import glob
import shutil
import numpy as np
import numpy.random as npr
import random
import kangaroo
import deepity
import itertools
import argparse
import cPickle
import smat
import scipy
import scipy.stats

datasource = kangaroo.datasource
train = kangaroo.train
loadcfg = kangaroo.loadcfg
calibrate = kangaroo.calibrate
acgt2ord = kangaroo.util.acgt2ord
ord2acgt = kangaroo.util.ord2acgt
revcomp = kangaroo.util.revcomp
makepath = kangaroo.util.makepath
predict = kangaroo.predict
calc_metrics = deepity.calc_metrics
globals = kangaroo.globals
load_metrics = deepity.load_metrics
save_gradientmaps = kangaroo.save_gradientmaps

def set_devices(devices):
    kangaroo.globals.set_devices(devices)


##################### generate PFMs ######################


def ord2mask_b(x):
    """
    Convert a vector of length N with integral values in range {0,1,2,3} 
    into an Nx4 numpy array, where for example "2" is represented by 
    row [0,0,1,0].
    """    
    mask = np.zeros((x.size,4))
    for i in range(x.size):
        if   x[0,i] <= 3:
            mask[i,x[0,i]] = 1
        elif x[0,i] == 255:     
            mask[i,:] = 0.25*np.ones((4,))
    return mask

def compute_pfms(data, filter_len=20, num_permut=1000, rev_comp=True):
    nf, num = data.shape    
    pfms = [0.0001*np.ones((filter_len,4)) for i in range(nf)]
    counts = [0 for _ in range(nf)]
    for i in xrange(num):
        seq = ord2mask_b(acgt2ord('N'*(filter_len-1) + data[0,i][0] + 'N'*(filter_len-1)))
        for j in range(nf):
            if data[j,i][1] <= 0:
                continue
            max_ind = data[j,i][2]
            idx_st = max_ind
            idx_sp = max_ind + filter_len
            pfms[j] += seq[idx_st:idx_sp,:]
            counts[j] += 1
    info = np.zeros((filter_len, nf))
    ic = np.zeros((nf,))
    for j in range(nf):
        for k in range(pfms[j].shape[0]):
            pfms[j][k,:] /= np.sum(pfms[j][k,:])
            info[k,j] =  2 + np.sum(pfms[j][k,:] * np.log2(pfms[j][k,:]))
        ic[j] = np.sum(info[:,j].ravel())
        pfm_st, pfm_sp = trim_pfm(info[:,j].ravel())
        pfms[j] = pfms[j][pfm_st:pfm_sp,:]
        
        ic[j] = 0
        for k in range(pfms[j].shape[0]):            
            ic[j] +=  2 + np.sum(pfms[j][k,:] * np.log2(pfms[j][k,:]))         
    kl_dist, pval = pfm_kl_dist(pfms, ic, num_permut, rev_comp)
    return pfms, ic, kl_dist, pval, counts + [num]

def trim_pfm(info):        
    max_ic = np.max(info)
    ic_threshold = np.max([0.1*max_ic, 0.1]);
    w = info.shape[0]
    pfm_st = 0
    pfm_sp = w
    for i in range(w):
        if info[i] < ic_threshold:
            pfm_st += 1
        else:
            break
    for inf in reversed(info):    
        if inf < ic_threshold:            
            pfm_sp -= 1            
        else:
            break
    return pfm_st, pfm_sp  

def rev_com_mask(p):
    q = p.copy()
    q = q[::-1,:]
    q = q[:,::-1]
    return q

def kldist(p, q):
    return np.sum(p*np.log(p/q))
    
    #r, c = p.shape
    #kl = 0
    #for i in range(r):
    #    for j in range(c):
    #        kl += p[i,j]*np.log(p[i,j]/q[i,j])
    #return kl      

def compute_kl(p, q):
    lp = p.shape[0]
    lq = q.shape[0]
    padding = 0.25*np.ones((lp-1,4))
    q = np.vstack((padding, q, padding))
    dist = np.inf
    for k in range(lq+lp-1):
        tq = q[k:k+lp,:]
        tkl = kldist(p, tq)
        if tkl < dist:
            dist = tkl            
    return dist/lp

def pfm_kl_dist(pfms, ic, num_permut, rev_comp):
    nf = len(pfms)
    dist = np.zeros((nf, nf))
    pval = np.ones((nf, nf))
    for i,vp in enumerate(pfms):
        for j,vq in enumerate(pfms):
            p = vp.copy()
            q = vq.copy()          
            if j <= i:
                continue
            if ic[i] * ic[j] == 0:
                dist[i,j] = -1
                dist[j,i] = -1
                continue
            if q.shape[0] < p.shape[0]:
                q, p = p, q
            dist[i,j] = compute_kl(p, q)
            if rev_comp:
                q_rev_com = rev_com_mask(q)
                rev_dist = compute_kl(p, q_rev_com)
                if rev_dist < dist[i,j]:
                    dist[i,j] = rev_dist
            if num_permut > 0:          
                for _ in range(num_permut):
                    tp = p.copy()
                    p_row = np.random.permutation(tp.shape[0])
                    p_col = np.random.permutation(tp.shape[1])
                    tp = tp[p_row,:]
                    tp = tp[:,p_col]
                    if rev_comp:
                        p_dist = np.min([compute_kl(tp, q), compute_kl(tp, q_rev_com)])
                    else:
                        p_dist = compute_kl(tp, q)
                    if p_dist <= dist[i,j]:
                        pval[i,j] += 1             
                pval[i,j] /= num_permut
                pval[j,i] = pval[i,j]   
            dist[j,i] = dist[i,j]
    return dist, pval


####################### doublet_shuffle ###############################


# This implementation is based on Altschul and Erickson's Algorithm
# for shuffling a sequence while preserving the doublet frequency 
# Babak Alipanahi
# Jan 24, 2014
# University of Toronto

# form the graph from sequence
def form_seq_graph(seq):
    graph = {}
    for i, s in enumerate(seq[:-1]):        
        if s not in graph:
            graph[s] = []
        graph[s].append(seq[i+1])     
    return graph

# sample a random last edge graph
def sample_le_graph(graph, last_nt):
    le_graph = {}
    for vx in graph:
        le_graph[vx] = []
        if vx not in last_nt:
            le_graph[vx].append(random.choice(graph[vx]))
    return le_graph        

# check whether there exists an Eulerian walk
# from seq[0] to seq[-1] in the shuffled
# sequence
def check_le_graph(le_graph, last_nt):
    for vx in le_graph:
        if vx not in last_nt:
            if not find_path(le_graph, vx, last_nt):
                return False
    return True            

# function from: http://www.python.org/doc/essays/graphs/
# check whether there is a path between two nodes in a 
# graph
def find_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    if not graph.has_key(start):
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath: return newpath
        return None       
        
# generate a new seq graph based on the last edge graph
# while randomly permuting all other edges        
def form_new_graph(graph, le_graph, last_nt):
    new_graph = {}
    for vx in graph:
        new_graph[vx] = []
        temp_edges = graph[vx]
        if vx not in last_nt:
            temp_edges.remove(le_graph[vx][0])
        random.shuffle(temp_edges)        
        for ux in temp_edges:
            new_graph[vx].append(ux)
        if vx not in last_nt:    
            new_graph[vx].append(le_graph[vx][0])
    return new_graph                     
      
# walk through the shuffled graph and make the
# new sequence       
def form_shuffled_seq(new_graph, init_nt, len_seq):
    is_done = False
    new_seq = init_nt
    while not is_done:
        last_nt  = new_seq[-1]
        new_seq += new_graph[last_nt][0]
        new_graph[last_nt].pop(0)
        if len(new_seq) >= len_seq:
            is_done = True
    return new_seq    

# verify the nucl
def verify_counts(seq, shuf_seq):
    kmers = {}
    # Forming the k-mer library
    kmer_range = range(1,3)
    for k in kmer_range:
        for tk in itert.product('ACGTN', repeat=k):
            tkey = ''.join(i for i in tk)
            kmers[tkey] = [0,0]
                        
    kmers[seq[0]][0] = 1
    kmers[shuf_seq[0]][1] = 1      
    for k in kmer_range:
        for l in range(len(seq)-k+1):
            tkey = seq[l:l+k]
            kmers[tkey][0] += 1
            tkey = shuf_seq[l:l+k]
            kmers[tkey][1] += 1
    for tk in kmers:
        if kmers[tk][0] != kmers[tk][1]:
            return False    
    return True

_preprocess_seq = ['N']*256;
_preprocess_seq[ord('a')] = _preprocess_seq[ord('A')] = 'A';  # Map A => A
_preprocess_seq[ord('c')] = _preprocess_seq[ord('C')] = 'C';  # Map C => C
_preprocess_seq[ord('g')] = _preprocess_seq[ord('G')] = 'G';  # Map G => G
_preprocess_seq[ord('t')] = _preprocess_seq[ord('T')] = 'T';  # Map T => T
_preprocess_seq[ord('u')] = _preprocess_seq[ord('U')] = 'T';  # Map U => T
_preprocess_seq = "".join(_preprocess_seq)
            
def preprocess_seq(seq):
    return seq.translate(_preprocess_seq) 

def doublet_shuffle(seq, verify=False):
    seq = preprocess_seq(seq)
    last_nt = seq[-1]
    graph = form_seq_graph(seq)
    # sample a random last edge graph
    is_ok = False
    while not is_ok:
        le_graph = sample_le_graph(graph, last_nt)
        # check the last edge graph
        is_ok = check_le_graph(le_graph, last_nt)
    new_graph = form_new_graph(graph, le_graph, last_nt)
    shuf_seq  = form_shuffled_seq(new_graph, seq[0], len(seq))
    if verify:
        assert(verify_counts(seq, shuf_seq))
    return shuf_seq


def kmer_former(seq, kmerlen):
    kmers = {}
    for j in range(len(seq) - kmerlen + 1):
        tkey = seq[j:j+kmerlen]
        if tkey not in kmers:
            kmers[tkey] = True
    return kmers

def verify_kmer(shuf_seq, kmers, kmerlen):
    for j in range(len(shuf_seq) - kmerlen + 1):
        tkey = shuf_seq[j:j+kmerlen]
        if tkey in kmers:
            return False
    return True



######################## AUC ######################

def calc_tpc_fpc_curve(z, y):
    assert len(z) == len(y)

    # Sort by decreasing score
    order = np.argsort(z, axis=0, kind="mergesort")[::-1].ravel()
    z = z[order]
    y = y[order]

    # Accumulate the true positives with decreasing threshold
    tpc = y.cumsum()
    fpc = 1 + np.arange(len(y)).astype(y.dtype) - tpc
    return tpc, fpc


def calc_auc(z, y, want_curve=False):

    tpc, fpc = calc_tpc_fpc_curve(z, y)

    # If curve doesn't have point at x=0, explicitly insert one
    if fpc[0] != 0:
        tpc = np.r_[0,tpc]
        fpc = np.r_[0,fpc]

    # If one of the classes was empty, return NaN
    if fpc[-1] == 0 or tpc[-1] == 0:
        return np.nan

    # Convert sums to rates
    tpr = tpc / tpc[-1]
    fpr = fpc / fpc[-1]

    # Calculate area under the curve using trapezoidal rule
    auc = np.trapz(tpr, fpr, axis=0)

    if want_curve:
        curve = np.hstack([fpr.reshape((-1,1)),tpr.reshape((-1,1))])
        return auc,curve
    return auc


def calc_pr_curve(z, y):
    tpc, fpc = calc_tpc_fpc_curve(z, y)

    assert not (fpc[-1] == 0 or tpc[-1] == 0)

    precision = tpc / (tpc + fpc)
    recall = tpc / tpc[-1]

    curve = np.hstack([recall.reshape((-1,1)),precision.reshape((-1,1))])
    return curve



def bootstrap_auc(z, y, ntrial=100):
    n = len(y)
    aucs = []
    for t in range(ntrial):
        sample = npr.randint(0,n,n)
        zt = z[sample].copy().reshape((-1,1))
        yt = y[sample].copy().reshape((-1,1))
        auc = calc_auc(zt,yt)
        if np.isnan(auc):
            return np.nan, np.nan  # If any of the samples returned NaN, the whole bootstrap result is NaN
        aucs.append(auc)
    return np.mean(aucs),np.std(aucs)


##################### IUPAC translation ####################


iupac = { "R" : "AG",
          "Y" : "CT",
          "S" : "GC",
          "W" : "AT",
          "K" : "GT",
          "M" : "AC",
          "B" : "CGT",
          "D" : "AGT",
          "H" : "ACT",
          "V" : "ACG",
          "N" : "ACGT" }

def iupac2re(seq):
    for code in iupac:
        seq = seq.replace(code, "[%s]"%iupac[code])
    return seq


#########################################################################

def parseargs(appname, args, shorthandids=None):
    args.add_argument("steps", type=str, help="Either 'all', or a Comma-separated list of calib,train,test,report. Default is all.")
    args.add_argument("id", type=str, nargs="*", metavar="LIST", help="Explicit list of ids to train")
    args.add_argument("-d","--device", type=str, default="0", help="GPUs permitted to use, e.g. \"0,1,2\". Default is \"0\".")
    args.add_argument("-c","--chunk", type=str, default=None, help="For spreading job across multiple workstations. Saying \"2/3\" divides the work into 3 chunks and makes this process responsible for the second chunk.")
    args.add_argument("-q","--quick", action="store_true", default=False, help="Quick mode. Only trains/tests a few models, and only on a subset of the data rows.")
    args = args.parse_args()

    if args.steps == "all":
        args.steps = "calib,train,test,report"
    args.steps = args.steps.split(",")

    args.outdir    = "../out/"+appname
    args.device = [int(id) for id in args.device.split(",")]
    kangaroo.globals.set_devices(args.device)
    
    args.calibdir  = args.outdir+"/calib"
    args.finaldir  = args.outdir+"/final"
    args.reportdir = args.outdir+"/report"
    args.testdir   = args.outdir+"/test"

    if shorthandids:
        for id in args.id:
            if id in shorthandids:
                args.id.remove(id)
                args.id.extend(shorthandids[id])

    args.nfold  = 2  if args.quick else 3
    args.ncalib = 3  if args.quick else 30
    args.ntrial = 2  if args.quick else 6

    args.id = list(set(args.id))
    return args

#################################

def loadmodels(args, modeldir):
    # Load one trainer for everything
    if args.quick:
        trainer = kangaroo.loadcfg("cfg/trainers/quick.cfg")
    else:
        trainer = kangaroo.loadcfg("cfg/trainers/default.cfg")

    models = {}
    if args.quick:
        modelnames = ("M1",)
    else:
        modelnames = ("M1","M2")

    # Load several types of models and pair them up with the trainer.
    for name in modelnames:
        model = kangaroo.loadcfg("%s/%s.cfg" % (modeldir,name))
        models[name] = { "model" : model, "trainer" : trainer }
    return models

#################################

def getchunktargets(args, targetnames):
    chunknames = [name for name in targetnames]

    if args.id:
        chunknames = [name for name in targetnames if any(re.search(id,name) for id in args.id)]

    if args.chunk:
        chunkidx = int(args.chunk.split("/")[0])
        numchunk = int(args.chunk.split("/")[1])
        assert chunkidx >= 1 and chunkidx <= numchunk

        splits = [int(x) for x in np.linspace(0, len(chunknames), numchunk+1)]
        #print splits
        lo = splits[chunkidx-1]
        hi = splits[chunkidx]
        if lo == hi:
            quit("Chunk is empty (no columns). Quitting.")
        print "Chunk %s working on columns %d..%d out of %d" % (args.chunk, lo+1, hi, len(chunknames))
        chunknames = chunknames[lo:hi]

    #idxs = [i for i in idxs if not os.path.exists("out/rnac/A/final/%s"%targetnames[i])]
    #print [targetnames[i] for i in idxs]
    return chunknames

#################################

def save_metrics(data, groupname, outdir, modeldir=None):
    if modeldir is None:
        modeldir = outdir

    if not isinstance(data,dict):
        data = { id : data.astargets([id]) for id in data.targetnames }

    pred = kangaroo.predict(data, modeldir, outdir)
    for targetname in data.keys():
        z = pred[targetname].ravel()
        y = data[targetname].Y.ravel()
        rowidx = data[targetname].rowidx
        _update_metrics(outdir, targetname, groupname, rowidx, z, y)

    check_repeat_correlation = False
    if check_repeat_correlation:
        for targetname in data.keys():
            z = pred[targetname].ravel()
            y = data[targetname].Y.ravel()
            s = [data[targetname].X_seq[i] for i in range(len(data[targetname]))]

            # Only get the bound examples.
            z = [s[i] for i in range(len(data[targetname])) if y[i] != 0]
            s = [s[i] for i in range(len(data[targetname])) if y[i] != 0]

            class revcomp_str(object):
                def __init__(self, s): self.s = s;
                def __eq__(self, other): return self.s == other.s or self.s == revcomp(other.s)

            # For each percentile, count the number of repeats
            print targetname
            order = np.argsort(z)
            for denom in reversed([2,8,32,128,512,2048]):
                top_s = [s[i] for i in order[len(order)//denom:]]
                top_unique = set([revcomp(top_s[i]) for i in range(len(top_s))])
                nall = float(len(top_s))
                nunique = float(len(top_unique))
                percent_repeat = 100*(nall - nunique) / nall
                print "  top %.4f%% (n=%d)\t=> %.2f%% repeats"  % (100./denom, nall, percent_repeat)
                


def _update_metrics(outdir, targetname, groupname, rowidx, z, y, aucthresh=(.5,.5)):
    modeldir = outdir+"/"+targetname

    # Load current predictions 
    with np.load(modeldir+"/predict.npz") as f:
        predict_npz = { name : f[name] for name in f.keys() }

    # Add prediction group
    group = predict_npz["groups"][()].setdefault(groupname,{})
    group["I"] = rowidx
    group["Z"] = z.reshape((-1,1))
    if y is not None:
        group["Y"] = y.reshape((-1,1))

    # Save predictions
    np.savez_compressed(modeldir+"/predict.npz", **predict_npz)
    deepity.call_dumpviz(modeldir+"/predict.npz")

    # Load current metrics
    metrics = deepity.load_metrics(modeldir+"/metrics.txt")
    metrics[groupname] = deepity.calc_metrics(z, y, aucthresh=aucthresh)
    deepity.save_metrics(modeldir+"/metrics.txt", metrics)

    """
    with open("%s/%s/predict.tsv" % (outdir,targetname),"w") as f:
        f.write("RowIndex\tPredict")
        if y is not None:
            assert len(z) == len(y)
            f.write("\tTarget")
            yfmt = "\t%d" if np.all(y.astype(np.int32) == y) else "\t%.4f"
        f.write("\n")
        for i in range(len(z)):
            f.write("%d\t%.4f" % (rowidx[i],z[i]))
            if y is not None:
                f.write(yfmt % y[i])
            f.write("\n")"""

def disable_softmax():
    kangaroo.globals.flags.push("disable_softmax",True)

def enable_softmax():
    kangaroo.globals.flags.pop("disable_softmax")


#################################

def save_featuremaps(data, modeldir, outdir, maxrows=1000000):
    if not isinstance(data,dict):
        data = { id : data.astargets([id]) for id in data.targetnames }

    disable_softmax()
    pred = kangaroo.predict(data, modeldir, outdir)

    # Find a list of rows, in order of decreasing confidence, and with all
    # duplicate sequences deleted
    for name, Z in pred.iteritems():
        Y = data[name].Y.ravel()
        allidx = []
        allseq = set()
        roworder = np.argsort(-Z.ravel())
        for i in range(len(roworder)):
            s = data[name].X_seq[roworder[i]]
            #if s in allseq:
            #    continue
            allidx.append(roworder[i])
            allseq.add(s)
            if maxrows and len(allidx) >= maxrows:
                break

        # Now actually dump the featuremaps for all the rows specified
        print "Generating feature maps...",
        datasub = data[name][allidx]
        kangaroo.globals.set_multiprocessing(False) # Needed to get back collected values in globals; uugh
        if "reverse_complement" in kangaroo.globals.flags:
            kangaroo.globals.flags.push("collect_Zmask",True)
        kangaroo.globals.flags.push("collect_featuremaps",True)
        kangaroo.globals.flags.push("disable_relu",True)
        kangaroo.predict(datasub, modeldir, outdir)
        kangaroo.globals.flags.pop("disable_relu")
        fmaps = kangaroo.globals.flags.pop("collect_featuremaps")
        Zmask = kangaroo.globals.flags.pop("collect_Zmask").ravel() if "reverse_complement" in kangaroo.globals.flags else None
        kangaroo.globals.set_multiprocessing(True)

        seqs = []
        for i in range(len(datasub)):
            s = datasub.sequences[i][0]
            seqs.append(s)
            if "reverse_complement" in kangaroo.globals.flags:
                seqs.append(revcomp(s))

        if "reverse_complement" in kangaroo.globals.flags:
            # Remove the strand that was not used for prediction
            fmaps = [fmaps[i] for i in range(len(fmaps)) if Zmask[i]]
            seqs  = [seqs[i]  for i in range(len(seqs))  if Zmask[i]]
        filter_len = fmaps[0].shape[1] - len(seqs[0]) + 1

        # Make tuples of (sequence, max_value, max_index) for each featuremap
        pfmargs = []
        for k in range(fmaps[0].shape[0]):
            pfmarg = [(seqs[i], float(np.max(fmaps[i][k])), int(np.argmax(fmaps[i][k]))) for i in range(len(seqs))]
            pfmargs.append(pfmarg)
        pfmargs = np.array(pfmargs, dtype='a%d,f4,i4' % max(len(seqs[i]) for i in range(len(seqs))))
        print "done"

        #np.savez_compressed(outdir + "/%s.pfm_info.npz"%(name), pfmargs=pfmargs)
        

        # Compute PFMs from the pfmargs array
        print "Computing PFMs for %s..." % name,
        pfms, ic, kl_dist, pval, counts  = compute_pfms(pfmargs, filter_len=filter_len, num_permut=500, rev_comp=False)
        print "done"
        makepath(outdir)
        with open(outdir + "/%s.pfms.pkl"%(name),"wb") as f:
            cPickle.dump({"pfms" : pfms, "ic" : ic, "kl_dist" : kl_dist, "pval" : pval, "counts" : counts}, f)

    enable_softmax()



#################################

def retrain(models, data, calibdir, finaldir, nfold=1, ntrial=1):

    kangaroo.globals.flags.push("disable_softmax",True)
    for trial in range(10):
        Z = kangaroo.predict(data, finaldir, finaldir)["bound"]
        I0 = [i for i in range(len(data)) if data.Y[i] < .5]
        I1 = [i for i in range(len(data)) if data.Y[i] > .5]
        #threshold = np.mean(Z[I1].ravel()) - 4*np.std(Z[I1].ravel())
        #threshold = np.percentile(Z[I0].ravel(), 95)
        threshold = np.percentile(Z[I1].ravel(), 1)
        numshuffle = 0
        for i in I0:
            if Z[i] > threshold:
                data.sequences[i][0] = doublet_shuffle(data.sequences[i][0])
                data.X_seq[i] = data.sequences[i][0]
                numshuffle += 1

        numshuffle_pct = float(numshuffle)/len(I0)*100
        print "retrain trial %d: had to shuffle %.1f%% unbound sequences" % (trial, numshuffle_pct)
        if numshuffle <= .0001*len(I0):
            break

    kangaroo.globals.flags.pop("disable_softmax")

    kangaroo.train(models, data, calibdir, finaldir, nfold=nfold, ntrial=ntrial)
    return


#################################

def train_without_outliers(cfgs, data, calibdir, outdir, ntrial=1, auxfilter=None):
    # Same as kangaroo.train, except this version trains twice: once with all
    # the data, and again with worst outliers 'removed'.

    # Step 1. Train with all training data
    #kangaroo.train(cfgs, data, calibdir, outdir, nfold=1, ntrial=ntrial, auxfilter=auxfilter)

    # Step 2. Load predictions on the training data, and remove%
    #         the examples that were misclassified the worst.
    if not isinstance(data,dict):
        data = { id : data.astargets([id]) for id in data.targetnames }

    kangaroo.globals.flags.push("disable_softmax",True)
    pred = kangaroo.predict(data, outdir, outdir)
    for targetname in sorted(data.targetnames):
        for name, Z in pred.iteritems():
            Y = data[name].Y.ravel()
            allidx = []
            allseq = set()
            roworder = np.argsort(-Z.ravel())


#################################

def disable_multiprocessing():
    kangaroo.globals.set_multiprocessing(False)
    #smat.set_default_dtype(smat.float64)

#################################

def enable_reversecomplement(force=False):
    kangaroo.globals.flags.push("reverse_complement", "force" if force else True )


def reversecomplement_enabled():
    return "reverse_complement" in kangaroo.globals.flags

#################################

def shufseq(s):
    return "".join([s[i] for i in npr.permutation(len(s))])

def randpad(seq, padsize):
    if not padsize:
        return seq
    n = len(seq)
    j = npr.randint(0, padsize)
    return randseq(j, j) + seq + randseq(padsize-j, padsize-j)


class _augment_seqdata(object):
    def __init__(self, minrows, padsize):
        self.minrows = minrows
        self.padsize = padsize

    def __call__(self, data):
        nrows = max(self.minrows, len(data)) if self.minrows else len(data)
        data = data[[i//2 % len(data) for i in range(nrows*2)]]
        for i in range(0,nrows*2,2):
            seq = data.sequences[i][0]
            seq = randpad(seq, self.padsize)
            shuf = doublet_shuffle(seq)
            data.sequences[i]   = [seq]
            data.sequences[i+1] = [shuf]
            data.X_seq[i]   = seq
            data.X_seq[i+1] = shuf
            data.Y[i+1] = 0

        return data

def load_seq(filename, minrows=None, maxrows=None, padsize=None):
    if maxrows:
        minrows = min(maxrows, minrows)
    data = datasource.fromtxt(filename, None, None, maxrows=maxrows)
    
    # Duplicate all the data as many times as needed to provide a match
    # for each background sequence
    if minrows:
        data.augmented = _augment_seqdata(minrows, padsize)
    return data

#################################

def randseq(minlen, maxlen, p=None):
    size = npr.randint(minlen, maxlen+1)
    if p is None:
        return ord2acgt(npr.randint(0, 4, size))
    return ord2acgt(npr.choice([0,1,2,3], size=size, p=p))


######################################################


def save_report(modeldir, reportdir, tfids, index_metric="auc", rna=False):

    makepath(reportdir)
    index_html = open(reportdir+"/index.html",'w')
    index_html.write("<html><head><title>Training report</title></head><body>\n")
    index_html.write("<table cellspacing=0 cellpadding=5 border=1>\n")
    index_html.write("<tr><th>Name</th><th>train %s</th><th>test %s</th></tr>\n" % (index_metric, index_metric))

    performance_txt = open(reportdir+"/performance.txt",'w')
    performance_txt.write("\ttrain.%s\ttest.%s\n" % (index_metric, index_metric))

    for tfid in tfids:
        print tfid
        makepath(reportdir+"/"+tfid)

        # Load PFMs, convert them to logo images, and dump them to the report directory
        logos = []
        if os.path.exists(reportdir+"/%s.pfms.pkl"%tfid):
            with open(reportdir+"/%s.pfms.pkl"%tfid) as f:
                _ = cPickle.load(f)
                pfms = _["pfms"]
                ics  = _["ic"]
                counts = _["counts"]
            pfm_order = np.argsort(-ics)
            for j in pfm_order:
                pfm = pfms[j]
                ic  = ics[j]
                if ic <= 0:
                    continue

                pfm_rev = np.fliplr(np.flipud(pfm))
                logo_filename = "%s/%s/pfm%02d" % (reportdir, tfid, len(logos))
                logos.append((j, os.path.basename(logo_filename)+"_fwd.png", ic, counts[j]))
                logo_fwd = deepity.tape2logo.tape2logo(pfm.T,     height=50, letterwidth=10, bufferzoom=4, vmax=1.0, style="seqlogo", rna=rna)
                logo_rev = deepity.tape2logo.tape2logo(pfm_rev.T, height=50, letterwidth=10, bufferzoom=4, vmax=1.0, style="seqlogo", rna=rna)
                scipy.misc.imsave(logo_filename+"_fwd.png", logo_fwd)
                scipy.misc.imsave(logo_filename+"_rev.png", logo_rev)

        # Load train/test metrics so we can print them in the HTML report
        metrics_file = "%s/%s/metrics.txt" % (modeldir, tfid)
        metrics = deepity.load_metrics(metrics_file)

        # Add row for this TF in main index.html
        index_html.write("<tr>\n")
        index_html.write("<td><a href=\"%s/index.html\">%s</a></td>\n" % (tfid, tfid))
        train_performance = ""
        test_performance = ""
        if index_metric=="auc":
            if "train" in metrics:
                index_html.write("<td>%s        &plusmn; %s</td>\n" % (metrics["train"]["auc.mean"], metrics["train"]["auc.std"]))
                train_performance = metrics["train"]["auc.mean"]
            if "test"  in metrics:
                index_html.write("<td><b>%s</b> &plusmn; %s</td>\n" % (metrics["test"]["auc.mean"],  metrics["test"]["auc.std"]))
                test_performance = metrics["test"]["auc.mean"]
        elif index_metric=="pearson":
            if "train" in metrics:
                index_html.write("<td>%s        (p=%s)</td>\n" % (metrics["train"]["pearson.r"], metrics["train"]["pearson.p"]))
                train_performance = metrics["train"]["pearson.r"]
            if "test"  in metrics:
                index_html.write("<td><b>%s</b> (p=%s)</td>\n" % (metrics["test"]["pearson.r"],  metrics["test"]["pearson.p"]))
                train_performance = metrics["test"]["pearson.r"]
        index_html.write("</tr>\n")

        performance_txt.write("%s\t%s\t%s\n" % (tfid, train_performance, test_performance))

        # Build page showing filters and sequence logos for this specific TF
        tf_html = open(reportdir+"/"+tfid+"/index.html", 'w')
        tf_html.write("<html><head><title>Training report - %s</title></head><body>\n" % tfid)
        tf_html.write("<h2>%s</h2>\n" % tfid)
        # tf_html.write("<a href=\"../gmaps/%s/index.html\">gradient maps</a>)<hr/>\n"%(tfid,tfid))
        tf_html.write("""
        <script language="javascript">
        function toggle_strand()
        {
            var pfms = document.getElementsByClassName('pfm')
            for (var i = 0; i < pfms.length; i++)
                if (pfms[i].src.search("_fwd") != -1)
                    pfms[i].src = pfms[i].src.replace("_fwd","_rev");
                else if (pfms[i].src.search("_rev") != -1)
                    pfms[i].src = pfms[i].src.replace("_rev","_fwd");
        }
        </script></head><body>
        """)

        with open(metrics_file) as f:
            metrics_text = f.read()
        tf_html.write("<pre>%s</pre>\n"% metrics_text)
        if os.path.exists(modeldir+"/%s/predict.scatter-train.png" % tfid):
            tf_html.write("<br/>Distribution of training predictions:<br/><img src=\"../../final/%s/predict.scatter-train.png\"/>\n" % tfid)

        # Then generate a table for the complete model
        tfdir = modeldir+"/"+tfid
        if logos:
            tf_html.write("<hr/><h3>Feature Logos</h3>\n")
            tf_html.write("<input type=\"button\" value=\"toggle strand\" onclick=\"toggle_strand();\"/><br/>")
            tf_html.write("<table cellspacing=0 cellpadding=4 border=0>\n")
            for filter_index, logo, ic, count in logos:
                tf_html.write("<tr><td>%d</td><td><img src=\"%s\" class=\"pfm\"/></td><td>%.1f bits,</td><td> %d activations</td></tr>" % (filter_index, logo, ic, count))
            tf_html.write("</table><br/><br/>\n")


        # Now show the actual model.
        '''
        shutil.copy(tfdir_final[0]+"/fold0.report/filters_.conv_seq(0).color.png", basedir+"report/%s/final_filters.png" % (tfid))
        shutil.copy(tfdir_final[0]+"/fold0.report/filters_.conv_seq(0).logo.png",  basedir+"report/%s/final_logos.png" % (tfid))
        shutil.copy(tfdir_final[0]+"/fold0.report/filters_.conv_seq(1).png",       basedir+"report/%s/final_biases.png"  % (tfid))
        shutil.copy(tfdir_final[0]+"/fold0.report/filters_.combiner.png",          basedir+"report/%s/final_weights.png" % (tfid))
        os.system("convert %sreport/%s/final_filters.png -rotate -90 %sreport/%s/final_filters_rot.png" % (basedir,tfid,basedir,tfid))
        '''

        tf_html.write("<hr/><h3>Actual DeepBind model</h3>\n")
        if os.path.exists(tfdir + "/model.conv_seq(1).color.png"):
            tf_html.write("Filters:<br/>\n")
            tf_html.write("<img src=\"../../final/%s/model.conv_seq(1).color.png\"/><br/>\n" % tfid)

        if os.path.exists(tfdir + "/model.combiner.png"):
            tf_html.write("<br/>Combiner layer:<br/>\n")
            tf_html.write("<img src=\"../../final/%s/model.combiner.png\"/><br/>\n" % tfid)

        tf_html.write("</body></html>\n")


    index_html.write("</table>\n")
    index_html.write("</body></html>\n")
    index_html.close()

