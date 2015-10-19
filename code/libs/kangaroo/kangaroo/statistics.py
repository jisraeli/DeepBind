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
# predict.py
#    
import os
import os.path
import numpy as np
import numpy.random as npr
import scipy
import scipy.misc
import scipy.stats
import scipy.stats.mstats

def is_binary(y):
    return np.all(np.logical_or(y==0, y==1))


def _calc_auc(z, y, want_curve = False):
    z = z.ravel()
    y = y.ravel()
    
    assert len(z) == len(y)
    m = ~np.isnan(y)
    y = y[m]
    z = z[m]

    assert is_binary(y), "Cannot calculate AUC for non-binary targets"

    order = np.argsort(z,axis=0)[::-1].ravel()   # Sort by decreasing order of prediction strength
    z = z[order]
    y = y[order]
    npos = np.count_nonzero(y)      # Total number of positives.
    nneg = len(y)-npos              # Total number of negatives.
    if nneg == 0 or npos == 0:
        return (np.nan,None) if want_curve else 1

    n = len(y)
    fprate = np.zeros((n+1,1))
    tprate = np.zeros((n+1,1))
    ntpos,nfpos = 0.,0.
    for i,yi in enumerate(y):
        if yi: ntpos += 1
        else:  nfpos += 1
        tprate[i+1] = ntpos/npos
        fprate[i+1] = nfpos/nneg
    auc = float(np.trapz(tprate,fprate,axis=0))
    if want_curve:
        curve = np.hstack([fprate,tprate])
        return auc, curve
    return auc

def _bootstrap_auc(z, y, ntrial=20):
    if ntrial <= 1:
        return _calc_auc(z, y), np.nan
    n = len(y)
    aucs = np.zeros(ntrial)
    for t in range(ntrial):
        sample = npr.randint(0,n,n)
        zt = z[sample].copy()
        yt = y[sample].copy()
        aucs[t] = _calc_auc(zt, yt)
    return np.mean(aucs), np.std(aucs)

#########################################################################

def statistics(predictions, data, bootstrap=20, auc_zscore=4):
    """
    Calculates correlations between predictions and the corresponding column of data.targets.
    If the targets are binary, then also calculates AUCs.
    The input 'predictions' should be a dictionary where each key is a target name and
    each value is a Nx1 numpy array, where N is the number of rows in data.
    If the targets are not binary, the AUC will be computed by assigning 1
    to all targets with Z-score >= auc_zscore, and 0 to the others.
    """
    stats = {}
    for targetname, predict in predictions.iteritems():
        if targetname not in data.targetnames:
            continue
        targetidx = data.targetnames.index(targetname)
        targets    = data.Y[:,targetidx]
        targetmask = data.Ymask[:,targetidx]

        pearson   = scipy.stats.pearsonr(predict[targetmask].ravel(), targets[targetmask].ravel())
        spearman  = scipy.stats.spearmanr(predict[targetmask].ravel(), targets[targetmask].ravel())

        stats[targetname] = { 'pearson'  : { 'r' : pearson[0],  'p' : pearson[1]  }, 
                              'spearman' : { 'r' : spearman[0], 'p' : spearman[1] }, 
                              }

        if is_binary(targets):
            labels = targets
        else:
            labels = targets.copy()
            labels[targetmask] = (scipy.stats.mstats.zscore(targets[targetmask].ravel()) >= auc_zscore).astype(np.float)
        auc_mean, auc_std  = _bootstrap_auc(predict, labels, ntrial=bootstrap)
        auc, auc_curve = _calc_auc(predict, labels, want_curve=True)
        stats[targetname]['AUC'] = { 'mean' : auc_mean, 'std' : auc_std, 'value' : auc, 'curve' : auc_curve }

    return stats
