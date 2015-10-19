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
import numpy as np
import numpy.random as npr
import smat as sm
import gc
from .util import makepath
from .data import make_predictions
from . import globals
import scipy
import scipy.stats

import subprocess

import warnings
#warnings.simplefilter('error', UserWarning)

####################################################################


def calc_auc(z, y, want_curve = False):
    assert len(z) == len(y)
    #m = ~np.isnan(y)
    #y = y[m]
    #z = z[m]

    #ymin = y.min()
    #ymax = y.max()
    #lo = (0.02-0) / (1.0 - 0.0) * (ymax-ymin) + ymin
    #hi = (0.10-0) / (1.0 - 0.0) * (ymax-ymin) + ymin

    #mlo = y<=lo
    #mhi = y>=hi
    #y[mlo] = 0
    #y[mhi] = 1
    #y[np.logical_and(mlo,mhi)] = np.nan

    #m = ~np.isnan(y)
    #y = y[m]
    #z = z[m]

    # Sort by decreasing score
    order = np.argsort(z, axis=0, kind="mergesort")[::-1].ravel()
    z = z[order]
    y = y[order]

    # Accumulate the true positives with decreasing threshold
    tpr = y.cumsum()
    fpr = 1 + np.arange(len(y)).astype(y.dtype) - tpr

    # If curve doesn't have point at x=0, explicitly insert one
    if fpr[0] != 0:
        tpr = np.r_[0,tpr]
        fpr = np.r_[0,fpr]

    # If one of the classes was empty, return NaN
    if fpr[-1] == 0 or tpr[-1] == 0:
        return (np.nan,None) if want_curve else np.nan

    # Convert sums to rates
    tpr = tpr / tpr[-1]
    fpr = fpr / fpr[-1]

    # Calculate area under the curve using trapezoidal rule
    auc = np.trapz(tpr,fpr,axis=0)

    # Done!
    if want_curve:
        curve = np.hstack([fpr,tpr])
        return auc,curve
    return auc

    
    '''
    #auc1 = auc
    npos = np.count_nonzero(y)      # Total number of positives.
    nneg = len(y)-npos              # Total number of negatives.
    if nneg == 0 or npos == 0:
        return (np.nan,None) if want_curve else np.nan

    n = len(y)
    fprate = np.zeros((n+1,1))
    tprate = np.zeros((n+1,1))
    ntpos,nfpos = 0.,0.
    for i,yi in enumerate(y):
        if yi: ntpos += 1
        else:  nfpos += 1
        tprate[i+1] = ntpos/npos
        fprate[i+1] = nfpos/nneg
    auc = np.trapz(tprate,fprate,axis=0)

    if want_curve:
        curve = np.hstack([fprate,tprate])
        return auc,curve
    return auc'''
    

#########################################################################

def bootstrap_auc(z, y, ntrial=10):
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


#####################################################################

def calc_metrics(z, y, aucthresh=(0.5,0.5)):
    metric = {}

    # Compute correlations using all non-NaN values
    M = ~np.isnan(y)
    z = z[M]
    y = y[M]
    metric["pearson.r"],  metric["pearson.p"]  = scipy.stats.pearsonr(z,y)
    metric["spearman.r"], metric["spearman.p"] = scipy.stats.spearmanr(z,y)


    if np.any(y < aucthresh[0]) and np.any(y >= aucthresh[1]):
        # Compute percentiles of positive and negative sets:
        #if np.sum(y < aucthresh[0]) > 1 and np.sum(y > .5) > 1:
        #    negmu,negstd = np.mean(z[y<.5]), np.std(z[y<.5])
        #    posmu,posstd = np.mean(z[y>.5]), np.std(z[y>.5])
        #    numer = posmu-negmu
        #    denom = posstd+negstd
        #    metric["gapratio"] = (numer / denom) if denom else np.nan

        # Convert y to binary using the aucrange threshold
        lo, hi = aucthresh
        Mlo = y< lo
        Mhi = y>=hi
        y[Mlo] = 0
        y[Mhi] = 1
        M = np.logical_or(Mlo,Mhi)
        y = y[M]
        z = z[M]

        # Calculate AUC stats
        metric["auc"] = calc_auc(z, y)
        metric["auc.mean"], metric["auc.std"] = bootstrap_auc(z, y, 20)

    return metric

#####################################################################


def _calc_performance_stats(cost, batches):
    Z = []
    Y = []
    L = []
    rowidx = []
    for batch in batches:
        args = { key : val for key,val in batch.data().items() if hasattr(cost,key) }
        cost.eval(clear=False,**args)
        Zi = cost.loss.Z.fpval.asnumpy()
        Yi = sm.asnumpy(batch.Y)
        Li = cost.loss.loss.fpval.asnumpy()
        rowidx_i = sm.asnumpy(batch.rowidx) if hasattr(batch,"rowidx") else None
        
        # Deal with possibility that only a subset of predictions contributed to the loss, due to some test performed after prediction (e.g. max of sequence and reverse_complement of sequence)
        if cost.Zmask.fpval is not None:
            Zmask = cost.Zmask.fpval.asnumpy()
            Zi = np.hstack([Zi[Zmask[:,col],col].reshape((-1,1)) for col in range(Zi.shape[1])])
            Yi = np.hstack([Yi[Zmask[:,col],col].reshape((-1,1)) for col in range(Yi.shape[1])])
            if rowidx_i is not None:
                rowidx_i = rowidx_i[Zmask[:,0],0].reshape((-1,1))

        Z.append(Zi)
        Y.append(Yi)
        L.append(Li)
        if rowidx_i is not None:
            rowidx.append(rowidx_i)
        cost.clear()

    Z = np.vstack(Z)
    Y = np.vstack(Y)
    L = np.vstack(L)
    L = np.mean(L,axis=0)
    rowidx = np.vstack(rowidx) if rowidx else None
    return { "L" : L, "Z" : Z, "Y" : Y, "I" : rowidx }
    #return (L,Z,Y,rowidx)


#####################################################

class training_report(object):
    '''
    class subreport(object):
        def __init__(self,filename):
            self.filename = filename
            makepath(os.path.dirname(filename))
            self.logfile = open(filename,'w')
            self.entries = []

        def update(self, step, model,
                   tloss, tZ, tY, tI,
                   vloss, vZ, vY, vI,
                   **kwargs):

            if tI is not None:
                p = np.argsort(tI.ravel())
                tZ = tZ[p,:]
                tY = tY[p,:]
                tI = tI[p,:]
            if vI is not None:
                p = np.argsort(vI.ravel())
                vZ = vZ[p,:]
                vY = vY[p,:]
                vI = vI[p,:]

            # Build a short string describing the current state
            txt = ""
            if step  is not None: txt += "%06d:" % step
            if tloss is not None: txt += "\t tloss=%.8f" % tloss
            if vloss is not None: txt += "\t vloss=%.8f" % vloss
            for key,val in kwargs.iteritems():
                txt += "\t%s=%s\t " % tloss

            for i in range(tY.shape[1]):
                tZi = tZ[:,i:i+1]
                tYi = tY[:,i:i+1]
                tMi = ~np.isnan(tYi)
                tpearson  = scipy.stats.pearsonr(tZi[tMi],tYi[tMi])
                tspearman = scipy.stats.spearmanr(tZi[tMi],tYi[tMi])
                tauc = bootstrap_auc(tZi[tMi],tYi[tMi])
                txt += "\t tp%02d=%.3f (%g)" % (i,tpearson[0],tpearson[1])
                txt += "\t ts%02d=%.3f (%g)" % (i,tspearman[0],tspearman[1])
                txt += "\t ta%02d=%.3f (%g)" % (i,tauc[0],tauc[1])
                if vY is not None:
                    vZi = vZ[:,i:i+1]
                    vYi = vY[:,i:i+1]
                    vMi = ~np.isnan(vYi)
                    vpearson  = scipy.stats.pearsonr(vZi[vMi],vYi[vMi])
                    vspearman = scipy.stats.spearmanr(vZi[vMi],vYi[vMi])
                    vauc = bootstrap_auc(vZi[vMi],vYi[vMi])
                    txt += "\t vp%02d=%.3f (%g)" % (i,vpearson[0],vpearson[1])
                    txt += "\t vs%02d=%.3f (%g)" % (i,vspearman[0],vspearman[1])
                    txt += "\t va%02d=%.3f (%g)" % (i,vauc[0],vauc[1])

            # If such state was provided, append it to our file.
            if txt != "":
                mode = "w" if self.entries == [] else "a"
                with open(self.filename,mode) as f:
                    f.write(txt+"\n")

            images = self.extract_filters(model) if "store_featuremaps" in globals.flags else None

            entry = {}
            if step   is not None: entry['step']   = step
            if tloss  is not None: entry['tloss']  = tloss
            if tloss  is not None: entry['tauc']   = tauc
            if tZ     is not None: entry['tZ']     = tZ
            if tY     is not None: entry['tY']     = tY
            if tI     is not None: entry['tI']     = tI
            if vloss  is not None: entry['vloss']  = vloss
            if vloss  is not None: entry['vauc']   = vauc
            if vZ     is not None: entry['vZ']     = vZ
            if vY     is not None: entry['vY']     = vY
            if vI     is not None: entry['vI']     = vI
            if images is not None: entry['images'] = images
            entry.update(kwargs)
            self.entries.append(entry)

        @staticmethod
        def extract_filters(model):
            # Extract any filters from the dependency graph nodes, and
            # convert them to images.
            if not model:
                return None

            filters = {}
            def collect_filters(path,obj):
                if hasattr(obj,"getfilters"):
                    F = obj.getfilters()
                    if F is not None:
                        filters[path] = F  
            model.visit(collect_filters)

            return filters if len(filters) > 0 else None


        def dump(self, want_html=False, want_anim=False):
            # Don't just dump the status text -- also dump all weights, in an archive
            if want_html:
                prefix = os.path.splitext(self.filename)[0]
                np.savez_compressed(prefix+".report.npz", entries=np.asarray(self.entries[-1:],dtype=object))
            
                # Fire off a separate process that will generate filter/prediction plots
                # using matplotlib; this should be done in a separate process so as to avoid
                # instability with matplotlib/tk in the multiprocess hyperparameter search
                subprocess.Popen(["python", os.path.dirname(__file__)+"/report_plotter.py", prefix+".report.npz"])
                '''

    ###############################

    def __init__(self, aucrange=(0.5,0.5)):
        """auclo and auchi can be used to explicitly threshold a non-binary target Y"""
        self.aucrange = aucrange
        self.foldid = 0
        self.entries = {}
        # An "entry" is identified by indices entries[task][step][dataset].
        # Each entry entry[quantity][foldid] 
        # is the quantity for the specific foldid, where quantity is something like 'Z' or 'Y' and foldid is some integer like 0,1,2
#        self.subreports = [self.subreport(logfile_pattern % {"sample_id" : sample_id, "task_id" : task_id})
#                           for (task_id,sample_id) in zip(task_ids,sample_ids)]

    def setfold(self, foldid):
        self.foldid = foldid

    ###############################

    @staticmethod
    def _get_submodel(model, index):
        submodel = copy.deepcopy(cost.model)
        sm.sync()
        submodel.slice_inst(index)
        sm.sync()
        return submodel

    ###############################

    def __call__(self, step, cost, datasets):
        """
        Record the performance of each submodel (of 'cost') at this particular step.
        """
        # Compute all loss values L, predictions Z, targets Y, and original row indices I both tdata, and for vdata
        for name, data in datasets.iteritems():
            stats = _calc_performance_stats(cost, data)

            # Figure out how many output dimensions (columns of Y) belong to each target.
            assert stats["L"].size == len(data.curr().targetnames)
            assert stats["Y"].shape[1]  % stats["L"].size == 0
            Ydim = stats["Y"].shape[1] // stats["L"].size

            # Split up the columns 
            for i, targetname in enumerate(data.curr().targetnames):

                # Store the loss, predictions Z, and targets Y for model i
                entry = self.entries.setdefault(i,{}).setdefault(step,{})
                entry.setdefault(name,{}).setdefault("L",[]).append(float(stats["L"][i]))
                entry.setdefault(name,{}).setdefault("Z",[]).append(stats["Z"][:,i*Ydim:(i+1)*Ydim])
                entry.setdefault(name,{}).setdefault("Y",[]).append(stats["Y"][:,i*Ydim:(i+1)*Ydim])
                entry.setdefault(name,{}).setdefault("I",[]).append(stats["I"])

    ###############################

    def combined(self):
        combined = {}
        for taskidx in self.entries:
            for step in self.entries[taskidx]:
                for group in self.entries[taskidx][step]:
                    entry = self.entries[taskidx][step][group]
                    combined_entry = combined.setdefault(taskidx,{}).setdefault(step,{}).setdefault(group,{})
                    combined_entry["L"] = np.mean(entry["L"])
                    combined_entry["Z"] = np.vstack(entry["Z"])
                    combined_entry["Y"] = np.vstack(entry["Y"])
                    combined_entry["I"] = np.vstack(entry["I"])
        return combined

    ###############################

    def curr(self):
        curr = {}
        for taskidx in self.entries:
            for step in self.entries[taskidx]:
                for group in self.entries[taskidx][step]:
                    entry = self.entries[taskidx][step][group]
                    curr_entry = curr.setdefault(taskidx,{}).setdefault(step,{}).setdefault(group,{})
                    curr_entry["L"] = entry["L"][self.foldid]
                    curr_entry["Z"] = entry["Z"][self.foldid]
                    curr_entry["Y"] = entry["Y"][self.foldid]
                    curr_entry["I"] = entry["I"][self.foldid]
        return curr

    ###############################
            


    '''
            pass
            curr = self.results.setdefault(targetname,{}).setdefault(self.foldid,{}).setdefault(step,{})


            submodel = self._get_submodel(cost, i)
    
            status = self._calc_status(step, submodel, 
                                       float(tloss[i]),
                                       tZ[:,i*Ydim:(i+1)*Ydim],
                                       tY[:,i*Ydim:(i+1)*Ydim],
                                       tI,
                                       float(vloss[i])          if vloss is not None else None,
                                       vZ[:,i*Ydim:(i+1)*Ydim]  if vloss is not None else None,
                                       vY[:,i*Ydim:(i+1)*Ydim]  if vloss is not None else None,
                                       vI)
            submodel = None
            sm.sync()
            gc.collect()
            # Compare it to the best we've found so far, if any
            #curr = self.results.setdefault(targetname,{}).setdefault(self.foldid,{}).setdefault(step,{})
            #if curr:
            #    perfkey = "va" if vloss is not None else "tr"
            #    if curr
    '''
    '''

    def _calc_status(self, step, model,
                     tloss, tZ, tY, tI,
                     vloss, vZ, vY, vI,
                     **kwargs):
            if tI is not None:
                p = np.argsort(tI.ravel())
                tZ = tZ[p,:]
                tY = tY[p,:]
                tI = tI[p,:]
            if vI is not None:
                p = np.argsort(vI.ravel())
                vZ = vZ[p,:]
                vY = vY[p,:]
                vI = vI[p,:]

            # Build a short string describing the current state
            txt = ""
            if step  is not None: txt += "%06d:" % step
            if tloss is not None: txt += "\t tloss=%.8f" % tloss
            if vloss is not None: txt += "\t vloss=%.8f" % vloss
            for key,val in kwargs.iteritems():
                txt += "\t%s=%s\t " % tloss

            for i in range(tY.shape[1]):
                tZi = tZ[:,i:i+1]
                tYi = tY[:,i:i+1]
                tMi = ~np.isnan(tYi)
                tpearson  = scipy.stats.pearsonr(tZi[tMi],tYi[tMi])
                tspearman = scipy.stats.spearmanr(tZi[tMi],tYi[tMi])
                tauc = bootstrap_auc(tZi[tMi],tYi[tMi])
                txt += "\t tp%02d=%.3f (%g)" % (i,tpearson[0],tpearson[1])
                txt += "\t ts%02d=%.3f (%g)" % (i,tspearman[0],tspearman[1])
                txt += "\t ta%02d=%.3f (%g)" % (i,tauc[0],tauc[1])
                if vY is not None:
                    vZi = vZ[:,i:i+1]
                    vYi = vY[:,i:i+1]
                    vMi = ~np.isnan(vYi)
                    vpearson  = scipy.stats.pearsonr(vZi[vMi],vYi[vMi])
                    vspearman = scipy.stats.spearmanr(vZi[vMi],vYi[vMi])
                    vauc = bootstrap_auc(vZi[vMi],vYi[vMi])
                    txt += "\t vp%02d=%.3f (%g)" % (i,vpearson[0],vpearson[1])
                    txt += "\t vs%02d=%.3f (%g)" % (i,vspearman[0],vspearman[1])
                    txt += "\t va%02d=%.3f (%g)" % (i,vauc[0],vauc[1])

            # If such state was provided, append it to our file.
            if txt != "":
                mode = "w" if self.entries == [] else "a"
                with open(self.filename,mode) as f:
                    f.write(txt+"\n")

            images = self.extract_filters(model) if "store_featuremaps" in globals.flags else None

            entry = {}
            if step   is not None: entry['step']   = step
            if tloss  is not None: entry['tloss']  = tloss
            if tloss  is not None: entry['tauc']   = tauc
            if tZ     is not None: entry['tZ']     = tZ
            if tY     is not None: entry['tY']     = tY
            if tI     is not None: entry['tI']     = tI
            if vloss  is not None: entry['vloss']  = vloss
            if vloss  is not None: entry['vauc']   = vauc
            if vZ     is not None: entry['vZ']     = vZ
            if vY     is not None: entry['vY']     = vY
            if vI     is not None: entry['vI']     = vI
            if images is not None: entry['images'] = images
            entry.update(kwargs)
            self.entries.append(entry)
            '''
    '''
    @staticmethod
    def merge_reports(logfile_pattern, task_ids, sample_ids, reports):
        merged = reports[0].__class__(logfile_pattern, task_ids, sample_ids)
        for i in range(len(sample_ids)):
            kwargs = {}
            for key in ("tZ","tY","tI","vZ","vY","vI"):
                if key in reports[0].subreports[0].entries[-1]:
                    kwargs[key] = np.vstack([r.subreports[i].entries[-1][key] for r in reports])
                else:
                    kwargs[key] = None

            merged.subreports[i].update(step=None,
                                        model=None,
                                        tloss=None,
                                        vloss=None,
                                        **kwargs)
        return merged

    @staticmethod
    def shutdown():
        import _tkinter as tk
        tk.quit()

    def update(self, step, cost, tdata, vdata):
        tloss,tZ,tY,tI = _calc_performance_stats(cost,tdata)
        vloss,vZ,vY,vI = _calc_performance_stats(cost,vdata) if vdata is not None else (None,None,None,None)
        assert tY.shape[1] % tloss.size == 0
        Ydim = tY.shape[1] // tloss.size
        for i in range(len(self.sample_ids)):
            model_i = copy.deepcopy(cost.model)
            sm.sync()
            model_i.slice_inst(i)
            sm.sync()
    
            # Slice the trainable weights
            self.subreports[i].update(step,
                                      model_i,
                                      float(tloss[i]),
                                      tZ[:,i*Ydim:(i+1)*Ydim],
                                      tY[:,i*Ydim:(i+1)*Ydim],
                                      tI,
                                      float(vloss[i])          if vloss is not None else None,
                                      vZ[:,i*Ydim:(i+1)*Ydim]  if vloss is not None else None,
                                      vY[:,i*Ydim:(i+1)*Ydim]  if vloss is not None else None,
                                      vI)
            model_i = None
            sm.sync()
            gc.collect()

        tauc = None
        vauc = None
        if Ydim == 1:
            tauc = np.zeros(len(self.sample_ids))
            if vloss is not None:
                vauc = np.zeros(len(self.sample_ids))
            for i in range(len(self.sample_ids)):
                tauc[i] = calc_auc(tZ[:,i].ravel(), tY[:,i].ravel())
                if vloss is not None:
                    vauc[i] = calc_auc(vZ[:,i].ravel(), vY[:,i].ravel())
        return tloss,vloss,tauc,vauc


    def dump(self, **kwargs):
        for subrep in self.subreports:
            subrep.dump(**kwargs)

    '''
