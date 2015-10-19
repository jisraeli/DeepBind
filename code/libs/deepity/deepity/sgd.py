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
import time
import logging
import smat as sm
import numpy as np
import numpy.random as npr
from . import _ext
from .trainer  import trainer
from .std  import trainable
from .gradcheck import gradcheck
from .     import globals
from .plug import disconnect
import report as _report


class sgd(trainer):
    """
    Stochastic (mini-batch) gradient descent with momentum.
    """
    def __init__(self, rate = 0.05, 
                       rate_decay = 1.0, #0.99998,
                       momentum = 0.85, 
                       nesterov = True, batchsize = 64,
                       max_epochs = 20, 
                       max_steps = 10000,
                       viz_steps = 10000,
                       checkpoints = 1,
                       checkpoint_save = True,
                       weight_decay_start = 100,
                       lossfunc = None,
                       gradcheck = False):
        self.rate        = rate
        self.rate_decay  = rate_decay
        self.momentum    = momentum
        self.nesterov    = nesterov
        self.batchsize   = batchsize
        self.max_epochs  = max_epochs
        self.max_steps   = max_steps
        self.viz_steps   = viz_steps
        self.checkpoints = checkpoints
        self.checkpoint_save = checkpoint_save
        self.weight_decay_start = weight_decay_start
        self.gradcheck   = gradcheck
        self.lossfunc = lossfunc
        self._report_tbatches = None
        self._report_vbatches = None
        self._last_drate_step = None
        self._last_drate_scale = None
        self._last_mrate_step  = None
        self._last_mrate_scale = None
        self._last_report_step = None
        self._last_report_time = None

    def _train(self, trainable_plugs, cost, datasets, checkpoint_callback):

        self.force_single_value("rate_decay","nesterov","batchsize","max_steps","max_epochs")

        # Ask tdata for an object that serves mini-batches
        batchsets = {name : data.asbatches(self.batchsize, reshuffle=False) for name, data in datasets.iteritems()}
        tbatches = batchsets["train"]
        vbatches = batchsets.get("validate", None)

        # Allocate and bind trainable parameters to the cost object
        P,dP,mP,drate,mrate,trnodes = self._train_setup(trainable_plugs, cost)
        bprop_trainable  = lambda: cost.bprop_trainable(trnodes, tbatches.next())

        if self.gradcheck:
            gradcheck(trnodes,cost,tbatches.next())

        def enforce_constraints():
            for trnode in trnodes:
                trnode.enforce_constraints()

        enforce_constraints()

        #sm.sync()
        #print "START"
        #t0 = time.time()
        #self.max_steps = 1000
        globals.flags.push("weight_decay_start", self.weight_decay_start)

        # Main training loop.
        max_steps = int(min(self.max_steps, self.max_epochs*len(tbatches)))
        checkpoints = [int(max_steps*float(cp)/self.checkpoints) for cp in range(1, self.checkpoints+1)]

        self._report_progress(0, cost, tbatches, vbatches)

        for step in xrange(max_steps):
            globals.flags.push("step", step)
            globals.flags.push("train_mode",True)

            # Compute learning rate and momentum for this particular step
            self._update_drate(step, drate)
            self._update_mrate(step, mrate)

            # Report our current performance
            #self._report_step(step, cost, tbatches, vbatches, report)

            # Step the parameters P, and also momentum vector mP
            _ext.gradstep(P, dP, drate,
                             mP, mrate,
                             grad=bprop_trainable, nesterov=self.nesterov)

            enforce_constraints()
            globals.flags.pop("train_mode")

            # Report our current performance
            if step+1 in checkpoints:
                self._report_progress(step+1, cost, tbatches, vbatches)
                checkpoint_callback(step=step+1, 
                                    cost=cost, 
                                    datasets=batchsets)

            globals.flags.pop("step")

        globals.flags.pop("weight_decay_start")
                             
        #sm.sync()
        #print "STOP. Time = ", time.time()-t0
        #quit()

        # Report the final performance on ALL batches
        #globals.flags.push("store_featuremaps",True)
        #tloss,vloss,tauc,vauc = self._report_performance(step, cost, tbatches, vbatches, None, report)
        #globals.flags.pop("store_featuremaps")

        # Affix the final trainable node weights to the inputs of each
        # corresponding plug on the actual model; then we can throw away
        # the outer 'cost' node and just leave the trained/fixed model.
        self._train_teardown(trnodes, cost)

        #stats = {}
        #stats["tloss"] = tloss
        #stats["vloss"] = vloss
        #stats["tauc"] = tauc
        #stats["vauc"] = vauc

        #return stats

    def _report_progress(self, step, cost, tbatches, vbatches):

        max_printwidth = 112

        if step == 0:
            header = "\n      "
            for name in tbatches[0].targetnames:
               header += "  "
               header += name[:13] + " "*max(0,13-len(name))
            print header[:max_printwidth]
            logging.info(header)

        # Get a subset of batch indices that we'll estimate our performance on
        trstats = _report._calc_performance_stats(cost, tbatches[:100])
        vastats = _report._calc_performance_stats(cost, vbatches[:100]) if vbatches is not None else None
        txt = "%06d: " % step
        if vastats:
            txt += "  ".join(["%.4f/%.4f" % (trloss, valoss) for trloss, valoss in zip(trstats["L"], vastats["L"])])
        else:
            txt += "  ".join(["%.4f" % (trloss) for trloss in trstats["L"]])
        
        print txt[:max_printwidth]
        logging.info(txt)


    '''
    def _report_step(self, step, cost, tbatches, vbatches, report):

        update_frequency = self.viz_steps
        if (not update_frequency) or (step % update_frequency != 0):# and step != 0:
            return

        sm.sync()

        now = time.time()
        time_per_step = None
        if self._last_report_step is not None:
            time_per_step = (now-self._last_report_time)/(step-self._last_report_step)/cost.ninst

        # Get a subset of batch indices that we'll estimate our performance on
        if self._report_tbatches is None:
            self._report_tbatches = npr.permutation(len(tbatches))[:max(1,len(tbatches)//10)]
        if vbatches is not None:
            if self._report_vbatches is None:
                self._report_vbatches = npr.permutation(len(vbatches))[:max(1,len(vbatches)//10)]

        self._report_performance(step, cost, 
                                 [tbatches[i] for i in self._report_tbatches], 
                                 [vbatches[i] for i in self._report_vbatches] if vbatches is not None else None, 
                                 time_per_step, report)

        #print "report time = %f" % (time.time() - now)

        self._last_report_step = step
        self._last_report_time = time.time()

    def _report_performance(self, step, cost, tdata, vdata, time_per_step, report):
        if report:
            tloss, vloss, tauc, vauc = report.update(step, cost, tdata, vdata)
        else:
            tloss,tZ,tY,tI = _report._calc_performance_stats(cost,tdata)
            vloss,vZ,vY,vI = _report._calc_performance_stats(cost,vdata) if vdata is not None else (None,None,None,None)
            tauc = None
            vauc = None
            if tdata[0].Y.shape[1] == len(tloss):
                tauc = np.array([_report.calc_auc(tZ[:,i],tY[:,i]) for i in range(tZ.shape[1])])
                vauc = np.array([_report.calc_auc(vZ[:,i],vY[:,i]) for i in range(vZ.shape[1])]) if vdata is not None else None

        for i in range(cost.ninst):
            txt = "%03d:%06d:\t t=%f" % (i,step,float(tloss[i]))
            if vloss is not None:
                txt += " v=%f" % float(vloss[i])
            txt += " r=%.4f" % ((self.rate     if np.isscalar(self.rate)     else self.rate[i]    )*self._last_drate_scale)
            txt += " m=%.2f" % ((self.momentum if np.isscalar(self.momentum) else self.momentum[i])*self._last_mrate_scale)
            if i == 0 and time_per_step:
                txt += " ms=%.3f" % (1000*time_per_step)

            print txt
            logging.info(txt)
        return tloss,vloss,tauc,vauc
        '''

    def _update_drate(self, step, drate):
        # Periodically re-scale the drate vector, depending on how much it's supposed 
        # to have changed since the last time we re-scaled it.
        if True or "disable_rate_schedule" in globals.flags:
            self._last_drate_step = step
            self._last_drate_scale=1.
            return
        i0,i1 = self._last_drate_step, step
        s0 = max(self.rate_decay ** i0,0.2) * (1-.75*(3./(i0+4))) if i0 is not None else 1
        s1 = max(self.rate_decay ** i1,0.2) * (1-.75*(3./(i1+4)))
        reldiff = abs(s1-s0)/min(s1,s0)
        if reldiff >= .10 or (i1-i0 > 50):
            #drate *= s1/s0
            self._last_drate_step = step
            self._last_drate_scale = s1
            #print "step %d drate_scale = %.3f" % (step,s1)

    def _update_mrate(self, step, mrate):
        if True or "disable_rate_schedule" in globals.flags:
            self._last_mrate_step = step
            self._last_mrate_scale=1.
            return
        i0,i1 = self._last_mrate_step, step
        s0 = (1-4./(i0+5)) if i0 is not None else 1
        s1 = (1-4./(i1+5))
        reldiff = abs(s1-s0)/min(s1,s0)
        if reldiff >= .10 or (i1-i0 > 50):
            #mrate *= s1/s0
            self._last_mrate_step = step
            self._last_mrate_scale = s1
            #print "step %d mrate_scale = %.3f" % (step,s1)

    def _train_setup(self, trainable_plugs, cost):

        # For each trainable plug, figure out how many weights it needs.
        sizes  = [ np.prod(p.shape)*cost.ninst for p in trainable_plugs ]
        offsets = np.asarray(np.cumsum([0] + [ size for size in sizes ]),np.uint32)

        # Allocate giant contiguous arrays for P, dP, and mP
        P  = sm.zeros((offsets[-1],1))
        dP = sm.zeros_like(P)
        mP = sm.zeros_like(P)

        # Per-inst learn rates / momentum rates go here.
        # Giant contiguous array maps to same indicies as in P, dP, mP
        drate = sm.zeros_like(P)
        mrate = sm.zeros_like(P)

        trnodes = []

        # For each plug, create a trainable node that is bound to 
        # a chunk of our P (parameter) and dP (gradient) vectors, where the node can 
        for i,tplug in enumerate(trainable_plugs):

            # Grow the actual shape of the trainable parameters, using the
            # axis specified by the trainable plug.
            shape = list(tplug.shape)
            shape[tplug.inst_axis] *= tplug.node.ninst

            # Allocate a new trainable node, and connect it to the plug
            trnode = trainable(P[offsets[i]:offsets[i+1]].reshape(tuple(shape)),
                              dP[offsets[i]:offsets[i+1]].reshape(tuple(shape)))
            trnode >> tplug
            trnodes.append(trnode)

            # Assign instance-specific learning rates and momentum rates
            # to each corresponding element in the giant drate/mrate vectors
            if tplug.inst_axis == 0:
                k = np.prod(tplug.shape)
            else:
                k = tplug.shape[1]
            dratevec = drate[offsets[i]:offsets[i+1]]
            mratevec = mrate[offsets[i]:offsets[i+1]]
            _ext.madd_bcast(sm.ones_like(dratevec),self.rate,k,dratevec)
            _ext.madd_bcast(sm.ones_like(mratevec),self.momentum,k,mratevec)

            # Also initialize elements of P based on the trainable plug's initialization scale,
            # which can be different for each individual instance
            Pvec = P[offsets[i]:offsets[i+1]]
            initval = tplug.origin().node.init
            if isinstance(initval, np.ndarray) and initval.ndim == 3:
                # Specific initialization of individual filters
                Pvec[:] = sm.asarray(np.require(np.rollaxis(initval,1),requirements="C").reshape((-1,1)))
            else:
                # Random initialization
                _ext.madd_bcast(sm.randn(Pvec.shape[0],Pvec.shape[1]),
                                initval,k,Pvec)

            if hasattr(tplug.origin().node,'init_mu'):
                initmu_val = tplug.origin().node.init_mu
                if isinstance(initmu_val, list):
                    # Specific initialization of individual bias elements
                    initmu_val = np.tile(initmu_val,tplug.origin().node.ninst)
                    Pvec[:] = sm.asarray(initmu_val).reshape(Pvec.shape)
                else:
                    _ext.madd_bcast(sm.ones_like(Pvec),
                                    tplug.origin().node.init_mu,k,Pvec)  # Add shift
                

        return (P,dP,mP,drate,mrate,trnodes)


    def _train_teardown(self, trnodes, cost):
        for trnode in trnodes:
            trplug = trnode.Z.dsts[0]
            disconnect(trnode.Z,trplug)
            trplug.fpval = trnode.P

