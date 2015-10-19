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
import gc
import re
import sys
import copy
import time
import random
import tempfile
import logging
import cPickle as cp
import multiprocessing
import subprocess
import deepity
import numpy as np
import numpy.random as npr
import smat as sm
import scipy
import scipy.stats
from . import std
from .        import hpsearch as hp
from .        import _io_
from .        import util
from .data    import datasource
from .        import globals
from .report  import training_report, calc_auc, bootstrap_auc
import node    as _node
import trainer as _trainer

class _object_factory_from_file(object):
    def __init__(self,filename,fieldname=None):
        self.filename = filename
        self.fieldname = fieldname
    
    def __call__(self):
        obj = _io_.load(self.filename)
        if self.fieldname and isinstance(obj,dict):
            return obj[self.fieldname]
        return obj

def _create_model(model_proto, hparams):
    model = copy.deepcopy(model_proto)
    for key,val in hparams.iteritems():
        prefix,path = key.split(":")  # look for hparams named "model:..."
        if prefix == "model":
            nodepath,attrname = path.rsplit(".",1)
            node = model.find(nodepath)
            if hasattr(node,"set_"+attrname):
                getattr(node,"set_"+attrname)(model,val)   # call model.set_xxx(val)
            else:
                setattr(node,attrname,val)
    return model

def _create_trainer(trainer_proto, hparams):
    trainer = copy.deepcopy(trainer_proto)
    for key,val in hparams.iteritems():
        prefix,attrname = key.split(":")  # look for hparams named "trainer:..."
        if prefix == "trainer":
            if hasattr(trainer,"set_"+attrname):
                getattr(trainer,"set_"+attrname)(model,val)   # call trainer.set_xxx(val)
            else:
                setattr(trainer,attrname,val)
    return trainer


def _slice_hparams(hparams, inst):
    h = copy.deepcopy(hparams)
    for key in h.keys():
        h[key] = h[key][inst]
    return h


def load_hparams_result(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
    params = {}
    result = 0.0
    for line in lines:
        # Look for validation performance
        matches = re.findall("# metric = (\S+)", line)
        if len(matches) > 0:
            result = float(matches[0])
            continue
        
        # Add hparam
        name, value = re.findall(" *(\S+) += (\S+)", line)[0]
        if name in [":cfgname"]:
            params[name] = value
        else:
            params[name] = float(value)
        
    return hp.sample(params, result)


def save_hparams_result(filename, hparams_result, metric_key):
    util.makepath(os.path.dirname(filename))
    with open(filename,'w') as f:
        if metric_key:
            f.write("# metric = %f (%s)\n" % (hparams_result.metrics[metric_key], metric_key))
        f.write(hparams2str(hparams_result.params))


def _save_model_inst(filename, inst, model, hparams):
    m = copy.deepcopy(model)
    sm.sync()
    
    # Slice the trainable weights
    m.slice_inst(inst)

    # Also slice the hyperparams, and replace corresponding 'arrayed' 
    # attributes in the model with their scalar (sliced element) counterpart.
    h = _slice_hparams(hparams,inst)
    for key,val in h.iteritems():
        prefix,path = key.split(":")  # look for hparams named "model:..."
        if prefix != "model":
            continue
        nodepath,attrname = path.rsplit(".",1)
        node = m.find(nodepath)
        if hasattr(node,"set_"+attrname):
            getattr(node,"set_"+attrname)(model,val)   # call model.set_xxx(val)
        else:
            setattr(node,attrname,val)

    # Dump the model
    util.makepath(os.path.dirname(filename))
    with open(filename,'wb') as f:
        cp.dump(m,f)

    sm.sync() # Make sure we wait until the sarrays are all dumped
   

def gen_predictions(model, data):
    # We must feed each sequence through the model several times
    # by applying the model repeatedly on sliding a window along the sequence.
    # That generates a prediction map, from which we can take max, sum, etc.
    predictions = []
    gmaps = {}
    batches = data.asbatches(batchsize=128, reshuffle=False)
    for batch in batches:
        args = batch.input_data()   
        args["want_bprop_inputs"] = False
        if isinstance(model.Z.origin().node,std.softmaxnode):
            args["bprop_inputs_loss"] = std.nll()
        else:
            args["bprop_inputs_loss"] = std.mse()
        outputs = model.eval(**args)
        Z = outputs['Z'].asnumpy()
        Zmask = outputs.get('Zmask',None)
        if Zmask is not None:
            Zmask = Zmask.asnumpy()
            Z = Z[Zmask.ravel()]
        predictions.append(Z)

    # Concatenate all numpy arrays if they're the same size
    predictions = np.vstack(predictions)

    return predictions


def getinstdir(outdir, targetname, trialnum, foldid):
    if isinstance(outdir,str):
        return outdir
    outdir = [_ for _ in outdir] # Make a copy that we can substitute elements in
    args = {"target" : targetname, 
            "trial"  : trialnum, 
            "fold"   : foldid}
    for i,item in enumerate(outdir):
        if isinstance(item, tuple):
            name, patt = item
            if args[name] is None:
                outdir[i] = None
            else:
                outdir[i] = patt % args[name]
    instdir = "/".join([part for part in outdir if part is not None])
    return instdir

def load_metrics(filename):
    metrics = {}
    with open(filename) as f:
        groupnames = f.readline().rstrip().split()
        for line in f:
            line = line.rstrip().split()
            for i,val in enumerate(line[1:]):
                metrics.setdefault(groupnames[i],{})[line[0]] = val
    return metrics

def save_metrics(outfile, metrics):
    with open(outfile,"w") as f:
        groupnames = sorted(metrics.keys())
        fieldnames = set()
        for groupname in groupnames:
            for fieldname in metrics[groupname].keys():
                fieldnames.add(fieldname)
        fieldnames = sorted(list(fieldnames))

        f.write(" "*14+"\t".join(groupnames) + "\n")
        rows = {}
        for groupname in groupnames:
            for fieldname in fieldnames:
                fieldval = metrics[groupname].setdefault(fieldname, np.nan)
                if not isinstance(fieldval,np.ndarray):
                    if isinstance(fieldval, float):
                        fmt = "%.2e" if fieldname.endswith(".p") else "%.6f"
                        fieldval = fmt%fieldval
                    rows.setdefault(fieldname,[]).append(str(fieldval))
        f.writelines([fieldname + " "*max(0,14-len(fieldname)) + "\t".join(rows[fieldname]) +"\n" for fieldname in fieldnames])

def call_dumpviz(dumpdir):
    subprocess.Popen(["python", os.path.dirname(__file__)+"/dumpviz.py", dumpdir])

##########################################



class hypertrain_worker(object):
    """
    Given a dataset and specific hyperparameters, this object will
    simply train a model (an array of models) and return the 
    validation error (array of validation errors).
    """
    def __init__(self, worker_id, model_proto, trainer_proto, datasrc, 
                                  nfold, allfolds, outdir, report_class, devices, verbose, 
                                  default_dtype, global_flags, auxfilter, mode, dumpviz):

        self.worker_id = worker_id
        # All the data subsets in 'trainset' will be merged into a single fold.
        self.model_proto   = model_proto
        self.trainer_proto = trainer_proto
        self.datasrc = datasrc    # Load a copy of the dataset into this worker process.
        self.nfold  = nfold
        self.allfolds = allfolds
        self.outdir  = outdir
        self.mode = mode
        self.aucrange = (0.5,0.5)  # threshold for making AUCs out of non-binary targets, presumed to be in range [0,1]
        self.report_class = report_class
        self.auxfilter = auxfilter
        self.dumpviz = dumpviz

        globals.flags.copy_from(global_flags)

        # If we've been called from a new process, create a separate log file.
        # Otherwise everything is logged into the original log file.
        if multiprocessing.current_process().name != "MainProcess":
            logdir = getinstdir(outdir,None,None,None)
            worker_logfile = os.path.join(logdir,"hypertrain_worker%d.log" % worker_id)
            globals.set_logging(worker_logfile,level=verbose,echo=False)
            logging.info("\n----------------------------- %s -----------------------------" % time.strftime("%y-%m-%d %H-%M-%S",time.localtime()))


        # Configure deepity to use this worker's GPU device.
        logging.info("worker %d starting on device %d using %s" % (worker_id,devices[worker_id],sm.get_default_dtype().__name__))
        rseed = int((time.time()*100000 + worker_id)  % 2000)
        globals.reset_backend(device=devices[worker_id], seed=rseed)
        random.seed(rseed)
        sm.set_default_dtype(default_dtype)
        npr.seed(rseed)

        # Seed this process's random number generator, for reproducibility
        sm.sync()

        # Prepare the datasource to serve data.
        self.datasrc.open()

    def __del__(self):
        self.datasrc.close()
        self.datasrc = None
        gc.collect()  # Clear out the cruft and make sure the backend can be destroyed
        sm.sync()
        sm.destroy_backend()


    def __call__(self, hparams, task_ids, sample_ids):

        # Determine what kind of targets we want to train on
        data = self.datasrc.astargets(task_ids) # Copies of arbitrary targets
        data = data[:] # Copy so that when we normalize etc we don't affect the original data

        # Normalize the targets. For logisitic-output models this means
        # scaling targets to [0,1]. For other models this means scaling
        # targets to have mean=0, variance=1.
        data.requirements = self.model_proto.data_requirements()
        #print np.percentile(data.Y[data.Ymask].ravel(), [99, 99.99, 99.995, 99.999])
        #print data.Y.size, int(data.Y.size*(100-99.95)/100)
        if "clamp_targets" in globals.flags:
            data.clamp_extremes(0.0,99.95)
        if "normalize_targets" in globals.flags:
            data.normalize_targets()
            #data.arcsinhtransform_targets()

        if self.mode != 'calib':
            # If we're not in calibration mode, then there's no need for multiple checkpoints
            #  -- just keep the last checkpoint so that it can be dumped to disk
            #del hparams["trainer:checkpoints"]
            self.trainer_proto.checkpoints = 1

        # Shuffle the individual rows of data, always the same random shuffle
        # and therefore always the same random split each time the code is run.
        data.shuffle()
        
        # Create a callback handler to collect predictions and evaluate final performance
        checkpoints = self.report_class()

        # Perform k-fold cross validation (k=nfold), training with one fold held out at a time.
        for foldid in range(self.nfold):
            checkpoints.setfold(foldid)  # Tell the checkpoint 
            
            # Create a new model and trainer with the given hyperparams
            model   = _create_model(self.model_proto,     hparams)
            trainer = _create_trainer(self.trainer_proto, hparams)

            # Split the data into training and validation sets
            trdata, vadata = data.split(foldid, self.nfold-1)
            trdata = trdata.augmented(trdata)
            datasets = { "train" : trdata }
            if vadata:
                vadata = vadata.augmented(vadata)
                datasets["validate"] = vadata
                if self.auxfilter:
                    datasets["validate_aux"] = vadata[[i for i in range(len(vadata)) if vadata.foldids[i] in self.auxfilter]]
            for dset in datasets.values():
                dset.requirements = model.data_requirements()

            #if not checkpoint_callback:
            #    trainer.viz_steps = False # Disable periodic updates if no reports

            # Train the model and remember how well it performed.
            trainer.train(model, datasets, checkpoints)

            if self.mode == 'train' and self.nfold > 1:
                entries = checkpoints.curr()
                metrics = self.calc_metrics(entries)
                self.save_model(model, hparams, task_ids, sample_ids, foldid)
                self.save_metrics(metrics, task_ids, sample_ids, foldid)
                self.save_predictions(entries, task_ids, sample_ids, foldid)
                self.call_dumpviz(task_ids, sample_ids, foldid)

            # If we`re only supposed to try one fold, then don`t bother looping over the other splits
            if not self.allfolds:
                break

        # Consolidate the separate folds, and dump them if need be
        entries = checkpoints.combined()

        # Calculate the performance stats associated with each target
        metrics = self.calc_metrics(entries)

        # Save the current model and predictions
        if self.mode == 'train':
            self.save_predictions(entries, task_ids, sample_ids, None)
            self.save_metrics(metrics, task_ids, sample_ids, None)
            if self.nfold == 1:
                self.save_model(model, hparams, task_ids, sample_ids, None)
                self.save_preprocessors(data, task_ids, sample_ids, None)
            #self.call_dumpviz(task_ids, sample_ids, None)


        # Return a new hparams object with the performance incorporated
        hpsearch_result = self.add_hparam_metrics(hparams, metrics)
        return hpsearch_result


    def save_model(self, model, hparams, task_ids, sample_ids, foldid):
        for i, taskid in enumerate(task_ids):
            dumpdir = getinstdir(self.outdir, taskid, sample_ids[i], foldid)
            util.makepath(dumpdir)
            
            # Slice out model i and save it to disk
            _save_model_inst(dumpdir+"/model.pkl", i, model, hparams)


    def save_predictions(self, entries, task_ids, sample_ids, foldid):
        for i, taskid in enumerate(task_ids):
            dumpdir = getinstdir(self.outdir, taskid, sample_ids[i], foldid)
            util.makepath(dumpdir)

            # Save out the predictions for model i
            assert len(entries[i]) == 1, "Bug. Expected only a single unique 'step' in the list of entries"
            groups = entries[i].values()[0]
            np.savez_compressed(dumpdir+"/predict.npz", 
                                targetname=np.asarray(taskid, dtype=object), 
                                groups=np.asarray(groups, dtype=object))


    def save_metrics(self, metrics, task_ids, sample_ids, foldid):
        for i, taskid in enumerate(task_ids):
            dumpdir = getinstdir(self.outdir, taskid, sample_ids[i], foldid)
            util.makepath(dumpdir)

            # Save out the predictions for model i
            assert len(metrics[i]) == 1, "Bug. Expected only a single unique 'step' in the list of entries"
            groups = metrics[i].values()[0]
            save_metrics(dumpdir+"/metrics.txt", groups)
        

    def call_dumpviz(self, task_ids, sample_ids, foldid):
        if not self.dumpviz:
            return
        for i, taskid in enumerate(task_ids):
            dumpdir = getinstdir(self.outdir, taskid, sample_ids[i], foldid)
            call_dumpviz(dumpdir)


    def save_preprocessors(self, data, task_ids, sample_ids, foldid):
        for i, taskid in enumerate(task_ids):
            dumpdir = getinstdir(self.outdir, taskid, sample_ids[i], foldid)
            data.dump_preprocessors(dumpdir, slice(i,i+1))


    def add_hparam_metrics(self, hparams, metrics):
        groupkey = "validate" if "validate" in metrics[0].values()[0] else "train"
        hpsearch_result = {}
        for i in metrics:
            for step in metrics[i]:
                hparams_i = { key : val[i] for key,val in hparams.iteritems() }
                hparams_i["trainer:max_steps"] = step
                metrics_i = metrics[i][step][groupkey]
                hpsearch_result.setdefault(i,[]).append((hparams_i, metrics_i)) # Thus tuple is returned to hpsearch
        return hpsearch_result



        """
            if "vloss" in stats and stats["vloss"] is not None:
                loss.append(stats["vloss"])
                auc.append(stats["vauc"])
            else:
                loss.append(stats["tloss"])
                auc.append(stats["tauc"])


            if self.testfilter is not None:
                tidx = [i for i in range(len(vdata)) if vdata.foldids[i] in self.testfilter]
                tdata = vdata[tidx]
                tpred = gen_predictions(model, tdata)
                testauc,teststd = bootstrap_auc(tpred.ravel(), tdata.Y.ravel(), ntrial=20)
                flogfile = self.outdir + "/%s_%04d/fold%d.log" % (task_ids[0], sample_ids[0], foldid)
                with open(flogfile) as fh:
                    flog = fh.readlines()
                flog[-1] = flog[-1].rstrip() + "\ttestAUC=%.3f (%f)\n" % (testauc,teststd)
                with open(flogfile,"w") as fh:
                    fh.writelines(flog)
                testaucs.append((testauc, teststd))

            if report:
                reports.append(report)
                report.dump(want_html=True)
                #report.dump(want_html=self.want_html)

                # Dump each model to a separate file
                for inst in range(len(sample_ids)):
                    filename = self.outdir + ("/%s_%04d/fold%d.model.pkl" % (task_ids[inst], sample_ids[inst], foldid))
                    _save_model_inst(filename, inst, model, hparams)
                    
            """
            #break
        """"
        if reports != []:
            # Dump the separate (individual) hyperparams that were used for each instance trained
            for inst in range(len(sample_ids)):
                dumpdir = self.outdir + ("/%s_%04d/" % (task_ids[inst], sample_ids[inst]))
                vloss = self.validation_performance[task_ids[inst]] if self.validation_performance else None
                _dump_hparams(dumpdir, _slice_hparams(hparams,inst), vloss)
                tdata.dump_preprocessors(dumpdir, slice(inst,inst+1))
            merged = self.report_class.merge_reports(self.outdir + "/%(task_id)s_%(sample_id)04d/final.log", task_ids, sample_ids, reports)
            #merged.dump(want_html=self.want_html)
            merged.dump()
            if testaucs:
                flogfile = self.outdir + "/%s_%04d/final.log" % (task_ids[0], sample_ids[0])
                with open(flogfile) as fh:
                    flog = fh.readlines()
                testauc = sum([_auc for _auc, _std in testaucs]) / len(testaucs)
                teststd = sum([_std for _auc, _std in testaucs]) / len(testaucs)
                flog[-1] = flog[-1].rstrip() + "\ttestAUC=%.3f (%f)\n" % (testauc,teststd)
                with open(flogfile,"w") as fh:
                    fh.writelines(flog)

        # Average the loss over each fold
        loss = np.mean(np.asarray(loss),axis=0)
        auc  = np.mean(np.asarray(auc),axis=0)

        # Dump each average loss and corresponding hyperparameters into a log file
        for inst in range(len(sample_ids)):
            util.makepath(self.outdir+"/hpsearch")
            with open(self.outdir+"/hpsearch/%s.log"%task_ids[inst],"a") as f:
                f.write("%.6f\t%.4f\t%s\n"%(loss[inst], auc[inst], hparams2str( _slice_hparams(hparams,inst) ).replace("\n",";")) )
                """
        sm.sync()

        # Return a list of objective values, one per search_id
        values = [float(x) for x in loss]
        return values


    def calc_metrics(self, entries):
        metrics = {}
        for taskidx in entries:
            for step in entries[taskidx]:
                for group in entries[taskidx][step]:
                    entry = entries[taskidx][step][group]
                    Z = entry["Z"]
                    Y = entry["Y"]

                    # Start computing stats
                    metric = metrics.setdefault(taskidx,{}).setdefault(step,{}).setdefault(group,{})
                    metric["loss"] = entry["L"]
                    if Z.shape[1] == 1:
                        metric.update(deepity.calc_metrics(Z.ravel(), Y.ravel(), self.aucrange))
        return metrics

def hparams2str(params):
    txt = ""
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, np.ndarray) and value.size > 10:
            value = "ndarray"
        txt += "  %s = %s\n" % (key + " "*max(0,20-len(key)),value)
    return txt

#######################################

def hypertrain(model, trainer, data,
               nfold=2, allfolds=True, outdir=None, nsample=20,
               devices=None, verbose=None, report_class=None,
               auxfilter=None):

    if report_class is None: report_class = training_report

    # Create the output directory if it doesn't already exist.
    if outdir is None:
        outdir = join(tempfile.gettempdir(),"hypertrain")

    # Define the search space
    space = _get_hypertrain_searchspace(model, trainer)

    # Perform the search, returning the best parameters in the search space.
    logging.info("calibrating...")
    samples = hp.search(space, 
                        objective  = hypertrain_worker, 
                        objective_initargs = (model,trainer,data,nfold,allfolds,outdir,report_class,devices,False,sm.get_default_dtype(),globals.flags,auxfilter,"calib",False),
                        task_ids = data.targetnames,
                        nsample  = nsample,
                        nprocess = len(devices),
                        nsample_per_process = 15,
                        print_progress = True)
    logging.info("...calibrating done")

    return samples

###########################################

def train(model, trainer, data, hparams=None, hparams_metric=None,
          nfold=1, outdir=None, nsample=1, 
          devices=None, verbose=None, report_class=None,
          auxfilter=None, dumpviz=True):

    if report_class is None: report_class = training_report

    if hparams:
        for targetname in data.targetnames:
            for sample in range(nsample):
                for fold in range(nfold):
                    save_hparams_result(getinstdir(outdir, targetname, sample, fold)+"/calib.txt", hparams[targetname], hparams_metric)


    space = _get_fixed_searchspace(model, trainer, data.targetnames, hparams)

    #space = _get_hypertrain_searchspace(model, trainer)

    #if space and not hparams:
    #    raise ValueError("The given model has undetermined hyperparamters. Must call hypertrain first.")

    # Replace the randomly sampled hparams with fixed values specified by 'hparams'
    #for pname in space._pdefs.iterkeys():
    #    pbest = np.asarray([hparams[task_id].params[pname] for task_id in data.targetnames])
    #    space._pdefs[pname] = hp.fixed(pbest, pname)
        #print "assigning hparam",pname,"<-",pbest

    final_outdir = outdir
    logging.info("train...")
    hp.search(space, 
              objective  = hypertrain_worker, 
              objective_initargs = (model,trainer,data,nfold,True,final_outdir,report_class,devices,verbose,sm.get_default_dtype(),globals.flags,auxfilter,"train",dumpviz),
              task_ids = data.targetnames,
              nsample  = nsample,
              nsample_per_process = 2,#len(data.targetnames),  # Hack: only train numtargets models at a time, to ensure that when nsample>1 the next sample gets a different minibatch order
              nprocess = len(devices))
    logging.info("...train done")

#######################################################

def _get_fixed_searchspace(model, trainer, targetnames, hparams):
    pdefs = []
    if hparams:
        # Convert the hparams list-of-dictionaries (all dictionaries having same key) 
        # into a single dictionary-of-lists
        hpvec = {}
        for targetname in targetnames:
            sample = hparams[targetname]
            for pkey in sample.params:
                hpvec.setdefault(pkey,[]).append(sample.params[pkey])
        for key in hpvec:
            pdefs.append(hp.fixed(np.array(hpvec[key]), key))
    space = hp.space(pdefs)
    return space



def _get_hypertrain_searchspace(model, trainer):
    # First, collect all hparams by visiting the model's dependency graph
    model_hparams = []
    def collect_hparam(path,attr):
        if isinstance(attr,hp.paramdef):
            attr.name = "model:" + path  # model:...path
            model_hparams.append(attr)
    model.visit(collect_hparam)

    # Next, ask the trainer for its hyperparams, and put a "trainer." prefix on the name of each one
    # so that they don't conflict with model_hparams
    trainer_hparams = []
    for name,attr in trainer.__dict__.iteritems():
        if isinstance(attr,hp.paramdef):
            attr.name = "trainer:" + name  # trainer:...path
            trainer_hparams.append(attr)

    # Return a search space built from model and trainer hyperparams
    return hp.space(trainer_hparams + model_hparams)


