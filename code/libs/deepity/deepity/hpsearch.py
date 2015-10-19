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
import sys
import time
import signal
import logging
import exceptions
import traceback
import warnings
import multiprocessing
import numpy as np
import numpy.random as npr
from os.path import exists,dirname,join
from . import globals

class local_dummy_pool(object):

    class job_handle(object):
        def __init__(self, objective, args):
            self.objective = objective
            self.args = args

        def get(self, timeout):
            args = self.args
            return _call_objective(*args)

    def __init__(self, objective, initargs):
        _init_objective(objective, initargs, 1)

    def apply_async(self, objective, args):
        return self.job_handle(objective, args)

    def close(self): pass
    def join(self): pass
    def terminate(self): pass


#################################################

def _makepath(dir):
    """
    Makes a complete path if it does not exist already. 
    Does not remove any existing files.
    Fixes periodic failure of os.makedirs on Windows.
    """
    if os.path.exists(dir):
        return dir
    retry = True
    while retry:
        try:
            time.sleep(0.001)
            os.makedirs(dir)
        except WindowsError, e:
            if e.errno != 13: # eaccess
                raise
        else:
            retry = False
    return dir

#################################################

class paramdef(object):
    def __init__(self,name,dtype):
        self.name = name
        self.dtype = dtype

class choice(paramdef):
    """A discrete choice, taken from a list of choices, e.g. [-1,0,1] or ['a','b','c']"""
    def __init__(self,choices,name=None):
        assert len(choices) > 0
        self.choices = np.asarray(choices)
        super(choice,self).__init__(name,self.choices.dtype)

    def sample(self,task_idx,sample_ids):
        return np.asarray([self.choices[npr.randint(0,len(self.choices))] for id in sample_ids])

class uniform(paramdef):
    """A float interval [lo,hi) sampled uniformly."""
    def __init__(self,lo,hi,name=None):
        assert lo <= hi
        super(uniform,self).__init__(name,np.float32)
        self.lo = lo
        self.hi = hi

    def sample(self,task_idx,sample_ids):
        return np.asarray([float(npr.rand()*(self.hi-self.lo) + self.lo) for id in sample_ids])

class loguniform(paramdef):
    """A float interval [lo,hi) sampled as 10**uniform(log10(lo),log10(hi))."""
    def __init__(self,lo,hi,name=None):
        assert lo <= hi
        super(loguniform,self).__init__(name,np.float32)
        self.lo = lo
        self.hi = hi

    def sample(self,task_idx,sample_ids):
        lo = np.log10(self.lo)
        hi = np.log10(self.hi)
        return np.asarray([10**(npr.rand()*(hi-lo) + lo) for id in sample_ids])

class powuniform(paramdef):
    """A float interval [lo,hi) sampled as lo+(hi-lo)*uniform(0,1)**p for 0 < p <= 1"""
    def __init__(self,lo,hi,p,name=None):
        assert lo <= hi
        assert 0 < p and p <= 1
        super(powuniform,self).__init__(name,np.float32)
        self.lo = lo
        self.hi = hi
        self.p = p

    def sample(self,task_idx,sample_ids):
        lo = self.lo
        hi = self.hi
        p = self.p
        return np.asarray([float((npr.rand()**p)*(hi-lo) + lo) for id in sample_ids])

class fixed(paramdef):
    """
    Used during training of "final" models; allows a specific 
    vector of params to be used, presumeably the best parameters
    from an earlier hyperparameter search.
    """
    def __init__(self,values,name=None):
        super(fixed,self).__init__(name,values.dtype)
        self.values = values

    def sample(self,task_idx,sample_ids):
        return self.values[task_idx]

#################################################

class space(object):
    """
    A search space is just a collection of paramdefs.
    The sample() function is a convenient way to sample from all paramdefs.
    """
    def __init__(self,paramdefs=None):
        self._pdefs = {}
        self.update(paramdefs)

    def pnames(self):
        return self._pdefs.keys()

    def update(self,paramdefs):
        for p in paramdefs:
            assert isinstance(p,paramdef), "Expected list of paramdef instances."
        self._pdefs.update({ p.name : p  for p in paramdefs })
        self.__dict__.update(self._pdefs)

    def __len__(self):        return len(self._pdefs)
    def __iter__(self):       return self._pdefs.itervalues()
    def __getitem__(self,i):  return self._pdefs[i]
    def empty(self):          return len(self) == 0

    def sample(self,task_ids,sample_ids):
        return { name : p.sample(task_ids,sample_ids)  for name,p in self._pdefs.iteritems() }

#################################################

class sample(object):
    """
    A hyperparameter search sample.
    """
    def __init__(self,params,metrics):
        self.params  = params
        self.metrics = metrics

#################################################

class sampleset(object):
    """
    A list of (params,result) pairs, where 
       - params is a dictionary of hyperparameter values used to generate the sample.
       - result is the performance using those hyperparameters.

    The (params,result) values are echoed to a text file (JSON format)
    and can also be loaded from the same file.
    """
    def __init__(self,space):
        self.space = space
        self._samples = []

    def add(self,params,metrics):
        self._samples.append(sample(params,metrics))

    def __len__(self):        return len(self._samples)
    def __getitem__(self,i):  return self._samples[i]
    def __iter__(self):       return self._samples.__iter__()

    def get_all(self):
        """Returns the entire list of samples."""
        # SIMPLE VERSION (one parameter per value)
        return self._samples

def get_best_sample(samples, metrickey, wantmax=False):
    """Returns the best params from the entire list of samples."""
    # SIMPLE VERSION (one parameter per value)
    assert len(samples) > 0, "Cannot call get_best on an empty sampleset."
    best = None
    for i in range(len(samples)):
        if not np.isnan(samples[i].metrics[metrickey]):
            if best is None:
                best = i
            else:
                a = samples[i].metrics[metrickey] 
                b = samples[best].metrics[metrickey]
                if (wantmax and a > b) or (a < b and not wantmax):
                    best = i

    return samples[best]
        

#################################################

# _search_objective
#   Each process gets its own copy of this variable, and it is
#   instantiated when 
global _search_objective
global _search_objective_init_err

# Sets up the _search_objective global variable so that,
# in the future, calling _eval_objective(params) will invoke
# _search_objective(params). We need the _eval_objective 
# "wrapper" so that _search_objective can be a class instance,
# because multiprocessing cannot pickle a class instance as an
# entry point for a task -- it can only pickle a global function.
def _init_objective(objective,objective_args,nprocess):
    global _search_objective
    global _search_objective_init_err
    _search_objective = None
    _search_objective_init_err = None

    sys.exc_clear()
    try:
        if globals._allow_multiprocessing:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        if isinstance(objective, type):
            process = multiprocessing.current_process()
            if "-" in process.name:
                process_type,process_id = process.name.split("-")           # Get unique 1-base "process_id", which gets larger every time a new pool is created
                worker_id = (int(process_id)-1) % nprocess                 # Get unique 0-based "worker_id" index, always in range {0,...,nprocess-1}
            else:
                worker_id = 0
            _search_objective = objective(worker_id, *objective_args)
        else:
            _search_objective = objective
    except BaseException as err:
        _search_objective = None     # or else multiprocessing.Pool will get stuck trying to initialize
        traceback_str = traceback.format_exc()
        _search_objective_init_err = (err,traceback_str)
        logging.info(err.message + "\n" + traceback_str)    # Do not allow the error to propagate during _call_objective,
        if not globals._allow_multiprocessing:
            raise

# A global function entry-point for calling _search_objective
# from a multiprocessing task
def _call_objective(params,task_ids,sample_ids):
    global _search_objective
    global _search_objective_init_err
    sys.exc_clear()
    if _search_objective_init_err:
        return _search_objective_init_err
    try:
        # Return the params too incase they were modified by _search_objective
        # (this can happen when _search_objective has to force certain param vectors
        # to have the same uniform value for this particular sample.
        result = _search_objective(params, task_ids, sample_ids)

    except BaseException as err:
        _search_objective = None     # or else multiprocessing.Pool will get stuck trying to initialize
        traceback_str = traceback.format_exc()
        logging.info(err.message + "\n" + traceback_str)    # Do not allow the error to propagate during _call_objective,
        if not globals._allow_multiprocessing:
            raise
        return (err,traceback_str)
    
    return (params,result)

#################################################

def _search_random(space, objective, objective_initargs, nsample, task_ids, pool, samples, nsample_per_process, print_progress=False):

    # Max number of samples to evaluate in each 'job' for the multiprocessing pool
    # TODO: THIS NUMBER IS VERY IMPORTANT, and should be chosen BASED ON MEMORY USAGE
    #       of the MODEL. Right now set by hand :( :( :(
    ntask = len(task_ids)
    nsample_per_job = min(nsample_per_process, nsample*ntask)

    # Ask the pool of subprocesses to evaluate the objective
    jobs = []
    for s in range(0,nsample*ntask,nsample_per_job):
        job_sample_ids = range(s, min(s+nsample_per_job, nsample*ntask))
        job_task_idx = [(i % ntask) for i in job_sample_ids]
        job_task_ids = [task_ids[i] for i in job_task_idx]
        job_params   = space.sample(job_task_idx, job_sample_ids)
        job_handle   = pool.apply_async(_call_objective, (job_params,job_task_ids,job_sample_ids))
        jobs.append((job_task_ids, job_handle))

    # Wait for all the results to complete, and store each within its corresponding sampleset
    jobs_complete = 0
    start_time = time.time()
    for job_task_ids,job_handle in jobs:
        jobret = job_handle.get(1000000) # huge timeout is workaround for multiprocessing bug where a KeyboardInterrupt causes everything to hang
        if isinstance(jobret[0], Exception):
            worker_exception, traceback_str = jobret
            quit("Error in Worker...\n" + worker_exception.message + "\n" + traceback_str)

        params,results = jobret
        for i in results:
            for params_i, metrics_i in results[i]:
                samples[job_task_ids[i]].add(params_i, metrics_i)  # Store a (params,result) sample for this objective
        #for i in range(len(job_task_ids)):
        #    params_i = { key : val[i] for key,val in params.iteritems() }
        #    samples[job_task_ids[i]].add(params_i,results[i])  # Store a (params,result) sample for this objective
        
        jobs_complete += 1
        if print_progress:
            percent_complete = float(jobs_complete) / len(jobs)
            print "------- %.1f%% complete -------" % (100 * percent_complete)
            sys.stdout.flush()

#################################################

def search(space, objective, objective_initargs = None, strategy = "random",
           nsample = 20, nsample_per_process = 10, task_ids = None, nprocess = None,
           print_progress = False):
    """
    Minimize 'objective' over the given search space.
    If objective is a class type, then an instance of that class will be created 
    with __init__(worker_id,**initargs) passed to the constructor
    """
    # Each objective can be multi-task, where ntask
    # Each task gets "nsample" random parameters
    global _search_objective

    # Create a pool of processes.
    # If no processes were requested, just use a thread instead (multiprocessing.dummy)
    # to avoid some of the limitations of multiprocessing (pickling, pipes, etc).
    if not globals._allow_multiprocessing:
        warnings.warn("PerformanceWarning: multiprocessing is disabled; all jobs will be run from main process.")
        pool = local_dummy_pool(objective=objective, initargs=objective_initargs)
    else:
        pool = multiprocessing.Pool(processes=nprocess, initializer=_init_objective, initargs=(objective,objective_initargs,nprocess))

    samples = { id : sampleset(space) for id in task_ids }
    try:
        # Perform the actual search according to the selected strategy.
        if "random" in strategy.split():
            _search_random(space, objective, objective_initargs, nsample, task_ids, pool, samples, nsample_per_process, print_progress)

        # TODO: other strategies like tree parzen window

    except:
        pool.terminate()
        pool.join()
        _search_objective = None
        raise
    else:
        pool.close() # Terminate the subprocesses
        pool.join()
        _search_objective = None


    return { task_id : s.get_all()  for task_id,s in samples.iteritems() }


