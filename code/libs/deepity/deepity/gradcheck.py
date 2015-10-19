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
import numpy as np
import numpy.random as npr
from .std import trainable
from .    import globals

def gradcheck(trnodes,cost,datasrc,maxtime=2,rtol=1e-5,atol=1e-4):
    """
    Checks the computation of grad() against numerical gradient.
    If any component of the numerical gradient does not match, an 
    AssertionError is raised with information about the problem.
    """
    # Collect each trainable node in the dependency graph
    if len(trnodes) == 0:
        return  # No trainable nodes, means zero-dimensional gradient

#    trnodes[0].P[:] = 0
    print "gradient check (dtype=%s)..." % str(trnodes[0].P.dtype)

    globals.flags.push("train_mode",False)

    # Update the dP value of each trainable node, so that it contains
    # the symbolic gradient for that trainable node's parameters
    cost.bprop_trainable(trnodes,datasrc)

    # Compute components numeric gradient by starting from the symbolic gradient 
    # and then checking several weight components by central-difference
    failures = {}

    data = datasrc.data()
    
    # Loop over each 
    for trnode in trnodes:

        # Get the parameter vector P, and also the symbolic gradient that was computed at the beginning
        P  = trnode.P.ravel()             # Keep as sarray
        dP = trnode.dP.asnumpy().ravel()  # Download

        # Decide on step size and what order to perturb the parameters of this trainable node
        step = 1e-8 if P.dtype == np.float64 else 1e-4
        #order = npr.permutation(len(P)) # Use same permutation every time for consistency in output
        order = np.arange(len(P))

        # For the time allotted, perturb each parameter and evaluate the cost
        starttime = time.time()
        numcheck,mincheck = 0,min(len(order),50)
        for i in order:

            # Temporarily perturb weight i by 'step' and evaluate the new cost at each position
            x = float(P[i])
            P[i] = x-step; c0 = np.sum(cost.eval(**data)["cost"].asnumpy());
            P[i] = x+step; c1 = np.sum(cost.eval(**data)["cost"].asnumpy());
            P[i] = x

            # Compute the numeric gradient for paraneter i, and check its closeness to symbolic gradient
            dc_num = float(c1-c0)/float(2*step)
            dc_sym = dP[i]
            if not np.allclose(dc_num,dc_sym,rtol=rtol,atol=atol) or not np.allclose(dc_sym,dc_num,rtol=rtol,atol=atol):
                if trnode not in failures:
                    failures[trnode] = []
                failures[trnode].append((i,dc_num,dc_sym))

            # Quit early if we ran out of time for this particular node
            numcheck += 1
            if time.time()-starttime >= maxtime and numcheck >= mincheck:
                break
        

    globals.flags.pop("train_mode")

    # No errors? Then simply return
    if len(failures) == 0:
        logging.info("gradient check PASSED")
        return

    msg = "...gradient check FAILED\n"
    for trnode,items in failures.iteritems():
        msg += "   %s FAILED at weights:\n" % (trnode.Z.dsts[0].origin().path)
        # Sort by index
        items.sort(cmp=lambda x,y: int(x[0]-y[0]))
        count = 0
        for index,dc_num,dc_sym in items:
            msg += "\t\t[%d] num: %.8f sym: %.8f\n" % (index,dc_num,dc_sym)
            count += 1
            if count > 8:
                msg += "\t\t...\n"
                break
    
    for trnode in trnodes:
        if trnode not in failures:
            msg += "   %s SUCCEEDED\n" % (trnode.Z.dsts[0].origin().path)

    logging.info("gradient check FAILED")
    logging.info(msg)
    raise AssertionError(msg)
