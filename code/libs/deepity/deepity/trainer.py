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
import logging
import warnings
import numpy as np
from .node import node,supernode
from .plug import plug,disconnect
from .std  import mse,nll,hinge,trainable
from .std.elemwise import logistic
from .std.softmax import softmax
from .     import globals

class costsum(node):
    def __init__(self):  super(costsum,self).__init__(["X"],["cost"])
    def _fprop(self, X): return X  # The incoming values were already accumulated at input plug X, automatically, so nothing to do
    def _bprop(self):    return 1  # Each incoming value 
    def _calc_shapes(self, cost): self.cost._shape = (1,1)

class trainer(object):

    class cost(supernode):
        def __init__(self, model, lossfunc):

            self.model = model
            self.costsum = costsum()

            # First set up subgraph "model -> loss -> costsum"
            if isinstance(model.Z.origin().node,softmax):
                self.loss = nll()    # If user ended model with a softmax, make loss nll implicitly
            elif lossfunc == "hinge":
                self.loss = hinge()
            else:
                self.loss = mse()    # Otherwise, minimize MSE loss
            
            self.model.Z >> self.loss >> self.costsum

            # Find all "cost" oplugs and make sure they get summed at the output
            for p in model.oplugs:
                if p.name.startswith("cost"):
                    p >> self.costsum
            super(trainer.cost,self).__init__([self.model,self.loss,self.costsum])

        def disconnect(self):
            for src in self.iplugs:
                if src.has_dst():
                    dst = src.dsts[0]
                    if dst.node is self.model:
                        disconnect(src,dst)
                        dst.fpval = src.fpval

            for p in self.model.oplugs:
                if p.name.startswith("cost"):
                    disconnect(p,self.costsum)

            disconnect(self.model.Z,self.loss)

        def eval_loss(self, batches):
            if batches is None:
                return None

            loss = None
            for batch in batches:

                # For each data attribute, set its value on the corresponding plug
                for attrname in batch.data_attrs():
                    if hasattr(self, attrname):
                        getattr(self, attrname).fpval = getattr(batch, attrname)
                    else:
                        warnings.warn("Data has attribute \"%s\", but model does not; that particular input will be ignored." % attrname)

                # Pull the final loss value for this minibatch
                batch_loss = self.loss.loss.fpval.asnumpy().ravel()
                if loss is None:
                    loss = np.zeros_like(batch_loss)
                loss += batch_loss 

                # Clear all stored values in the dependency graph, effectively resetting it
                self.clear()
                for attrname in batch.data_attrs():
                    if hasattr(self, attrname):
                        getattr(self, attrname).fpval = None

            loss /= len(batches)

            return loss

        def eval_model(self, batches):
            if batches is None:
                return None

            Z = []
            for batch in batches:

                # For each data attribute, set its value on the corresponding plug
                for attrname in batch.data_attrs():
                    if hasattr(self, attrname):
                        getattr(self, attrname).fpval = getattr(batch, attrname)
                    else:
                        warnings.warn("Data has attribute \"%s\", but model does not; that particular input will be ignored." % attrname)

                # Pull the final loss value for this minibatch
                Z.append(self.model.Z.fpval.asnumpy())

                # Clear all stored values in the dependency graph, effectively resetting it
                self.clear()
                for attrname in batch.data_attrs():
                    if hasattr(self, attrname):
                        getattr(self, attrname).fpval = None

            return np.vstack(Z)

        def bprop_trainable(self, trnodes, datasrc):
            # For each data attribute, set its value on the corresponding plug
            for attrname in datasrc.data_attrs():
                if hasattr(self, attrname):
                    getattr(self, attrname).fpval = getattr(datasrc, attrname)
                #else:
                #    warnings.warn("Data has attribute \"%s\", but model does not; that particular input will be ignored." % attrname)

            # Set the initial bpvalue to be simply 1
            self.cost.bpval = 1

            # Tell each trainable node to update its dP value
            globals.flags.push("bprop_mode",True)
            for trnode in trnodes:
                trnode.bprop()
            globals.flags.pop("bprop_mode")

            # Clear all the stored values in the dependency graph, effectively resetting it
            self.clear()
            for attrname in datasrc.data_attrs():
                if hasattr(self, attrname):
                    getattr(self, attrname).fpval = None

    def train(self, model, datasets, checkpoint_callback):
        tdata = datasets["train"]
        ninst = len(tdata.targetnames)
        cost = trainer.cost(model, self.lossfunc)
        cost.set_ninstance(ninst)

        # Set the shape of all model inputs
        for attrname in tdata.input_attrs():
            #assert hasattr(cost,attrname), "Data has input named \"%s\", but model does not" % attrname
            if hasattr(cost, attrname):
                getattr(cost, attrname).shape = (None,tdata.attrdim(attrname))
            else:
                logging.warn("Data has input named \"%s\", but model does not" % attrname)
        
        # Assert that all model outputs match the size of the corresponding targets, if any
        for attrname in tdata.output_attrs():
            assert hasattr(cost,attrname), "Data has output named \"%s\", but model does not" % attrname
            #assert getattr(cost,attrname).shape[1] == tdata.attrdim(attrname)/ninst, "Model's output attribute \"%s\" (dim=%d) doesn't match corresponding target dimension (dim=%d)" % (attrname,getattr(cost,attrname).shape[1],tdata.attrdim(attrname)/ninst)

        # Collect all input plugs on the cost function that are not 
        # determined by the data inputs/targets
        trainable_plugs = [ p for p in cost.iplugs if (p.origin().trainable and p.shape and p.name not in tdata.data_attrs()) ]

        # Call subclass to do the actual training
        self._train(trainable_plugs, cost, datasets, checkpoint_callback)

        cost.disconnect()
        model._parent = None

    def force_single_value(self,*attrnames):
        for attrname in attrnames:
            attr = getattr(self,attrname)
            if isinstance(attr,np.ndarray):
                if any(attr != attr[0]):
                    logging.info("forcing parameter \"%s\" to single value %s" % (attrname,str(np.asscalar(attr[0]))))
                attr[:] = attr[0]
                setattr(self,attrname,np.asscalar(attr[0]))

