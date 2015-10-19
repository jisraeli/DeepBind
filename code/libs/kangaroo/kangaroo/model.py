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
import deepity
from . import _ext
import smat as sm

###############################################################

def loadcfg(cfgfile, *args):
    return deepity.io.load(cfgfile, *args)


class sequencenet(deepity.supernode):
    """
    A basic model where the input is:
         - several sequences X0..Xk
         - a single feature vector F
    The output is computed by convolving a separate
    convnet over each input sequence, then stacking the 
    output of each convnet along with F, and sending it through
    the outputnet to generate prediction Z.

                               
             X0 -- convnet0 --+-- outputnet -> Z 
             ...              |
             Xk -- convnetk --+
                              |
             F -------------- + 
    """

    def __init__(self, convnets, outputnet, combiner_size=None, combiner_decay=None, combiner_init=None, combiner_init_mu=None, combiner_start=None, featurenet_size=32):

        # By default, the combiner layer jas as many hidden units as there are featuremaps
        if not combiner_size:
            combiner_size = sum([convnet.Z.shape[1] for convnet in convnets])

        # Create a combiner node that does two things:
        #   1. it implicitly concatenates all the input matrices, 
        #      but does so in a way that suppports self.ninst>1
        #      (i.e. it is aware that groups of columns of input matrices 
        #       interleave separate model instances)
        #
        #   2. acts as a fully-connected layer between the concatenated
        #      inputs and the combiner's output.
        #
        self.combiner = deepity.std.combine(len(convnets)+1, 
                                            size   = combiner_size,
                                            ishape = (1,-1),
                                            decay  = combiner_decay,
                                            init   = combiner_init,
                                            init_mu = combiner_init_mu,
                                            start_training = combiner_start)
        self.outputnet = outputnet
        self.combiner.Z >> outputnet.X # Connect combiner node to the outputnet's input

        # Connect the convnets to the combiner's inputs.
        # Each output plug "convnet[i].Z" is connected input plug "combiner.Xi"
        for i,convnet in enumerate(convnets):
            convnet_attrname = "conv"+("_"+convnet.name if convnet.name else str(i))
            self.__dict__[convnet_attrname] = convnet
            convnet.Z >> getattr(self.combiner, "X%d"%i) # Connect to the ith input attribute of combiner

        # Create a linear node that:
        #   - Has an input plug X that will be renamed "F" and thereby
        #     connected to the separate "features" vector.
        #   - Forwards the features vector to the combiner's LAST input 
        #     position (hence the len(convnets)+1)
        # 
        self.featurenet = deepity.std.linear()
        """
        self.featurenet = deepity.std.chain([
                        deepity.std.full(size = featurenet_size,
                                         init = 0.005,
                                         oshape = (1,featurenet_size)),
                        deepity.std.bias(viz = True),
                        deepity.std.relu()
                        ])
        self.featurenet[0].X.trainable = False
        """
        self.featurenet.Z >> getattr(self.combiner, "X%d"%len(convnets))

        # Call supernode constructor to create the (renumbered) public plugs.
        super(sequencenet,self).__init__(convnets + [self.featurenet, self.combiner, self.outputnet])

        # Finally, rename some of the plugs after their parent convnet node,
        # so that the datasource's attributes will automatically 
        # be connected to the  convnet with the matching plug name.
        # Convnet i's input plug "convnet[i].X" will end up 
        # being named "Xi" on this supernode.
        for i,convnet in enumerate(convnets):
            Xp = getattr(self,"X%d"%i)
            Rp = getattr(self,"R%d"%i,None) or getattr(self,"R")
            assert convnet.path in Xp.origin().node.path
            Xp.rename(("X_%s" % convnet.name) if convnet.name else ("X%d"%i))
            Rp.rename(("R_%s" % convnet.name) if convnet.name else ("R%d"%i))

        # Rename the featurenet's input attribute "F"
        getattr(self,"X%d"%len(convnets)).rename("F")

        return
