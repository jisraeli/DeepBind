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
from ..node    import node
from ..trainer import trainer
from .. import std
import numpy
import re

def _hpsearch_not_found_error(self,i):
    raise ImportError("Cannot create hpsearch object; the hpsearch module could not be imported.")



def _all_subclasses(basetype):
    subtypes = basetype.__subclasses__()
    subtypes += sum([_all_subclasses(st) for st in subtypes],[])
    return subtypes

def convert_cfg_to_instance(cfg, cfg_globals):
    # hack that totally breaks any last semblance of 'modularity' but nonetheless makes config files simpler-looking
    if not isinstance(cfg, list):
        return cfg

    is_after_pooling = False
    for i in range(len(cfg)):
        if type(cfg[i]).__name__ in ("allpool", "maxpool", "avgpool"):
            is_after_pooling = True
        if type(cfg[i]).__name__ in ("full"):
            combiner_layer = i
            break

    convnet = std.chain(cfg[:combiner_layer], name="seq")
    combiner = cfg[combiner_layer]
    outputnet = std.chain(cfg[combiner_layer+1:])

    return cfg_globals["sequencenet"]([convnet], outputnet,
                        combiner_size  = combiner.size,
                        combiner_decay = combiner.decay,
                        combiner_init  = combiner.init,
                        )



##########################################################################

def load(filename, *args):
    assert type(filename)==str, "Expected a string filename."

    # Set up the namespace for config files, including all possible 
    # node types, and (if available) all possible hyperparameter types
    load_cfg_locals  = {}
    load_cfg_globals = {}
    load_cfg_globals.update({ ntype.__name__ : ntype for ntype in _all_subclasses(node)})
    load_cfg_globals.update({ ttype.__name__ : ttype for ttype in _all_subclasses(trainer)})
    load_cfg_globals.update({'numpy' : numpy })
    try:
        from .. import hpsearch
        hparam_types = { name : getattr(hpsearch,name)  for name in dir(hpsearch)
                                                        if type(getattr(hpsearch,name)) == type and issubclass(getattr(hpsearch,name),hpsearch.paramdef) }
        load_cfg_globals.update(hparam_types)
    except:
        load_cfg_globals.update({ name : _hpsearch_not_found_error for name in ("choice","uniform","loguniform")})

    try: 
        with open(filename) as f:
            cfg_code = f.read()
            
            # Convert "return model" to "__result=model"
            retmatch = re.search("^return", cfg_code, re.MULTILINE)
            if retmatch:
                cfg_code = cfg_code[:retmatch.start(0)] + "__result=" + cfg_code[retmatch.end(0):]

            # Execute the config file as a python script
            exec cfg_code in load_cfg_globals,load_cfg_locals

            # Extract the return value, either through a single __result or through specified 'args'
            if "__result" in load_cfg_locals:
                cfg_inst = load_cfg_locals["__result"]  # If there was a return statement, return that object
            elif len(args) > 0:
                cfg_inst = (load_cfg_locals[arg] for arg in args)
            else:
                cfg_inst = load_cfg_locals                  # Otherwise, return the complete dictionary of locals that were created
            return convert_cfg_to_instance(cfg_inst, load_cfg_globals)

    except Exception as err:
        print "ERROR while parsing config file \"%s\"." % filename
        raise


class deferred_load(object):
    """
    A deferred_load object stores a filename and, at some future point, 
    will load the object from the file. If the file defines multiple objects,
    e.g. a "model" and a "trainer", then use objectname to specify which object
    should be constructed.
    """
    def __init__(self,filename,objectname=None):
        self.filename = filename
        self.objectname = objectname

    def create(self):
        return load(self.filename)[self.objectname]

