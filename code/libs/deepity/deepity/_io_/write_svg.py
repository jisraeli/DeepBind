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
from . import pydot

##########################################################################

def write_svg(filename,root_node):
    nodes = []
    edges = []
    _collect_dot_graph(root_node,nodes,edges)
    g = pydot.Dot(splines="ortho")
    g_nodes = [pydot.Node(name=node.__class__.__name__,
                          shape="box") 
               for node in nodes]
    for n in g_nodes:
        g.add_node(n)
    for tail,head in edges:
        g_tail = g_nodes[nodes.index(tail.node)]
        g_head = g_nodes[nodes.index(head.node)]
        g.add_edge(pydot.Edge(g_tail,g_head,
                              taillabel=tail.name,
                              headlabel=head.name,
                              sametail=tail.name,
                              samehead=head.name,
                              fontname="courier",
                              fontsize=10,
                              arrowsize=0.4,
                              dir="both",
                              arrowtail="box",
                              arrowhead="obox"))
    g.write_svg(filename)


def _collect_dot_graph(node,nodes,edges):
    if node in nodes:
        return
    nodes.append(node)
    for head in node.iplugs:
        for tail in head.srcs:
            _collect_dot_graph(src.node,nodes,edges)
    for tail in node.oplugs:
        for head in tail.dsts:
            edge = (tail,head)
            if edge not in edges:
                edges.append(edge)
            _collect_dot_graph(dst.node,nodes,edges)
