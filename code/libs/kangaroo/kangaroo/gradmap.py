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
# gradmap.py
#    
import os
import os.path
import sys
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from . import util
import deepity
import deepity.tape2logo

import matplotlib
matplotlib.rcParams.update({'font.size': 9, 
                            'font.family': 'sans serif', 
                            'text.usetex' : False})
if (not os.environ.has_key("DISPLAY")) and (not os.environ.has_key("HOMEDRIVE")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import colors as colors
from matplotlib import cm
from matplotlib.figure import Figure

gridimg_pal_RedBlue = matplotlib.colors.LinearSegmentedColormap('RedBlue', {
         'red':   ((0.00, 0.0, 0.0),
                   (0.50, 1.0, 1.0),
                   (1.00, 1.0, 1.0)),

         'green': ((0.00, 0.0, 0.0),
                   (0.50, 1.0, 1.0),
                   (1.00, 0.0, 0.0)),

         'blue':  ((0.00, 1.0, 1.0),
                   (0.50, 1.0, 1.0),
                   (1.00, 0.0, 0.0)),

        },gamma =1.0)(np.arange(256))

_redbluepal = np.asarray(gridimg_pal_RedBlue[:,:3]*255, dtype=np.uint8)

gridimg_pal_Black = matplotlib.colors.LinearSegmentedColormap('Black', {
         'red':   ((0.00, 1.0, 1.0),
                   (0.50, 1.0, 1.0),
                   (1.00, 0.0, 0.0)),

         'green': ((0.00, 1.0, 1.0),
                   (0.50, 1.0, 1.0),
                   (1.00, 0.0, 0.0)),

         'blue':  ((0.00, 1.0, 1.0),
                   (0.50, 1.0, 1.0),
                   (1.00, 0.0, 0.0)),

        },gamma =1.0)(np.arange(256))

_blackpal = np.asarray(gridimg_pal_Black[:,:3]*255, dtype=np.uint8)

gridimg_pal_Gray = matplotlib.colors.LinearSegmentedColormap('Gray', {
         'red':   ((0.00, 0.0, 0.0),
                   (0.50, 0.0, 0.0),
                   (1.00, 1.0, 1.0)),

         'green': ((0.00, 1.0, 1.0),
                   (0.50, 0.0, 0.0),
                   (1.00, 1.0, 1.0)),

         'blue':  ((0.00, 1.0, 1.0),
                   (0.50, 0.0, 0.0),
                   (1.00, 1.0, 1.0)),

        },gamma =1.0)(np.arange(256))

_graypal = np.asarray(gridimg_pal_Gray[:,:3]*255, dtype=np.uint8)

_fixedwidth_font = ImageFont.truetype(os.path.abspath(os.path.dirname(__file__))+"/cour.ttf", 10)
#_fixedwidth_font = ImageFont.load_default()

import scipy
import scipy.misc

def _zoomimg(I, zoom, smooth=False):
    if smooth:
        img = Image.fromarray(I) if isinstance(I,np.ndarray) else I
        img = img.resize((int(img.size[0]*zoom),int(img.size[1]*zoom)), Image.ANTIALIAS)
        I   = np.array(img) if isinstance(I,np.ndarray) else img
    else:
        # Zoom 3x3
        if isinstance(I,np.ndarray):
            I = np.repeat(I,zoom,0)
            I = np.repeat(I,zoom,1)
        else:
            I = I.resize((int(I.size[0]*zoom),int(I.size[1]*zoom)), Image.NEAREST)
    return I

def _gray2rgb(I, pal=None):
    if pal is None:
        pal = _redbluepal
    # Convert to colour
    return np.array(scipy.misc.toimage(I, pal=pal).convert("RGBA"))


def _array2img(A, vmax):
    if A.ndim==1:
        A = A.reshape((-1,1))
    I = A.T.copy()
    I -= -vmax
    I /= 2*vmax
    I = np.maximum(0,I)
    I = np.minimum(1,I)
    I *= 255
    I = np.uint8(I)

    return I

def _save_seqimg(filename, X, dX, vmax, zoom=1, style="grow", pal=None, complement=False, trim=(0,0)):
    trim = slice(trim[0],len(X)-trim[1])
    X = X[trim]
    dX = dX[trim]

    if style != "tape":
        # Use the tape2logo script
        dXlogo = deepity.tape2logo.tape2logo(dX.T, height=45*zoom, letterwidth=6*zoom, vmax=vmax, refseq=X, style=style, complement=complement)
        scipy.misc.imsave(filename, dXlogo)
        return

    # Otherwise, create a ticker-tape representation
    I = _array2img(dX, vmax)
    I = _zoomimg(I, 3)
    I = _zoomimg(I, 4,   True)
    I = _zoomimg(I, 0.5, True)
    I = _gray2rgb(I, pal=pal)

    if True:
        # Publish-quality version
        Ipub = _array2img(dX, vmax)
        Ipub = _zoomimg(Ipub, 5)
        Ipub = _zoomimg(Ipub, 6,   True)
        Ipub = _zoomimg(Ipub, 0.5, True)
        Ipub = _gray2rgb(Ipub, pal=pal)
        dXpub = deepity.tape2logo.tape2logo(dX.T, height=100, letterwidth=15, vmax=vmax, refseq=X, style="grow", complement=complement)

        # Add 1 pixel white border so that interpolation goes to white at the edges
        hb  = np.zeros_like(Ipub[:,:2,:])+210
        hb[:,:,3] = 255
        Ipub= np.hstack([hb, Ipub, hb])
        vb  = np.zeros_like(Ipub[:2,:,:])+210
        vb[:,:,3] = 255
        Ipub= np.vstack([vb, Ipub, vb])
        if complement:
            Ipub = Ipub[::-1,:,:]
        hb  = np.zeros_like(dXpub[:,:2,:])+255
        dXpub= np.hstack([hb, dXpub, hb])
        vb  = np.zeros_like(dXpub[:2,:,:])+255
        dXpub= np.vstack([vb, dXpub, vb])

        pub_img = np.vstack([Ipub, vb, dXpub] if complement else [dXpub, vb, Ipub])
        hb  = np.zeros_like(pub_img[:,:1,:])+255
        pub_img= np.hstack([hb, pub_img, hb])
        vb  = np.zeros_like(pub_img[:1,:,:])+255
        pub_img= np.vstack([pub_img, vb])

        scipy.misc.imsave(os.path.splitext(filename)[0]+"_pub.png", pub_img)

    Ivec = _array2img((abs(dX).max(axis=1)-abs(dX).min(axis=1)).reshape((-1,1)), vmax)
    Ivec = _zoomimg(Ivec, 3)
    Ivec = _zoomimg(Ivec, 4,   True)
    Ivec = _zoomimg(Ivec, 0.5, True)
    Ivec = _gray2rgb(Ivec, pal=_blackpal)

    # Add 1 pixel border
    hb = np.zeros_like(I[:,:1,:])+192
    I  = np.hstack([hb, I, hb])
    vb = np.zeros_like(I[:1,:,:])+192
    I  = np.vstack([vb, I, vb])

    hb = np.zeros_like(Ivec[:,:1,:])
    Ivec = np.hstack([hb+255, Ivec, hb+255])
    #Ivec = Ivec[:4]

    colors = { 'A' : (  0,205,  0),
               'C' : (  0, 30,205),
               'G' : (245,175,  0),
               'T' : (205,  0,  0),
               'N' : (128,128,128) }
    for synonym, base in [("a","A"),("c","C"),("g","G"),("t","T"),("u","T"),("U","T"),(".","N")]:
        colors[synonym] = colors[base]

    I = np.vstack([Ivec,
                   255*np.ones((1,I.shape[1],4),np.uint8),
                   255*np.ones((12,I.shape[1],4),np.uint8), 
                   I])
    I = Image.fromarray(I)  # convert to Image so that we can use ImageDraw
    draw = ImageDraw.Draw(I)
    for j in range(len(X)):
        draw.text((j*6+1,Ivec.shape[0]+1),X[j],colors[X[j]],font=_fixedwidth_font)
    I = np.array(I)
    scipy.misc.imsave(filename, I)

def _save_vecimg(filename, X, dX, vmax):
    # Convert to colour
    X  = _array2img(X,  vmax)
    dX = _array2img(dX, vmax)

    X = _zoomimg(X, 3)
    X = _zoomimg(X, 4,   True)
    X = _zoomimg(X, 0.5, True)
    dX = _zoomimg(dX, 3)
    dX = _zoomimg(dX, 4,   True)
    dX = _zoomimg(dX, 0.5, True)
    X  = _gray2rgb(X)
    dX = _gray2rgb(dX)

    # Add 1 pixel border
    hb = np.zeros_like(X[:,:1,:])
    X  = np.hstack([hb,  X, hb])
    dX = np.hstack([hb, dX, hb])
    vb = np.ones_like(X[:1,:,:])
    I = np.vstack([vb*0,
                   X,
                   vb*0,
                   vb*255,
                   vb*255,
                   vb*255,
                   vb*0,
                   dX,
                   vb*0])

    scipy.misc.imsave(filename, I)

def save_gradientmaps(data, predictions, outdir, maxrow=50, zoom=1, want_indices=False, apply_sigmoid=False, want_sort=True, trim=(0,0)):

    if not want_indices:
        # Create main index of all targets
        util.makepath(os.path.join(outdir))
        index_html = open(os.path.join(outdir, "index.html"), "w")
        index_html.write("<html><head><title>Gradient maps</title></head><body>\n")
        index_html.write("<table cellspacing=0 cellpadding=5 border=1>\n")
        index_html.write("<tr><th>Name</th></tr>\n")

    # For each target that has a ".gmaps" entry in predictions, generate a report and 
    # add it to the index_html
    indices = {}
    for targetname in predictions:
        if targetname.endswith(".gmaps"):
            continue

        targetdata = data[targetname]

        rowidx = targetdata.rowidx
        Z = predictions[targetname]
        print targetname, Z

        if targetdata.targetnames:
            Y = targetdata.Y
        else:
            Y = np.zeros_like(Z)
            Y[:] = np.nan

        nrow = len(Z)
        maxrow = min(maxrow,nrow)

        # Only use the first one
        Y = Y[:,0] 
        Z = Z[:,0]

        if want_sort:
            roworder = np.argsort(-Z.ravel(), kind="mergesort")
        else:
            roworder = np.arange(len(Z))
        
        if maxrow < nrow:
            #roworder = roworder[:maxrow]
            Ysum = np.nansum(Y)
            if np.isnan(Ysum) or Ysum < 1:
                Ysum = len(Z)
            nrow_pos = max(nrow//3,min(nrow,int())) 
            roworder = [roworder[i] for i in range(0,nrow_pos,max(1,nrow_pos//maxrow))]

        if want_indices:
            indices[targetname] = roworder
            continue

        gmaps = predictions[targetname+".gmaps"]
        inputnames = sorted(gmaps.keys())

        # Add this target to the index_html
        index_html.write("<tr><td><a href=\"%s/index.html\">%s</a></td></tr>\n" % (targetname, targetname))


        # Determine the min/max gradmap value range, across all inputs
        vmax = -np.inf
        vmin =  np.inf
        for inputname in inputnames:
            gmap_values = gmaps[inputname]
            for row in roworder:
                X,dX = gmap_values[row]
                # First, for any sequences, subtract the channel mean to account for the
                # fact that the sum of channel inputs must equal one
                if isinstance(X,str):
                    #dX -= dX.mean(axis=1).reshape((-1,1)) # sense map
                    idx = util.acgt2ord(X).ravel()
                    
                    for i,base in enumerate(idx):
                        if base in range(0,4):
                            dX[i,:] -= dX[i,base]
                        else:
                            dX[i,:] -= dX[i,:].mean()
                            
                    # Now, take note of the absmax value
                    vmax = max(vmax, dX.max())
                    vmin = min(vmin, dX.min())
        #vmin,vmax = 0.0, 1.0  # mutation map

        # Create HTML report for this specific target
        util.makepath(os.path.join(outdir, targetname))
        target_html = open(os.path.join(outdir, targetname, "index.html"), "w")
        target_html.write("<html><head><title>Gradient maps - %s</title>\n" % targetname)
        target_html.write("""
        <script language="javascript">
        function show_logo()
        {
            var smaps = document.getElementsByClassName('sensitivitymap')
            for (var i = 0; i < smaps.length; i++)
                if (smaps[i].src.search("_tape") != -1)
                    smaps[i].src = smaps[i].src.replace("_tape","_logo").replace("_pfm","_logo");
        }
        function show_tape()
        {
            var smaps = document.getElementsByClassName('sensitivitymap')
            for (var i = 0; i < smaps.length; i++)
                if (smaps[i].src.search("_logo") != -1)
                    smaps[i].src = smaps[i].src.replace("_logo","_tape").replace("_pfm","_tape");
        }
        function show_pfm()
        {
            var smaps = document.getElementsByClassName('sensitivitymap')
            for (var i = 0; i < smaps.length; i++)
                if (smaps[i].src.search("_pfm") != -1)
                    smaps[i].src = smaps[i].src.replace("_logo","_pfm").replace("_tape","_pfm");
        }
        function onkeypress() {
            var smaps = document.getElementsByClassName('sensitivitymap');
            for (var i = 0; i < smaps.length; i++) {
                if (smaps[i].src.search("_tape") != -1) {
                    show_logo();
                    break;
                }
                if (smaps[i].src.search("_logo") != -1) {
                    //show_pfm();
                    show_tape();
                    break;
                }
                //if (smaps[i].src.search("_pfm") != -1) {
                //    show_tape();
                //    break;
                //}
            }
        }
        document['onkeypress'] = onkeypress
        </script></head><body>
        """)
        target_html.write("<h2>%s</h2><hr/>\n"%targetname)
        target_html.write("<input type=\"button\" value=\"logo\" onclick=\"show_logo();\"/>")
        target_html.write("<input type=\"button\" value=\"tape\" onclick=\"show_tape();\"/>")
        target_html.write("range = [%.3f, %.3f]<br/>\n"%(vmin,vmax))
        target_html.write("<table cellspacing=0 cellpadding=5 border=1  style=\"font-size:7pt\">\n")
        target_html.write("<tr><th>Row#</th><th>Y</th><th>Z</th>%s</tr>\n" % "".join(["<th align=left>%s<br/>%s</th>"%(name[1:],name) for name in inputnames]))

        np.savez(os.path.join(outdir, targetname, "X.npz"), 
                 X=np.array([{"row" : rowidx[row]+1, "X" : gmaps["dX_seq"][row][0], "dX" : gmaps["dX_seq"][row][1], "Y" : Y[row], "Z" : Z[row] } for i,row in enumerate(roworder)], dtype=object))


        # For each row in the data/predictions, write out a sequence and its corresponding gradient
        for i,row in enumerate(roworder):
            # For each row, loop over the inputnames
            target_html.write("<tr><td>%d</td><td>%.4f</td><td>%.4f</td>"%(rowidx[row]+1, Y[row], Z[row]))
            for inputname in inputnames:
                X,dX = gmaps[inputname][row]
                if isinstance(X,str):
                    # Format X like a sequence, in two styles
                    dXfilename_logo = "%06d_%s_logo.png" % (i, inputname)
                    dXfilename_tape = "%06d_%s_tape.png" % (i, inputname)
                    dXfilename = dXfilename_tape
                    dXsig = 1./(1+np.exp(-dX)) if apply_sigmoid else dX
                    complement = False
                    vrange = max(-vmin,vmax)
                    _save_seqimg(os.path.join(outdir, targetname, dXfilename_logo), X, dXsig, vrange, zoom=zoom, style="grow", complement=complement, trim=trim)
                    _save_seqimg(os.path.join(outdir, targetname, dXfilename_tape), X, dX, vrange, zoom=zoom, style="tape", complement=complement, trim=trim) # sensemap
                else:
                    # Format X like a vector
                    dXfilename = "%06d_%s.png" % (i, inputname)
                    _save_vecimg(os.path.join(outdir, targetname, dXfilename), X, dX, vmax)

                target_html.write("<td><img src=\"%s\" class=\"sensitivitymap\"/></td>" % (dXfilename))

            target_html.write("</tr>\n")

        target_html.close()

    index_html.close()
