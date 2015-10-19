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
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

_logo_fonts = { "Arial"     : ImageFont.truetype(os.path.abspath(os.path.dirname(__file__))+"/arial.ttf", 200),
                "ArialBold" : ImageFont.truetype(os.path.abspath(os.path.dirname(__file__))+"/arialbd.ttf", 200),
                "Courier"   : ImageFont.truetype(os.path.abspath(os.path.dirname(__file__))+"/cour.ttf", 200) }
_lettercache_unsized = {}
_lettercache = {}

def _autocrop(I):
    I = np.array(I)
    
    # Cut out all the rows/columns that are all white
    I = I[np.where(np.any(I[:,:,:3].min(axis=1)!=255,axis=1))[0],:,:] # Crop vertical
    I = I[:,np.where(np.any(I[:,:,:3].min(axis=0)!=255,axis=1))[0],:] # Crop horizontal

    # Add white border. Helps avoid edge artifacts when resizing down with anti-aliasing
    pad1 = 255*np.ones_like(I[:1,:,:]); pad1[:,:,3] = 0
    I = np.vstack([pad1, I, pad1])
    pad2 = 255*np.ones_like(I[:,:3,:]); pad2[:,:,3] = 0
    I = np.hstack([pad2, I, pad2])

    return Image.fromarray(I)

uparrow_chr = u'\u25B2'

def _get_letterimg_unsized(letter, font):
    global _lettercache_unsized
    global _logo_fonts

    colors = { "A" : (0,200,0),
               "C" : (0,0,200),
               "G" : (235,140,0),
               "T" : (200,0,0),
               "U" : (200,0,0),
               "N" : (128,128,128),
               uparrow_chr : (128,128,128) }

    assert letter in colors, "Unrecognized letter"
    assert font in _logo_fonts, "Unrecognized font"
    
    if (letter,font) not in _lettercache_unsized:
        # Draw full-sized versions of this letter
        letterimg = 255*np.ones((256,256,4), np.uint8)
        letterimg[:,:,3] = 0 # Transparent by default
        letterimg = Image.fromarray(letterimg)
        draw = ImageDraw.Draw(letterimg)
        draw.text((1,1), letter, colors[letter], font=_logo_fonts[font])
        letterimg = _autocrop(letterimg)
        _lettercache_unsized[(letter,font)] = letterimg
    
    return _lettercache_unsized[(letter,font)]


def get_letterimg(letter, width, height, font="ArialBold"):
    global _lettercache
    assert width and height

    # If we've never been asked for a letter of this width/zheight before, 
    # then we use Image.resize to generate a new one.
    if (letter,width,height,font) not in _lettercache:
        letterimg = _get_letterimg_unsized(letter, font)
        letterimg = letterimg.resize((width, height), Image.ANTIALIAS)
        _lettercache[(letter,width,height,font)] = np.array(letterimg).reshape((height,width,4))

    return _lettercache[(letter,width,height,font)]


def tape2logo(tape, height=51, letterwidth=6, bufferzoom=4, refseq=None, vmax=None, style=None, rna=False, transparent=False, complement=False):
    # Styles "stack" "grow" "growclip" "growfade" "bars" "bar"

    tapedim,tapelen = tape.shape # n = number of filters
    if tapedim != 4:
        raise NotImplementedError("Expected tape with 4 rows")

    if vmax is not None:
        tape = np.maximum(-vmax, np.minimum(vmax, tape))

    zheight = height*bufferzoom
    zletterwidth = letterwidth*bufferzoom
    mid1 = (zheight-bufferzoom)//2
    mid2 = (zheight-bufferzoom)//2 + bufferzoom


    if refseq:
        assert len(refseq) == tapelen
        refseq_height = int(letterwidth*bufferzoom*1.1)

        # Create an up-arrow image
        arrowheight = int(refseq_height*0.15)
        uparrow_img = get_letterimg(uparrow_chr, zletterwidth//2, arrowheight, font="Arial")
        pad1 = 255*np.ones((arrowheight, zletterwidth//4, 4))
        pad1[:,:,3] = 0
        uparrow_img = np.hstack([pad1, uparrow_img])
        pad2 = 255*np.ones((arrowheight, zletterwidth-uparrow_img.shape[1], 4))
        pad2[:,:,3] = 0
        uparrow_img = np.hstack([uparrow_img, pad2])

        mid1 -= refseq_height//2+2*bufferzoom
        mid2  = mid1+refseq_height+4*bufferzoom

    positive_only = bool(np.all(tape.ravel() >= 0)) or (style in ("grow", "growfade","bar"))
    if positive_only:
        mid1 = zheight
        mid2 = zheight

    translate = { "A":"A", "C":"C", "G":"G", "T":"T", "U":"U", "N":"N" } 
    if complement:
        translate = { "A":"T", "C":"G", "G":"C", "T":"A", "U":"A", "N":"N" } 


    lettertable = ["A","C","G","U" if rna else "T"]
    barcolors = { "A" : (128,220,128),
                  "C" : (128,128,220),
                  "G" : (245,200,90),
                  "T" : (220,128,128),
                  "U" : (220,128,128),
                  "N" : (192,192,192) }

    def make_lettercol(t, colheight, reverse):
        # Only show letters with positive coefficient in f
        idx = [i for i in range(4) if t[i] > 0]

        # Put largest positive value first in "above", and put largest negative value last in "below"
        idx = sorted(idx, key=lambda i: t[i])

        # Calculate the individual zheight of each letter in pixels
        zheights = [int(round(t[i]/sum(t[idx])*colheight)) for i in idx]
        idx      = [i for i,h in zip(idx,zheights) if h > 0]
        zheights = [h for h in zheights if h > 0]

        # While the stack of letters is too tall, remove pixel rows from the smallest-zheight entries
        #print sum(zheights) - mid1
        while sum(zheights) > mid1:
            zheights[-1] -= 1
            if zheights[-1] == 0:
                zheights.pop()
                idx.pop()

        # Make the individual images, reversing their order if so requested
        imgs = [get_letterimg(lettertable[i],  zletterwidth, h) for i,h in zip(idx, zheights)]
        if reverse:
            imgs = [img for img in reversed(imgs)]

        return np.vstack(imgs) if imgs else np.empty((0, zletterwidth, 4))
    
    if style == "seqlogo":
        assert positive_only
        L = 255*np.ones((zheight,tapelen*zletterwidth,4), np.uint8)
        L[:,:,3] = 0 # Transparent
        for j in range(tapelen):
            bits = 2 + np.sum(tape[:,j] * np.log2(tape[:,j]))
            letterimg = make_lettercol( tape[:,j], mid1 * bits/2., reverse=True)
            L[mid1-letterimg.shape[0]:mid1,j*zletterwidth:(j+1)*zletterwidth,:] = letterimg

        # Rescale it down to the original requested size
        L = np.array(Image.fromarray(L).resize((tapelen*letterwidth, height), Image.ANTIALIAS))
        if not transparent:
            L[:,:,3] = 255  # full opacity
        return L

    pos_tape = np.maximum(1e-16, tape)
    neg_tape = np.maximum(1e-16,-tape)

    pos_colheights = pos_tape.max(axis=0)
    neg_colheights = neg_tape.max(axis=0)

    #max_colheight  = np.maximum(pos_colheights, neg_colheights).max()
    #max_colheight  = (pos_colheights + neg_colheights).max()
    max_colheight  = neg_colheights.max()
    #neg_colheights = np.minimum(max_colheight,neg_colheights)


    pos_colheights /= max_colheight
    neg_colheights /= max_colheight

    
    # If we've been told to scale everything relative to a certain maximum, then adjust our scales accordinly
    if vmax:
        pos_colheights *= pos_tape.max() / vmax
        neg_colheights *= neg_tape.max() / vmax

    L = 255*np.ones((zheight,tapelen*zletterwidth,4), np.uint8)
    L[:,:,3] = 0  # Start transparent

    # For each column of the filter, generate a stack of letters for the logo
    for j in range(tapelen):

        if style in (None,"stack"):
            # Generate the stack of letters that goes above, and below, the dividing ling
            aboveimg = make_lettercol( tape[:,j], mid1 * pos_colheights[j], reverse=True)
            belowimg = make_lettercol(-tape[:,j], mid1 * neg_colheights[j], reverse=False) if not positive_only else None

            # Insert the stacked images into column j of the logo image
            L[mid1-aboveimg.shape[0]:mid1,j*zletterwidth:(j+1)*zletterwidth,:] = aboveimg
            if not positive_only:
                L[mid2:mid2+belowimg.shape[0],j*zletterwidth:(j+1)*zletterwidth,:] = belowimg  

            if refseq:
                letterimg = get_letterimg(refseq[j], zletterwidth, refseq_height, font="ArialBold")
                L[mid1+2*bufferzoom:mid2-2*bufferzoom,j*zletterwidth:(j+1)*zletterwidth,:] = letterimg

        elif style == "growclip":
            # Grow the height of each letter based on binding
            zletterheight = int(mid1 * neg_colheights[j])
            if zletterheight:
                letterimg = get_letterimg(refseq[j] if refseq else "N", zletterwidth, zletterheight, font="ArialBold")
                L[mid1-letterimg.shape[0]:mid1,j*zletterwidth:(j+1)*zletterwidth,:] = letterimg

        elif style == "refseq":
            letterimg = get_letterimg(refseq[j], zletterwidth, refseq_height, font="Arial")
            L[mid1-letterimg.shape[0]:mid1,j*zletterwidth:(j+1)*zletterwidth,:] = letterimg

        elif style == "growfade" or style == "grow":
            # Grow the height of each letter based on binding
            arrowpad_top = 3*bufferzoom
            arrowpad_btm = 4*bufferzoom
            arrowheight_padded = 0#arrowheight+arrowpad_top+arrowpad_btm
            growheight = int((mid1-arrowheight_padded-refseq_height) * neg_colheights[j])
            fademin = refseq_height
            fademax = refseq_height+0.333*(mid1-arrowheight_padded-refseq_height)
            zletterheight = refseq_height + growheight
            fade    = max(0, min(0.85, (fademax-zletterheight)/(fademax-fademin)))
            letterimg = get_letterimg(translate[refseq[j]] if refseq else "N", zletterwidth, zletterheight, font="ArialBold")
            if style == "growfade":
                letterimg = letterimg*(1-fade) + 255*fade
            mid0 = mid1-letterimg.shape[0]
            L[mid0:mid1,j*zletterwidth:(j+1)*zletterwidth,:] = letterimg[::-1,::] if complement else letterimg

            """
            #aboveimg = make_lettercol(tape[:,j], (mid1-bufferzoom*2) * pos_colheights[j], reverse=True)
            #intensity  = max(0, min(1.0, (pos_colheights[j]-0.4*refseq_height/mid1)/(1.5*refseq_height/mid1)))
            #aboveimg = aboveimg*intensity + 255*(1-intensity)
            tapej = tape[:,j].copy()
            tapej[tapej < 0.10*abs(tape).max()] = 0.0
            #if pos_colheights[j] >= 0.15*max(pos_colheights.max(),neg_colheights[j].max()):
            if np.any(tapej > 0):
                aboveimg = make_lettercol(tapej, (mid1-bufferzoom*3) * pos_colheights[j], reverse=True)
                aboveimg = np.minimum(255,aboveimg*0.61 + 255*0.4)
                assert mid0-arrowheight-arrowpad_btm >= 0
                assert mid0-arrowheight_padded-aboveimg.shape[0] >= 0
                L[mid0-arrowheight-arrowpad_btm:mid0-arrowpad_btm,j*zletterwidth:(j+1)*zletterwidth,:] = uparrow_img
                L[mid0-arrowheight_padded-aboveimg.shape[0]:mid0-arrowheight_padded,j*zletterwidth:(j+1)*zletterwidth,:] = aboveimg

                #grey = aboveimg.mean(axis=2).reshape(aboveimg.shape[:2]+(1,))
                #aboveimg[:,:,:] = np.minimum(255,grey.astype(np.float32)*160./grey.min())
                #L[mid0-arrowpad_btm-aboveimg.shape[0]:mid0-arrowpad_btm,j*zletterwidth:(j+1)*zletterwidth,:] = aboveimg
                """

        elif style == "bar":
            assert refseq, "style topbar needs refseq"
            # Put the refseq letter, with fixed height
            letterimg = get_letterimg(refseq[j], zletterwidth, refseq_height, font="Arial")
            L[mid1-letterimg.shape[0]:mid1,j*zletterwidth:(j+1)*zletterwidth,:] = letterimg

            # Draw a bar plot along the top based on neg_colheights
            barheight = int((mid1-refseq_height-2*bufferzoom) * neg_colheights[j])
            L[mid1-letterimg.shape[0]-barheight-2*bufferzoom:mid1-letterimg.shape[0]-2*bufferzoom,j*zletterwidth:(j+1)*zletterwidth,:] = np.array(barcolors[refseq[j]]).reshape((1,1,4))

        elif style == "bars":
            assert refseq, "style topbar needs refseq"
            # Put the refseq letter, with fixed height
            letterimg = get_letterimg(refseq[j], zletterwidth, refseq_height, font="Arial")
            L[mid1+2*bufferzoom:mid2-2*bufferzoom,j*zletterwidth:(j+1)*zletterwidth,:] = letterimg

            # Draw a bar plot along the top based on neg_colheights
            aboveheight = int(mid1 * neg_colheights[j])
            belowheight = int(mid1 * pos_colheights[j])
            L[mid1-aboveheight:mid1,j*zletterwidth:(j+1)*zletterwidth,:] = np.array(barcolors[refseq[j]]).reshape((1,1,4))
            L[mid2:mid2+belowheight,j*zletterwidth:(j+1)*zletterwidth,:] = np.array(barcolors[refseq[j]]).reshape((1,1,4))

        else:
            raise NotImplementedError("Unrecognzied style type")

    if style in (None, "stack") and not refseq:
        # Put a horizontal line across the middle of this logo
        L[mid1:mid1+bufferzoom,:,:] = 100
        if not positive_only:
            L[mid2-bufferzoom:mid2,:,:] = 100

    if not transparent:
        L[:,:,3] = 255  # full opacity

    # Rescale it down to the original requested size
    L = np.array(Image.fromarray(L).resize((tapelen*letterwidth, height), Image.ANTIALIAS))

    if complement:
        L = L[::-1,:,:] # vertical flip

    return L
