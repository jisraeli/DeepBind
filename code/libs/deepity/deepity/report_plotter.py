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
# DO NOT IMPORT THIS MODULE
# It is meant to be spawned by report.py as a separate process,
# in order to avoid problems using matplotlib from within child
# processes of the multiprocessing module 
# (crashing on exit; instability in subsequent child processes)
import os
import os.path
import sys
import copy
import glob
import re
import numpy as np
import argparse
import tape2logo
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import time

import matplotlib
matplotlib.rcParams.update({'font.size': 9, 
                            'font.family': 'sans serif', 
                            'text.usetex' : False,
                            'figure.max_num_figures' : 100})
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

gridimg_pal = np.asarray(gridimg_pal_RedBlue[:,:3]*255, dtype=np.uint8)


gridimg_pal_Red = matplotlib.colors.LinearSegmentedColormap('Red', {
         'red':   ((0.00, 1.0, 1.0),
                   (1.00, 1.0, 1.0)),

         'green': ((0.00, 1.0, 1.0),
                   (1.00, 0.0, 0.0)),

         'blue':  ((0.00, 1.0, 1.0),
                   (1.00, 0.0, 0.0)),

        },gamma =1.0)(np.arange(256))

gridimg_pal_positive = np.asarray(gridimg_pal_Red[:,:3]*255, dtype=np.uint8)


_image_grid_font = ImageFont.truetype(os.path.abspath(os.path.dirname(__file__))+"/arial.ttf", 9)
#_image_grid_font = ImageFont.load_default()

import scipy
import scipy.misc
import scipy.stats

import warnings
#warnings.simplefilter('error', UserWarning)



def calc_auc(z,y, want_curve = False):
    assert len(z) == len(y)
    m = ~np.isnan(y)
    y = y[m]
    z = z[m]

    ymin = y.min()
    ymax = y.max()
    lo = (0.02-0) / (1.0 - 0.0) * (ymax-ymin) + ymin
    hi = (0.10-0) / (1.0 - 0.0) * (ymax-ymin) + ymin

    mlo = y<=lo
    mhi = y>=hi
    y[mlo] = 0
    y[mhi] = 1
    y[np.logical_and(mlo,mhi)] = np.nan

    m = ~np.isnan(y)
    y = y[m]
    z = z[m]

    order = np.argsort(z,axis=0)[::-1].ravel()   # Sort by decreasing order of prediction strength
    z = z[order]
    y = y[order]
    npos = np.count_nonzero(y)      # Total number of positives.
    nneg = len(y)-npos              # Total number of negatives.
    if nneg == 0 or npos == 0:
        return (1,None) if want_curve else 1

    n = len(y)
    fprate = np.zeros((n+1,1))
    tprate = np.zeros((n+1,1))
    ntpos,nfpos = 0.,0.
    for i,yi in enumerate(y):
        if yi: ntpos += 1
        else:  nfpos += 1
        tprate[i+1] = ntpos/npos
        fprate[i+1] = nfpos/nneg
    auc = np.trapz(tprate,fprate,axis=0)
    if want_curve:
        curve = np.hstack([fprate,tprate])
        return auc,curve
    return auc

def makepath(dir):
    """
    Makes a complete path if it does not exist already. 
    Does not remove any existing files.
    Fixes periodic failure of os.makedirs on Windows.
    """
    if os.path.exists(dir):
        return dir
    retries = 8
    while retries >= 0:
        try:
            time.sleep(0.001)
            os.makedirs(dir)
            retries = -1
        except Exception, e:
            if retries == 0:
                raise
        else:
            retries -= 1
    return dir

#####################################################################

class filter_images(object):
    def __init__(self,F,per_image_scale=False,symmetric_range=True):
        assert len(F.shape) == 3 # (size_y,size_x,num_filters)

        # Get information about shape and range of filter elements
        self.vmean = F.reshape((F.shape[0],-1)).mean(axis=1).reshape((-1,1,1))
        self.vmin  = F.reshape((F.shape[0],-1)).min(axis=1).reshape((-1,1,1))
        self.vmax  = F.reshape((F.shape[0],-1)).max(axis=1).reshape((-1,1,1))

        if symmetric_range:
            self.vmax = np.maximum(abs(self.vmin),abs(self.vmax))
            self.vmin = -self.vmax

        self.vranges = [(F[i,:,:].min(), F[i,:,:].max()) for i in range(F.shape[0])]
            
        # Convert filters to images
        self.raw = F
        I = F.copy()
        if per_image_scale:
            I -= self.vmin
            I *= 1./(self.vmax-self.vmin)
        else:
            I -= self.vmin.min()
            I *= 1./(self.vmax.max()-self.vmin.min())
        I = np.maximum(0,I)
        I = np.minimum(1,I)
        I *= 255
        I = np.uint8(I)

        self.images = I

    def __len__(self):       return self.images.shape[0]
    def __getitem__(self,i): return self.images[i,:,:]
    def __iter__(self):      return [self.images[i,:,:] for i in range(len(self))].__iter__()

    def zoom(self, factor):
        self.images = np.repeat(self.images,factor,1)
        self.images = np.repeat(self.images,factor,2)

    def zoom_smooth(self, factor):
        I = self.images
        Z = np.zeros((I.shape[0],I.shape[1]*factor,I.shape[2]*factor), dtype=I.dtype)
        for i in range(len(I)):
            img = Image.fromarray(I[i])
            img = img.resize((int(img.size[0]*factor),int(img.size[1]*factor)), Image.ANTIALIAS)
            Z[i,:,:] = np.array(img)
        self.images = Z

def L2RGB(I,cmap):
    I = scipy.misc.toimage(I, pal=cmap).convert("RGB")
    return np.array(I)

def addborder(I, color=0):
    I = np.vstack([color*np.ones_like(I[:1,:,:]),
                   I,
                   color*np.ones_like(I[:1,:,:])])
    I = np.hstack([color*np.ones_like(I[:,:1,:]),
                   I,
                   color*np.ones_like(I[:,:1,:])])
    return I


def image_grid(fimgs, maxwd = 6000, maxht = 5000, cmap=None, fade=False, positive=False):
    I = fimgs.images
    max_vmax = fimgs.vmax.max()
    min_vmin = fimgs.vmin.min()
    if cmap is None:
        cmap = np.repeat(np.arange(256).reshape((-1,1)),3)

    n,fht,fwd = I.shape[:3] # n = number of features
    if fwd > maxwd or fht > maxht:
        print "image larger than maximum size allowed"
        return None

    is_logo = (len(I.shape) == 4)

    #if not is_logo:
    fwd += 2 # border
    fht += 2 # border

    idwd  = 20
    decwd = 12
    txtwd = 48
    wd = idwd+decwd+fwd+txtwd
    ht = fht

    maxrows = min(int(maxht-1)//(ht+1), n)     # How many images can stack on top of each other? (-1 / +3 because of 1 pixel border and 1 pixel cell padding)
    maxcols = min(int(maxwd-1)//(wd+1), (n+maxrows-1)//maxrows)

    # Start from a blank white image
    G = np.ones(((ht+1)*maxrows-1,(wd+1)*maxcols-1,3),dtype=I.dtype)
    if G.dtype == np.uint8:
        G *= 255

    # Copy each image into a spot in the grid
    for i in range(n):
        col = i // maxrows
        row = i % maxrows
        if col >= maxcols:
            break
        x0  = col*(wd+1)
        y0  = row*(ht+1)
        if is_logo:
            F = I[i,:,:,:].copy()
        else:
            F = L2RGB(I[i,:,:], cmap)
        F = addborder(F)

        rmin,rmax = fimgs.vranges[i]

        mid = ht//2
        hi = int((mid-1) * float(max(0, rmax)) / abs(max_vmax) + 0.5)
        lo = int((mid-1) * float(max(0,-rmin)) / abs(min_vmin) + 0.5) if not positive else 0

        #if positive:
        #    F[1:-1,1:-1] = 255-F[1:-1,1:-1]

        if fade and hi <= 1 and lo <= 1:
            F = 192 + F//4   # Fade out anything that's extremely weak, so that user's attention is drawn to the other filters.

        img = np.hstack([255*np.ones((ht,idwd,3),np.uint8),
                         F,
                         255*np.ones((ht,decwd+txtwd,3),np.uint8)])


        # Draw a little vertical bar thingy to visually indicate magnitude of range
        # relative to other filters
        img[mid-hi:mid,     idwd+fwd+3:idwd+fwd+decwd-3,:] = np.asarray([255,0,0]).reshape((1,1,3))
        img[mid+1:mid+1+lo, idwd+fwd+3:idwd+fwd+decwd-3,:] = np.asarray([0,0,255]).reshape((1,1,3))
        #img[mid,idwd+fwd+2:idwd+fwd+decwd-2,:] *= (1-max(float(rmax) / abs(max_vmax),float(-rmin) / abs(min_vmin)))**0.8

        img = Image.fromarray(img)  # convert to Image so that we can use ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((0,2),"%3d"%i,(200,200,200),font=_image_grid_font)
        if rmax != rmin and not positive:
            draw.text((idwd+decwd+F.shape[1]+0, 2),"%+.3f"%rmax,0,font=_image_grid_font)
            draw.text((idwd+decwd+F.shape[1]+2,13),"%+.3f"%rmin,0,font=_image_grid_font)
        else:
            draw.text((idwd+decwd+F.shape[1]+2, 2),"%+.3f"%rmax,0,font=_image_grid_font)
        img = np.array(img)  # convert back to numpy

        G[y0:y0+ht,x0:x0+wd,:] = img

    return G

def fimgs2logos(fimgs, finalheight, finalletterwidth):

    filters = fimgs.raw
    nfilter,_,fsize = filters.shape # n = number of filters

    logos = 255*np.ones((nfilter,finalheight,fsize*finalletterwidth,3), np.uint8)
    limgs = copy.deepcopy(fimgs)
    limgs.images = logos
    for k in range(nfilter):
        logos[k,:,:,:] = tape2logo.tape2logo(filters[k,:,:], finalheight, finalletterwidth, 5)

    return limgs

def reverse_complement_weights(W):
    W = W[:,:,::-1]
    W = W[:,[3,2,1,0],:]
    return W

########################################################

def unpack(npzfile, zoom):
    entries = np.load(npzfile)['entries']
    outdir = os.path.splitext(npzfile)[0]
    makepath(outdir)

    dpi = 80.0

    # First dump filter images
    if "images" in entries[-1]:
        for name,weights in entries[-1]["images"].iteritems():
            name = "filters_"+name.replace("[","(").replace("]",")")
            is_conv = (".conv_" in name and weights.shape[-1] > 1) # convnet filters treated differently.

            # First, generate 
            if is_conv:
                # Save colourized filters
                fimgs = filter_images(weights, per_image_scale=True)
                fimgs.zoom(max(1,12//fimgs.images.shape[1]))
                fimgs.zoom_smooth(4)
                fimgs.zoom_smooth(0.5)
                #fimgs.zoom(max(1,24//fimgs.images.shape[1]))
                scipy.misc.imsave(outdir+"/"+name+".color.png",image_grid(fimgs, cmap=gridimg_pal, fade=True))

                # Save black and white filters
                #fimgs = filter_images(weights, per_image_scale=True)
                #fimgs.zoom(max(1,24//fimgs.images.shape[1]))
                #scipy.misc.imsave(outdir+"/"+name+".png",image_grid(fimgs, fade=True))

                logoheight, letterwd = 41, 6

                weights -= (np.sort(weights, axis=1)[:,1:2,:] + np.sort(weights, axis=1)[:,2:3,:])/2
                #weights -= np.sort(weights, axis=1)[:,1:2,:]  # Subtract off the second-smallest value in each column
                #weights -= weights.min(axis=1).reshape((weights.shape[0],1,weights.shape[2])); logoheight = 24
                
                logoheight = int(round(zoom*logoheight))
                letterwd   = int(round(zoom*letterwd))

                fwdlogo = fimgs2logos(filter_images(weights, per_image_scale=True), logoheight, letterwd)
                revlogo = fimgs2logos(filter_images(reverse_complement_weights(weights), per_image_scale=True), logoheight, letterwd)
                fwdlogo = image_grid(fwdlogo, fade=True)
                revlogo = image_grid(revlogo, fade=True)
                fwdtitle = Image.fromarray(255*np.ones((20,fwdlogo.shape[1],3),np.uint8)); ImageDraw.Draw(fwdtitle).text((20,2),"actual",(0,0,0),font=_image_grid_font)
                revtitle = Image.fromarray(255*np.ones((20,revlogo.shape[1],3),np.uint8)); ImageDraw.Draw(revtitle).text((20,2),"reverse complement",(0,0,0),font=_image_grid_font)
                scipy.misc.imsave(outdir+"/"+name+".logo.png", np.hstack([np.vstack([np.array(fwdtitle), fwdlogo]), np.vstack([np.array(revtitle), revlogo])]))

            else:
                # Save colourized weight maps
                fimgs = filter_images(weights, per_image_scale=False)
                fimgs.zoom(max(1,12//fimgs.images.shape[1]))
                scipy.misc.imsave(outdir+"/"+name+".png",image_grid(fimgs, cmap=gridimg_pal))
                            
    # Then dump loss plots 
    if "tloss" in entries[-1]:
        fig = plt.figure(figsize=(300/dpi,200/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=False)
        ax = fig.add_axes([0.1,0.08,.88,.88])

        colours = [[0.0,0.0,0.0],  # train
                    [0.0,0.7,0.4],  # validate
                    [1.0,0.0,0.0]]  # test (if any)
        styles = ['-','-','-']
        
        ax.set_autoscale_on(False)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True,which='major',axis='y',linestyle=':',color=[0.2,0.2,0.2])
        ax.grid(True,which='minor',axis='y',linestyle=':',color=[0.8,0.8,0.8])

        xval   = np.asarray([entry["step"]  for entry in entries])
        yval_t = np.asarray([entry["tloss"] for entry in entries])
        yval_v = np.asarray([entry["vloss"] for entry in entries]) if "vloss" in entries[-1] else None
        xmax = xval.max()
        ymin = yval_t.min()
        ymax = yval_t.max()
        ax.semilogy(xval,yval_t,color=colours[0],linestyle=styles[0],label="tloss")
        if yval_v is not None:
            ymin = min(ymin,yval_v.min())
            ymax = min(ymax,yval_v.max())
            ax.semilogy(xval,yval_v,color=colours[1],linestyle=styles[1],label="vloss")

        ymin = max(ymin,1e-8)        # Just update the axis limits, since this isn't done 
        ymax = max(ymax,1e-8+ymin)   # automatically when you modify the line data directly.
        ymax = min(ymax,ymin*10.1)
        ax.set_xlim(0,xmax)
        ax.set_ylim(ymin,ymax)
        ax.legend()
        ax.set_xlim(0,xmax*1.1)
        ax.set_ylim([10**np.floor(np.log10(ymin)),10**np.ceil(np.log10(ymax))])
        fig.savefig(outdir+"/loss.png",dpi=dpi)
        #fig.savefig(outdir+"/loss.pdf",dpi=dpi)
        del fig
        plt.close()

                    
    # Then dump the prediction/target scatterplots
    for foldname in ("t","v"):
        if not foldname+"Z" in entries[-1]:
            continue

        xval = entries[-1][foldname+"Y"]
        yval = entries[-1][foldname+"Z"]

        for i in range(entries[-1]["tY"].shape[1]):
            xval_i = xval[:,i:i+1]
            yval_i = yval[:,i:i+1]

            fig = plt.figure(figsize=(300/dpi,300/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=False)
            ax = fig.add_axes([0.14,0.14,.74,.74])
            ybar = fig.add_axes([0.89, 0.14, 0.11, 0.74])

            keep = ~np.isnan(xval_i)
            xval_i = xval_i[keep]
            yval_i = yval_i[keep]
            lo = min(xval_i.min(),yval_i.min())
            hi = max(xval_i.max(),yval_i.max())

            aximg = ax.hist2d(xval_i,yval_i,bins=50,range=[[lo-.05,hi+.05],[lo-.05,hi+.05]],cmap=cm.hot,cmin=1)[3]
            ybar.hist(yval_i, bins=30, orientation='horizontal', histtype='bar', cumulative=False, color=[0.0,0.0,1.0], range=[lo-0.05, hi+0.05], edgecolor="none")
            ybar.set_ylim([lo-0.05, hi+0.05])
            ybar.axis("off")
                    
            ax.set_xlabel("target")
            ax.set_ylabel("prediction")
            fig.savefig(outdir+"/predict_%02d_%s.png"%(i,foldname),dpi=dpi)
            #fig.savefig(outdir+"/predict_%02d%s.pdf"%(i,foldname),dpi=dpi)
            del fig
            plt.close()

    # Finally, write AUCs
    for foldname in ("t","v"):
        if not foldname+"Z" in entries[-1]:
            continue

        x = entries[-1][foldname+"Y"]
        z = entries[-1][foldname+"Z"]

        for i in range(entries[-1]["tY"].shape[1]):

            y_i = x[:,i:i+1]
            z_i = z[:,i:i+1]
            auc,curve = calc_auc(z_i,y_i,want_curve=True)
            if curve is not None:
                xval = curve[:,0]
                yval = curve[:,1]
                    
                fig = plt.figure(figsize=(300/dpi,300/dpi),dpi=dpi,facecolor='w',edgecolor='b',frameon=False)
                ax = fig.add_axes([0.18,0.1,.8,.8])
                ax.plot(xval,yval,'-r')
                ax.set_xlabel("FP rate")
                ax.set_ylabel("TP rate")
                fig.savefig(outdir+"/predict_%02d_%s_auc.png"%(i,foldname),dpi=dpi)
                #fig.savefig(outdir+"/predict_%02d_%s_auc.pdf"%(i,foldname),dpi=dpi)
                del fig
                plt.close()

args = argparse.ArgumentParser(description="Unpack a \"foldX.report.npz\" file and generate visualizations of its contents (filters, predictions, etc)")
args.add_argument("path", type=str, help="A report.npz file, or a directory to search.")
args.add_argument("-z","--zoom", type=float, default=1, help="Zoom factor relative to default size; for generating print-quality bitmaps.")
args = args.parse_args()

if not os.path.exists(args.path):
    quit("Cannot find file \"%s\"."%args.path)

if os.path.isdir(args.path):
    for dirpath, dirnames, filenames in os.walk(args.path):
        for filename in filenames:
            if len(re.findall("fold.\.report\.npz", filename)) > 0:
                unpack(os.path.join(dirpath, filename), args.zoom)
else:
    unpack(args.path,args.zoom)


