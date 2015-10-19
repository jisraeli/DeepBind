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
import sys
import re
import os
import os.path
import argparse
import numpy as np
import numpy.random as npr
import gzip
import csv
import cPickle
import deepbind_util as util
import deepity
import scipy
import shutil

# Warning: The multiprocessing module may be used indirectly, so do not put any 
# unintentional statements outside of main().

def main():
    util.enable_reversecomplement()

    args   = loadargs()
    models = loadmodels(args)
    tfgroups = load_tfgroups(args)
    util.globals.flags.push("normalize_targets", True)

    for tfgroup in tfgroups:
        trdata = None
        if len(tfgroup["ids"]) == 0:
            print "No TFs to train on microarray %s"%tfgroup["train_fold"]
            continue

        if "calib" in args.steps:
            trdata = load_pbmdata(trdata, tfgroup["ids"], tfgroup["train_fold"], args, remove_probe_bias=True)
            util.calibrate(models, trdata, args.calibdir, nfold=args.nfold, ncalib=args.ncalib, allfolds=True)

        if "train" in args.steps:
            trdata = load_pbmdata(trdata, tfgroup["ids"], tfgroup["train_fold"], args, remove_probe_bias=True)
            util.train(models, trdata, args.calibdir, args.finaldir, nfold=1,          ntrial=args.ntrial, metric_key="pearson.r")

    for tfgroup in tfgroups:
        tedata = None

        newids = []
        for id in tfgroup["ids"]:
            if os.path.exists(args.outdir+"/final/"+id+"/model.pkl"):
                newids.append(id)
            else:
                print "WARNING: did not find model for %s, skipping" % id
        tfgroup["ids"] = newids

        if len(tfgroup["ids"]) == 0:
            print "No TFs to test on microarray %s"%tfgroup["train_fold"]
            continue

        if "test" in args.steps:
            tedata = load_pbmdata(tedata, tfgroup["ids"], tfgroup["test_fold"], args, remove_probe_bias=False)
            save_test_performance(tedata, tfgroup["ids"], tfgroup["test_fold"], args)

        if "report" in args.steps:
            tedata = load_pbmdata(tedata, tfgroup["ids"], tfgroup["test_fold"], args, remove_probe_bias=False)
            util.save_featuremaps(tedata, args.finaldir, args.reportdir)

    if "report" in args.steps:
        all_tfids = sum([tfgroup["ids"] for tfgroup in tfgroups],[])
        save_report(args.finaldir, args.reportdir, all_tfids, index_metric="pearson")
        save_pbm_performance_table(args, all_tfids)

    if "chip" in args.steps:
        all_tfids = sum([tfgroup["ids"] for tfgroup in tfgroups],[])
        save_chip_performance_table(args, all_tfids)

#########################################################################

def load_tfgroups(args):
    with open("../data/dream5/pbm/tfids.txt") as f:
        tfids = sorted([line.rstrip("\r\n") for line in f.readlines()])

    # Narrow down the list of ids based on which chunk we've been asked to compute
    tfids  = util.getchunktargets(args, tfids)

    # Group the TF ids into those that are trained on A versus those that are trained on B
    tfgroups = [ {"ids" : list(set(tfids).intersection(["C_%d"%(i+1) for i in range(20)] 
                                                      + ["TF_%d"%(i+1) for i in range( 0,33)])), "train_fold" : "A", "test_fold" : "B" },
                 {"ids" : list(set(tfids).intersection( ["TF_%d"%(i+1) for i in range(33,66)])), "train_fold" : "B", "test_fold" : "A" }]

    return tfgroups

#################################

def loadargs():
    args = argparse.ArgumentParser(description="Generate the DREAM5 PBM and/or CHIP experiments.")
    args = util.parseargs("dream5", args)
    return args

#################################

def loadmodels(args):
    models = util.loadmodels(args, "cfg/regression/maxpool")
    for cfg in models.itervalues():
        cfg["model"].conv_seq[0].fsize = 24  # Override default max motif length
    return models

#################################

def load_probe_biases():

    if not os.path.exists("../data/dream5/pbm/probe_biases.npz"):

        # First find which rows belong to which microarray (A or B)
        with gzip.open("../data/dream5/pbm/sequences.tsv.gz") as f:
            f.readline()
            foldids = [line[0] for line in f]
            rowidx  = { "A" : [i for i in range(len(foldids)) if foldids[i] == "A"], 
                        "B" : [i for i in range(len(foldids)) if foldids[i] == "B"] }

        # Then load the targets and split the rows into rows for A and rows for B
        with gzip.open("../data/dream5/pbm/targets.tsv.gz") as f:
            reader = csv.reader(f, delimiter='\t')
            colnames = reader.next()
            targets = np.array([[float(x) for x in row] for row in reader])

        biases = {}
        for foldid in ("A","B"):
            # For this microarray (A or B), calculate a multiplicative bias
            # for each one of its individual probes (rows).
            microarray_measurements = targets[rowidx[foldid]]
            bias = []
            for i in range(len(microarray_measurements)):
                probe_measurements = microarray_measurements[i,:].ravel()
                if np.all(np.isnan(probe_measurements)):
                    bias.append(np.nan)
                else:
                    bias.append(np.median(probe_measurements[~np.isnan(probe_measurements)]))
            biases[foldid] = np.array(bias)

        np.savez("../data/dream5/pbm/probe_biases.npz", **biases)

    with np.load("../data/dream5/pbm/probe_biases.npz") as f:
        biases = { key : f[key] for key in f.keys() }
    return biases


#################################

def load_pbmdata(trdata, tfids, fold, args, remove_probe_bias=False):
    if trdata is not None:
        return trdata  # Training data was already loaded in earlier step.
    maxrows = 10000 if args.quick else None

    # Determine which targetnames we're responsible for
    targetnames = gzip.open("../data/dream5/pbm/targets.tsv.gz").readline().rstrip().split("\t")
    targetcols = [i for i in range(len(targetnames)) if targetnames[i] in tfids]


    data = util.datasource.fromtxt("../data/dream5/pbm/sequences.tsv.gz", None, "../data/dream5/pbm/targets.tsv.gz", targetcols=targetcols, foldfilter=fold,  maxrows=maxrows)

    if remove_probe_bias:
        # Remove per-probe multiplicative bias
        biases = load_probe_biases()[fold].reshape((-1,1))
        data.targets /= biases
        data.Y /= biases

    return data

#################################

def save_test_performance(data, tfids, fold, args):

    print "Predicting...", tfids
    # Generate predictions for each model
    predictions = util.predict(data, args.finaldir, args.reportdir)
    print "done"

    for tfname in predictions:
        print tfname
        Z = predictions[tfname]


        # Restore the original (unpreprocessed) scale and normalization of the predictions
        # Then add the opposite microarray's estimated probe value to each probe
        with open(args.finaldir + "/" + tfname +"/preprocessors.pkl") as f:
            normalizer = cPickle.load(f)['targets'][0]
        Z *= normalizer.scales[0]
        Z += normalizer.biases[0]

        # Add per-probe multiplicative bias to predictions
        probe_biases = load_probe_biases()[fold].reshape((-1,1))
        if args.quick:
            probe_biases = probe_biases[:len(Z),:]
        M = ~np.isnan(probe_biases)
        Z[M] *= probe_biases[M]

        #with open("predictions/%s_%s.tsv" % (tfname, {"A":"B","B":"A"}[foldid]), 'w') as f:
        #    f.writelines([str(x)+"\n" for x in Z.ravel()])
        z = Z.ravel()
        y = data.Y[:,data.targetnames.index(tfname)].ravel()
        rowidx = data.rowidx.ravel()
        mask = ~np.isnan(y)
        zscore4 = np.std(y[mask])*4+np.mean(y[mask])
        util._update_metrics(args.finaldir, tfname, "test", rowidx, z, y, aucthresh=(zscore4,zscore4))

#################################

def save_pbm_performance_table(args, tfids):
    metric_keys = ["pearson.r","spearman.r","pearson.p","spearman.p","auc","auc.mean"]
    with open(args.reportdir+"/performance_pbm.txt", "w") as f:
        f.write("protein\t%s\n" % "\t".join(metric_keys))
        for tfid in ["TF_%d" % i for i in range(1,67)]:
            f.write(tfid)
            if tfid in tfids:
                metrics = util.load_metrics(args.finaldir+"/"+tfid+"/metrics.txt")
                for key in metric_keys:
                    f.write("\t%s" % metrics["test"].get(key,np.nan))
            f.write("\n")

    return

#################################

def save_chip_performance_table(args, tfids):
    with open(args.reportdir+"/performance_chipseq.txt", "w") as f:
        f.write("Background\tFile\tauc.mean\tauc.std\n")
        chipseqids = ["TF_%d"%i for i in [23,25,31,40,44]]
        for tfid in chipseqids:
            if tfid not in tfids:
                continue
            for windowsize in [51,100]:
                for background in ["dinuc", "full_genomic", "genomic"]:
                    seqfile = "../data/dream5/chipseq/%s_CHIP_%d_%s.seq" % (tfid, windowsize, background)
                    invivodata = util.datasource.fromtxt(seqfile+"[0]", None, seqfile+"[1]", sequencenames=["seq"], targetnames=[tfid])
                    invivodata.targetnames = [tfid]

                    predictions = util.predict(invivodata, args.finaldir, args.reportdir, include=[tfid])
                    z = predictions[tfid].ravel()
                    y = invivodata.Y.ravel()
                    metrics = util.calc_metrics(z, y)
                    s = "%s\t%s\t%.3f\t%.3f\n" % (background, seqfile, metrics['auc.mean'], metrics['auc.std'])
                    print s,
                    f.write(s)


#################################

def save_report(modeldir, reportdir, tfids, index_metric="auc"):

    util.makepath(reportdir)
    index_html = open(reportdir+"/index.html",'w')
    index_html.write("<html><head><title>Training report</title></head><body>\n")
    index_html.write("<table cellspacing=0 cellpadding=5 border=1>\n")
    index_html.write("<tr><th>Name</th><th>train %s</th><th>test %s</th></tr>\n" % (index_metric, index_metric))

    for tfid in ["TF_%d" % i for i in range(1,67)]:
        if tfid not in tfids:
            continue
        print tfid
        util.makepath(reportdir+"/"+tfid)

        # Load PFMs, convert them to logo images, and dump them to the report directory
        with open(reportdir+"/%s.pfms.pkl"%tfid) as f:
            _ = cPickle.load(f)
            pfms = _["pfms"]
            ics  = _["ic"]
            counts = _["counts"]
        pfm_order = np.argsort(-ics)
        logos = []
        for j in pfm_order:
            pfm = pfms[j]
            ic  = ics[j]
            if ic <= 0:
                continue

            pfm_rev = np.fliplr(np.flipud(pfm))
            logo_fwd = deepity.tape2logo.tape2logo(pfm.T,     height=50, letterwidth=10, bufferzoom=4, vmax=1.0, style="seqlogo", rna=False)
            logo_rev = deepity.tape2logo.tape2logo(pfm_rev.T, height=50, letterwidth=10, bufferzoom=4, vmax=1.0, style="seqlogo", rna=False)
            logo_filename = "%s/%s/pfm%02d" % (reportdir, tfid, len(logos))
            scipy.misc.imsave(logo_filename+"_fwd.png", logo_fwd)
            scipy.misc.imsave(logo_filename+"_rev.png", logo_rev)
            logos.append((j, os.path.basename(logo_filename)+"_fwd.png", ic, counts[j]))

        # Load train/test metrics so we can print them in the HTML report
        metrics_file = "%s/%s/metrics.txt" % (modeldir, tfid)
        metrics = deepity.load_metrics(metrics_file)

        # Add row for this TF in main index.html
        index_html.write("<tr>\n")
        index_html.write("<td><a href=\"%s/index.html\">%s</a></td>\n" % (tfid, tfid))
        if index_metric=="auc":
            index_html.write("<td>%s        &plusmn; %s</td>\n" % (metrics["train"]["auc.mean"], metrics["train"]["auc.std"]))
            index_html.write("<td><b>%s</b> &plusmn; %s</td>\n" % (metrics["test"]["auc.mean"],  metrics["test"]["auc.std"]))
        elif index_metric=="pearson":
            index_html.write("<td>%s        (p=%s)</td>\n" % (metrics["train"]["pearson.r"], metrics["train"]["pearson.p"]))
            index_html.write("<td><b>%s</b> (p=%s)</td>\n" % (metrics["test"]["pearson.r"],  metrics["test"]["pearson.p"]))
        index_html.write("</tr>\n")

        # Build page showing filters and sequence logos for this specific TF
        tf_html = open(reportdir+"/"+tfid+"/index.html", 'w')
        tf_html.write("<html><head><title>Training report - %s</title></head><body>\n" % tfid)
        tf_html.write("<h2>%s</h2>\n" % tfid)
        # tf_html.write("<a href=\"../gmaps/%s/index.html\">gradient maps</a>)<hr/>\n"%(tfid,tfid))
        tf_html.write("""
        <script language="javascript">
        function toggle_strand()
        {
            var pfms = document.getElementsByClassName('pfm')
            for (var i = 0; i < pfms.length; i++)
                if (pfms[i].src.search("_fwd") != -1)
                    pfms[i].src = pfms[i].src.replace("_fwd","_rev");
                else if (pfms[i].src.search("_rev") != -1)
                    pfms[i].src = pfms[i].src.replace("_rev","_fwd");
        }
        </script></head><body>
        """)

        with open(metrics_file) as f:
            metrics_text = f.read()
        tf_html.write("<pre>%s</pre>\n"% metrics_text)
        if os.path.exists(modeldir+"/%s/predict.scatter-test.png" % tfid):
            tf_html.write("<table cellspacing=0 cellpadding=0 style=\"display:inline;margin-right:20px;\"><tr><td align=center>TEST predictions</td></tr><tr><td><img src=\"../../final/%s/predict.scatter-test.png\"/></td></tr></table>\n" % tfid)
        if os.path.exists(modeldir+"/%s/predict.scatter-train.png" % tfid):
            tf_html.write("<table cellspacing=0 cellpadding=0 style=\"display:inline\"><tr><td align=center>TRAINING predictions</td></tr><tr><td><img src=\"../../final/%s/predict.scatter-train.png\"/></td></tr></table>\n" % tfid)

        # Then generate a table for the complete model
        tfdir = modeldir+"/"+tfid
        tf_html.write("<hr/><h3>Feature Logos</h3>\n")
        tf_html.write("<input type=\"button\" value=\"toggle strand\" onclick=\"toggle_strand();\"/><br/>")
        tf_html.write("<table cellspacing=0 cellpadding=4 border=0>\n")
        for filter_index, logo, ic, count in logos:
            tf_html.write("<tr><td>%d</td><td><img src=\"%s\" class=\"pfm\"/></td><td>%.1f bits,</td><td> %d activations</td></tr>" % (filter_index, logo, ic, count))
        tf_html.write("</table><br/><br/>\n")


        # Now show the actual model.
        '''
        shutil.copy(tfdir_final[0]+"/fold0.report/filters_.conv_seq(0).color.png", basedir+"report/%s/final_filters.png" % (tfid))
        shutil.copy(tfdir_final[0]+"/fold0.report/filters_.conv_seq(0).logo.png",  basedir+"report/%s/final_logos.png" % (tfid))
        shutil.copy(tfdir_final[0]+"/fold0.report/filters_.conv_seq(1).png",       basedir+"report/%s/final_biases.png"  % (tfid))
        shutil.copy(tfdir_final[0]+"/fold0.report/filters_.combiner.png",          basedir+"report/%s/final_weights.png" % (tfid))
        os.system("convert %sreport/%s/final_filters.png -rotate -90 %sreport/%s/final_filters_rot.png" % (basedir,tfid,basedir,tfid))
        '''

        tf_html.write("<hr/><h3>Actual DeepBind model</h3>\n")
        if os.path.exists(tfdir + "/model.conv_seq(1).color.png"):
            tf_html.write("Filters:<br/>\n")
            tf_html.write("<img src=\"../../final/%s/model.conv_seq(1).color.png\"/><br/>\n" % tfid)

        if os.path.exists(tfdir + "/model.combiner.png"):
            tf_html.write("<br/>Combiner layer:<br/>\n")
            tf_html.write("<img src=\"../../final/%s/model.combiner.png\"/><br/>\n" % tfid)

        tf_html.write("</body></html>\n")


    index_html.write("</table>\n")
    index_html.write("</body></html>\n")
    index_html.close()


if __name__=="__main__":
    #util.disable_multiprocessing()
    main()


