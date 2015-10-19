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
import argparse
import gzip
import numpy as np
import deepbind_util as util

# Warning: The multiprocessing module may be used indirectly, so do not put any 
# unintentional statements outside of main().

def main():
    args   = loadargs()
    models = loadmodels(args)
    trdata = None
    util.globals.flags.push("clamp_targets", True)     # Clamp the top .995 percentile of target values, to avoid extremely large targets (e.g. 00038/MBNL suffers from this)
    util.globals.flags.push("normalize_targets", True) # Make the targets have unit variance

    if "calib" in args.steps:
        trdata = load_traindata(trdata, args)
        util.calibrate(models, trdata, args.calibdir, nfold=args.nfold, ncalib=args.ncalib, allfolds=True)

    if "train" in args.steps:
        trdata = load_traindata(trdata, args)
        util.train(models, trdata, args.calibdir, args.finaldir, nfold=1, ntrial=args.ntrial)

    if "test" in args.steps:
        save_test_predictions(args)

    if "report" in args.steps:
        save_pfms(args)
        ids, _ = get_chunktargets(args)
        ids = sorted(ids)
        util.save_report(args.finaldir, args.reportdir, ids, index_metric="pearson", rna=True)



#########################################################################

# Baseline AUCs computed beforehand, by applying RNAcompete PFMs to the in vivo test data sets and summing/maxing/averaging the per-position scores
rnac_all_aucs ="""Protein	File	RNAcompete ID	direct	max	avg	sum	len	direct.std	max.std	avg.std	sum.std	len.std
RBM4	RBM4.txt	RNCMPT00052	nan	0.57954	0.53739	0.59715	0.68759	nan	0.01316	0.01179	0.01116	0.01185
RBM4	RBM4.txt	RNCMPT00113	nan	0.57840	0.53501	0.60088	0.68759	nan	0.01395	0.01253	0.01129	0.01185
FUS	FUS.txt	RNCMPT00018	nan	0.26564	0.25616	0.39401	0.50894	nan	0.01001	0.00706	0.01116	0.00798
CPEB4	CPEB4.txt	RNCMPT00158	nan	0.46494	0.51228	0.45046	0.30682	nan	0.01787	0.01636	0.01699	0.01312
PUM2	PUM2.txt	RNCMPT00046	nan	0.92388	0.93771	0.93146	0.48923	nan	0.00570	0.00514	0.00533	0.01019
PUM2	PUM2.txt	RNCMPT00101	nan	0.91999	0.93490	0.92761	0.48923	nan	0.00599	0.00518	0.00549	0.01019
PUM2	PUM2.txt	RNCMPT00102	nan	0.92309	0.93684	0.92996	0.48923	nan	0.00556	0.00455	0.00517	0.01019
PUM2	PUM2.txt	RNCMPT00103	nan	0.92704	0.93744	0.93271	0.48923	nan	0.00556	0.00548	0.00568	0.01019
PUM2	PUM2.txt	RNCMPT00104	nan	0.93297	0.93929	0.93711	0.48923	nan	0.00505	0.00525	0.00503	0.01019
PUM2	PUM2.txt	RNCMPT00105	nan	0.91665	0.92218	0.91714	0.48923	nan	0.00616	0.00607	0.00628	0.01019
HuR	ELAVL1_Hafner.txt	RNCMPT00274	nan	0.80117	0.81005	0.80901	0.50492	nan	0.00766	0.00784	0.00760	0.00867
HuR	ELAVL1_Lebedeva.txt	RNCMPT00274	nan	0.80521	0.82457	0.80981	0.50640	nan	0.00509	0.00434	0.00491	0.01020
HuR	ELAVL1_MNASE.txt	RNCMPT00274	nan	0.81389	0.82680	0.82515	0.50907	nan	0.00927	0.00912	0.00902	0.00873
HuR	ELAVL1_Mukharjee.txt	RNCMPT00274	nan	0.88430	0.89534	0.89195	0.50741	nan	0.00291	0.00251	0.00245	0.00562
FMR1	FMR1_RIP_top1K.txt	RNCMPT00016	nan	0.76716	0.75601	0.89024	0.81773	nan	0.01173	0.01059	0.00754	0.01080
FMR1	FMR1_table2a_top1K.txt	RNCMPT00016	nan	0.56646	0.59232	0.59202	0.49905	nan	0.01074	0.01077	0.01125	0.01624
FMR1	FMR1_table2a_top5K.txt	RNCMPT00016	nan	0.57225	0.59665	0.59672	0.49990	nan	0.00679	0.00625	0.00634	0.00498
FMR1	FMR1_table2b_top1K.txt	RNCMPT00016	nan	0.56580	0.59210	0.59223	0.50034	nan	0.01480	0.01520	0.01524	0.01267
FMR1	FMR1_table2b_top5K.txt	RNCMPT00016	nan	0.55623	0.57975	0.57952	0.49705	nan	0.00476	0.00475	0.00478	0.00714
QKI	QKI.txt	RNCMPT00047	nan	0.92322	0.93431	0.93004	0.50657	nan	0.00611	0.00593	0.00576	0.01082
SRSF1	SFRS1.txt	RNCMPT00106	nan	0.85875	0.87453	0.87285	0.54262	nan	0.01365	0.01363	0.01384	0.02879
SRSF1	SFRS1.txt	RNCMPT00107	nan	0.85443	0.86763	0.86887	0.54262	nan	0.01640	0.01597	0.01555	0.02879
SRSF1	SFRS1.txt	RNCMPT00108	nan	0.84777	0.85921	0.85865	0.54262	nan	0.01782	0.01638	0.01690	0.02879
SRSF1	SFRS1.txt	RNCMPT00109	nan	0.84414	0.85777	0.85645	0.54262	nan	0.01754	0.01659	0.01711	0.02879
SRSF1	SFRS1.txt	RNCMPT00110	nan	0.79507	0.81253	0.81173	0.54262	nan	0.01474	0.01493	0.01425	0.02879
SRSF1	SFRS1.txt	RNCMPT00163	nan	0.86275	0.87639	0.87327	0.54262	nan	0.01372	0.01296	0.01290	0.02879
Vts1p	Vts1p.txt	RNCMPT00082	nan	0.67892	0.68584	0.68750	0.56794	nan	0.01679	0.01765	0.01601	0.02029
Vts1p	Vts1p.txt	RNCMPT00111	nan	0.68317	0.69048	0.69310	0.56794	nan	0.01815	0.02086	0.01746	0.02029
HNRNPA2B1	hnRNPA2B1.txt	RNCMPT00024	nan	0.54903	0.56342	0.56330	0.49990	nan	0.00959	0.01012	0.01000	0.00921
MBNL1	Mbnl1_B6Brain.txt	RNCMPT00038	nan	0.61514	0.62822	0.62443	0.50256	nan	0.00614	0.00593	0.00599	0.00849
MBNL1	Mbnl1_B6Heart.txt	RNCMPT00038	nan	0.60240	0.61387	0.61421	0.52043	nan	0.01131	0.01098	0.01126	0.01359
MBNL1	Mbnl1_B6Muscle.txt	RNCMPT00038	nan	0.60721	0.61079	0.61372	0.51123	nan	0.02156	0.02032	0.02063	0.01794
MBNL1	Mbnl1_B129Brain.txt	RNCMPT00038	nan	0.58529	0.59620	0.59205	0.49597	nan	0.00416	0.00431	0.00413	0.00369
MBNL1	Mbnl1_C2C12.txt	RNCMPT00038	nan	0.67202	0.68331	0.67960	0.50211	nan	0.00207	0.00213	0.00205	0.00269
PTBP1	PTBP1.txt	RNCMPT00268	nan	0.73912	0.77290	0.74726	0.51303	nan	0.00812	0.00741	0.00797	0.00916
PTBP1	PTBP1.txt	RNCMPT00269	nan	0.73258	0.76753	0.74049	0.51303	nan	0.00769	0.00737	0.00754	0.00916
LIN28	LIN28_hES_3UTR.txt	RNCMPT00036	nan	0.64158	0.66601	0.64597	0.49502	nan	0.00391	0.00440	0.00385	0.00364
LIN28	LIN28_hES_3UTR.txt	RNCMPT00162	nan	0.61623	0.63471	0.61896	0.49502	nan	0.00508	0.00574	0.00511	0.00364
LIN28	LIN28_hES_coding_exons.txt	RNCMPT00036	nan	0.61977	0.63383	0.62368	0.51739	nan	0.00737	0.00775	0.00789	0.00587
LIN28	LIN28_hES_coding_exons.txt	RNCMPT00162	nan	0.59463	0.60173	0.59540	0.51739	nan	0.00739	0.00747	0.00753	0.00587
LIN28	LIN28_v5_3UTR.txt	RNCMPT00036	nan	0.62030	0.63684	0.62011	0.48639	nan	0.00871	0.00824	0.00843	0.01021
LIN28	LIN28_v5_3UTR.txt	RNCMPT00162	nan	0.64040	0.65904	0.64265	0.48639	nan	0.00873	0.00834	0.00889	0.01021
LIN28	LIN28_v5_coding_exons.txt	RNCMPT00036	nan	0.59787	0.60869	0.59829	0.49904	nan	0.00947	0.00996	0.00914	0.00765
LIN28	LIN28_v5_coding_exons.txt	RNCMPT00162	nan	0.61716	0.62985	0.61869	0.49904	nan	0.01133	0.01165	0.01068	0.00765
MSI1	MSI.txt	RNCMPT00176	nan	0.63872	0.85384	0.61312	0.35007	nan	0.05670	0.03968	0.05180	0.05094
HNRNPA1	hnRNPA1.txt	RNCMPT00023	nan	0.63787	0.66090	0.66060	0.49896	nan	0.01842	0.01709	0.01739	0.01959
SHEP	SHEP_bg3_normal.txt	RNCMPT00068	nan	0.68478	0.59950	0.78179	0.80603	nan	0.02868	0.03654	0.01885	0.01860
SHEP	SHEP_bg3_normal.txt	RNCMPT00174	nan	0.67722	0.56947	0.77837	0.80603	nan	0.02655	0.03655	0.01767	0.01860
SHEP	SHEP_bg3_normal.txt	RNCMPT00175	nan	0.68777	0.60964	0.78866	0.80603	nan	0.02748	0.03319	0.01870	0.01860
SHEP	SHEP_bg3_stringent.txt	RNCMPT00068	nan	0.65654	0.61508	0.75189	0.76043	nan	0.02804	0.02777	0.02695	0.03001
SHEP	SHEP_bg3_stringent.txt	RNCMPT00174	nan	0.67423	0.59333	0.75680	0.76043	nan	0.02515	0.02641	0.02562	0.03001
SHEP	SHEP_bg3_stringent.txt	RNCMPT00175	nan	0.66576	0.62182	0.75435	0.76043	nan	0.02719	0.02632	0.02765	0.03001
SHEP	SHEP_kc_normal.txt	RNCMPT00068	nan	0.58452	0.66421	0.60800	0.48666	nan	0.02354	0.02042	0.02061	0.01753
SHEP	SHEP_kc_normal.txt	RNCMPT00174	nan	0.60444	0.68226	0.61162	0.48666	nan	0.01583	0.01687	0.01707	0.01753
SHEP	SHEP_kc_normal.txt	RNCMPT00175	nan	0.60845	0.68181	0.62068	0.48666	nan	0.01919	0.01680	0.01763	0.01753
SHEP	SHEP_kc_stringent.txt	RNCMPT00068	nan	0.57643	0.66430	0.58256	0.44887	nan	0.02101	0.02489	0.02209	0.01979
SHEP	SHEP_kc_stringent.txt	RNCMPT00174	nan	0.56663	0.65829	0.56107	0.44887	nan	0.02144	0.02681	0.02334	0.01979
SHEP	SHEP_kc_stringent.txt	RNCMPT00175	nan	0.57928	0.66535	0.58411	0.44887	nan	0.02430	0.02639	0.02401	0.01979
IGF2BP2	IGF2BP1-3.txt	RNCMPT00033	nan	0.58948	0.60332	0.57859	0.49313	nan	0.00662	0.00529	0.00624	0.00703
TARDBP	TARDBP_iCLIP.txt	RNCMPT00076	nan	0.69432	0.76630	0.74933	0.50791	nan	0.00441	0.00413	0.00402	0.00548
TARDBP	TARDBP_RIP.txt	RNCMPT00076	nan	0.51319	0.51618	0.50674	0.48739	nan	0.01787	0.01786	0.01817	0.02001
TAF15	TAF15.txt	RNCMPT00018	nan	0.27793	0.26921	0.39900	0.50233	nan	0.01085	0.01096	0.01464	0.01268
TIA1	TIA1.txt	RNCMPT00077	nan	0.77897	0.80216	0.78685	0.46814	nan	0.00993	0.00899	0.01005	0.01164
TIA1	TIA1.txt	RNCMPT00165	nan	0.79207	0.80957	0.79501	0.46814	nan	0.01001	0.00883	0.00975	0.01164
TIAL1	TIAL1.txt	RNCMPT00077	nan	0.79665	0.82024	0.80884	0.49757	nan	0.00544	0.00530	0.00569	0.01029
TIAL1	TIAL1.txt	RNCMPT00165	nan	0.81188	0.82708	0.81648	0.49757	nan	0.00540	0.00484	0.00516	0.01029
LARK	Lark_shared.txt	RNCMPT00035	nan	0.60496	0.55183	0.63621	0.69572	nan	0.05016	0.05038	0.05092	0.04579
LARK	Lark_shared.txt	RNCMPT00097	nan	0.62910	0.58226	0.65882	0.69572	nan	0.05477	0.05532	0.05376	0.04579
LARK	Lark_shared.txt	RNCMPT00124	nan	0.63998	0.59298	0.68235	0.69572	nan	0.05991	0.05948	0.05403	0.04579
LARK	Lark_union.txt	RNCMPT00035	nan	0.64096	0.59872	0.66861	0.69956	nan	0.02697	0.02756	0.02547	0.02800
LARK	Lark_union.txt	RNCMPT00097	nan	0.65407	0.60527	0.66818	0.69956	nan	0.02743	0.02961	0.02634	0.02800
LARK	Lark_union.txt	RNCMPT00124	nan	0.63400	0.58167	0.66941	0.69956	nan	0.02865	0.03266	0.02820	0.02800""".split("\n")

# Generate test AUCs on in vivo data
invivo_ids = { "PUM2"      : ([ 46,101,102,103,104,105],["PUM2"]),
                "SRSF1"     : ([106,107,108,109,110,163],["SFRS1"]),
                #"FUS"       : ([19,88,89,90],            ["FUS"]),
                #"TAF15"     : ([19,88,89,90],            ["TAF15"]),
                "FUS"       : ([18],                     ["FUS"]),
                "TAF15"     : ([18],                     ["TAF15"]),
                "SHEP"      : ([68,174,175],             ["SHEP_bg3_normal","SHEP_bg3_stringent","SHEP_kc_normal","SHEP_kc_stringent"]),
                "Vts1p"     : ([82,111],                 ["Vts1p"]),
                "HuR"       : ([274],                    ["ELAVL1_Hafner","ELAVL1_Lebedeva","ELAVL1_MNASE","ELAVL1_Mukharjee"]),
                "LARK"      : ([35,97,124],              ["Lark_shared","Lark_union"]),
                "QKI"       : ([47],                     ["QKI"]),
                "FMR1"      : ([16],                     ["FMR1_RIP_top1K","FMR1_table2a_top1K","FMR1_table2a_top5K","FMR1_table2b_top1K","FMR1_table2b_top5K"]),
                "TIA1"      : ([77,165],                 ["TIA1"]),
                "TIAL1"     : ([77,165],                 ["TIAL1"]),
                "TARDBP"    : ([76],                     ["TARDBP_iCLIP","TARDBP_RIP"]),
                "PTBP1"     : ([268,269],                ["PTBP1"]),
                "HNRNPA1"   : ([23],                     ["hnRNPA1"]),
                "LIN28"     : ([36,162],                 ["LIN28_hES_3UTR","LIN28_hES_coding_exons","LIN28_v5_3UTR","LIN28_v5_coding_exons"]),
                "MSI1"      : ([176],                    ["MSI"]),
                "RBM4"      : ([52,113],                 ["RBM4"]),
                "IGF2BP2"   : ([33],                     ["IGF2BP1-3"]),
                "HNRNPA2B1" : ([24],                     ["hnRNPA2B1"]),
                "MBNL1"     : ([38],                     ["Mbnl1_B6Brain","Mbnl1_B6Heart","Mbnl1_B6Muscle","Mbnl1_B129Brain","Mbnl1_C2C12"]),
                "CPEB4"     : ([158],                    ["CPEB4"]),
                }

all_invivo_ids = sorted(sum([value[0] for value in invivo_ids.itervalues()],[]))
shorthandids = { #"invivo_sub" : ["RNCMPT00016", "RNCMPT00111", "RNCMPT00038", "RNCMPT00105", "RNCMPT00274", "RNCMPT00076", "RNCMPT00107", "RNCMPT00176", "RNCMPT00269", "RNCMPT00023"],
                 "invivo_sub" : ["RNCMPT00268", "RNCMPT00076", "RNCMPT00038", "RNCMPT00023"],
                 "invivo"     : ["RNCMPT%05d"%id for id in all_invivo_ids ],
                 }

def loadargs():
    # Give names to specific groups of RNAcompete ids
    args = argparse.ArgumentParser(description="Generate the RNAcompete experiments.")
    args.add_argument("mode", type=str, help="Either \"A\" (train A, test B), \"B\" (train B, test A), or \"AB\" (train AB, test invivo).")
    args = util.parseargs("rnac", args, shorthandids=shorthandids)
    args.calibdir  = args.calibdir.replace(args.outdir, args.outdir+"/"+args.mode)
    args.finaldir  = args.finaldir.replace(args.outdir, args.outdir+"/"+args.mode)
    args.reportdir = args.reportdir.replace(args.outdir,args.outdir+"/"+args.mode)
    args.outdir    = args.outdir+"/"+args.mode
    return args

#################################

def loadmodels(args):
    models = util.loadmodels(args, modeldir="cfg/regression/allpool")
    return models


#################################

def get_chunktargets(args):
    # Determine which targetnames we're responsible for
    targetnames = gzip.open("../data/rnac/targets.tsv.gz").readline().rstrip().split("\t")
    chunktargets = util.getchunktargets(args, targetnames)
    chunkcols = [i for i in range(len(targetnames)) if targetnames[i] in chunktargets]
    return chunktargets, chunkcols

def load_traindata(trdata, args):
    if trdata is not None:
        return trdata  # Training data was already loaded in earlier step.

    # In quick mode, only load a subset of the data
    maxrows = 10000 if args.quick else None
    chunktargets, chunkcols = get_chunktargets(args)
    
    # Load only a specific fold
    if args.mode == "A":
        trdata = util.datasource.fromtxt("../data/rnac/sequences.tsv.gz", None, "../data/rnac/targets.tsv.gz", targetcols=chunkcols, foldfilter="A",  maxrows=maxrows)
    elif args.mode == "B":
        trdata = util.datasource.fromtxt("../data/rnac/sequences.tsv.gz", None, "../data/rnac/targets.tsv.gz", targetcols=chunkcols, foldfilter="B",  maxrows=maxrows)
    elif args.mode == "AB":
        trdata = util.datasource.fromtxt("../data/rnac/sequences.tsv.gz", None, "../data/rnac/targets.tsv.gz", targetcols=chunkcols, foldfilter="AB", maxrows=maxrows)
    else:
        quit("Unrecognized mode")

    return trdata



#################################

def save_test_predictions(args):
    # In quick mode, only load a subset of the data
    maxrows = 10000 if args.quick else None
    chunktargets, chunkcols = get_chunktargets(args)
    tedata = {}

    abs_auc = lambda x: max(x, 1-x)

    if args.mode == "AB":
        
        invivo_data_dir = "../data/rnac/invivo"
        util.makepath(args.testdir+"/invivo")
        aucdump = open(args.testdir+"/invivo/deepbind_all.txt","w")
        aucdump.write("Protein\tFile\tModel\tauc\tauc.mean\tauc.std\n")

        for invivo_id in invivo_ids.keys():
            rnac_ids, invivo_files = invivo_ids[invivo_id]
            rnac_names = ["RNCMPT%05d" % id for id in rnac_ids]
            rnac_names = [name for name in rnac_names if name in chunktargets]
            if not rnac_names:
                continue
            
            for invivo_file in invivo_files:
                if not os.path.exists(invivo_data_dir + "/" + invivo_file + ".txt"):
                    continue
                print "File %s using models %s..." % (invivo_file, ",".join(rnac_names))

                # Convert the invivo sequence file into a format that predict.py expects,
                # i.e. with a Fold ID, Event ID, and Sequence column.
                data = util.datasource.fromtxt(invivo_data_dir+"/" + invivo_file + ".txt[1]", None, invivo_data_dir+"/" + invivo_file + ".txt[0]",
                                               sequencenames=["bound","seq"], targetnames=["bound"] )

                # First generate predictions based on "trivial" features for this particular invivo file
                sequences = [row[0] for row in data.sequences]
                labels = data.targets
                predictions = {}
                predictions["len"] = np.asarray([len(s) for s in sequences],np.float32).reshape((-1,1))
                predictions["A"]   = np.asarray([s.upper().count("A") / float(len(s)) for s in sequences], np.float32).reshape((-1,1))
                predictions["C"]   = np.asarray([s.upper().count("C") / float(len(s)) for s in sequences], np.float32).reshape((-1,1))
                predictions["G"]   = np.asarray([s.upper().count("G") / float(len(s)) for s in sequences], np.float32).reshape((-1,1))
                predictions["T"]   = np.asarray([(s.upper().count("U")+s.upper().count("T")) / float(len(s)) for s in sequences], np.float32).reshape((-1,1))
                predictions["GC"]  = predictions["G"] + predictions["C"]

                # Next, generate predictions for each model on this same data file
                data.targetnames = rnac_names
                data.Y = np.repeat(data.Y, len(rnac_names), 1)
                data.Ymask = np.repeat(data.Ymask, len(rnac_names), 1)
                data.targets = data.Y.copy()
                #pred = util.predict(data, "../data/rnac/pfms", args.reportdir, scan=20)
                pred = util.predict(data, args.finaldir, args.reportdir, scan=20)
                predictions.update(pred)

                # Dump all performance stats to the file
                for pname in sorted(predictions.keys(), key=lambda x: x if "RNCMPT" not in x else " "+x): # Make sure RNCMPT items go first in each group, for readability of all_aucs.txt
                    # Calculate the AUC of this particular prediction, of this particular model,
                    # on this particular invivo file.
                    z = predictions[pname].ravel()
                    y = np.array(labels).ravel()
                    metrics = util.calc_metrics(z, y)

                    # Write out a row indicating performance of each model on this file
                    aucdump.write("%s\t%s\t%s\t%.4f\t%.4f\t%.6f\n" % (invivo_id, invivo_file, pname, metrics["auc"], metrics["auc.mean"], metrics["auc.std"]))


        aucdump.close()
        
        
        # Re-open the AUC dump file, pull out all experiments associated with a single protein,
        # and collect only the best of each type 
        all_aucs = {}
        with open(args.testdir+"/invivo/deepbind_all.txt") as f:
            f.readline() # discard header line
            for line in f:
                protein_name, invivo_file, model_name, auc, auc_mean, auc_std = line.rstrip().split("\t")
                model_suffix = ".deepbind" if "RNCMPT" in model_name else ""
                if model_name == "GC":
                    continue
                all_aucs.setdefault(protein_name,[]).append({"file" : invivo_file, "model" : model_name+model_suffix, "auc" : float(auc), "auc.mean" : float(auc_mean), "auc.std" : float(auc_std)})

        # Open the rnac_all.txt and pull out the AUCs for the PFMs from the RNAcompete paper
        # The rnac_all.txt file is in a different format, for legacy reasons
        head = rnac_all_aucs[0].rstrip().split("\t")
        lines = [line.rstrip().split("\t") for line in rnac_all_aucs[1:]]
        cols = { item : head.index(item) for item in head }
        for line in lines:
            protein_name = line[0]
            for scantype in ("max","sum","avg","direct"):
                invivo_file = line[1].rsplit(".",1)[0]
                model_name = line[2]+"."+scantype
                auc = "nan"
                auc_mean = line[cols[scantype]]
                auc_std = line[cols[scantype+".std"]]
                if protein_name in all_aucs:
                    all_aucs[protein_name].append({"file" : invivo_file, "model" : model_name+".rnac", "auc" : float(auc), "auc.mean" : float(auc_mean), "auc.std" : float(auc_std)})


        for scantype in ("direct","max","sum","avg"):
            for modeltype in ("deepbind","rnac"):
                with open(args.testdir+"/invivo/%s_best_%s.txt" % (modeltype, scantype),"w") as f:
                    f.write("Protein\tFile")
                    f.write("\t"+"\t".join(["deeputil.model","deeputil.auc","deeputil.auc.mean","deeputil.auc.std"]))
                    f.write("\t"+"\t".join(["rnac.model",    "rnac.auc",    "rnac.auc.mean",    "rnac.auc.std"]))
                    f.write("\t"+"\t".join(["trivial.model", "trivial.auc", "trivial.auc.mean", "trivial.auc.std"]))
                    f.write("\n")
                    for protein in sorted(all_aucs.keys()):
                        trials = all_aucs[protein]
                        # For this particular protein, first find the best row based on non-trivial models, 
                        # e.g. among RNCMPTXXXXX.direct and RNCMPTXXXXX.max, while ignoring "A" and "len"
                        best = None
                        for trial in trials:
                            if trial["model"].endswith(scantype+"."+modeltype):
                                if best is None or trial["auc.mean"] > best["auc.mean"]:
                                    best = trial

                        # Also find the best trivial feature associated with this file
                        best_trivial = None
                        for trial in trials:
                            if trial["file"] == best["file"] and "RNCMPT" not in trial["model"]:
                                if best_trivial is None or abs_auc(trial["auc.mean"]) > abs_auc(best_trivial["auc.mean"]):
                                    best_trivial = trial

                        # Also find the best competing feature associated with this file
                        best_other = None
                        if modeltype == "rnac":
                            other_scantype = "max"
                        elif modeltype == "deepbind":
                            other_scantype = "avg"
                        else:
                            other_scantype = scantype

                        for trial in trials:
                            if trial["file"] == best["file"] and ("."+other_scantype+".") in trial["model"] and not trial["model"].endswith(other_scantype+"."+modeltype):
                                if best_other is None or trial["auc.mean"] > best_other["auc.mean"]:
                                    best_other = trial

                        best_deepbind = best if modeltype == "deepbind" else best_other
                        best_rnac     = best if modeltype == "rnac"     else best_other

                        f.write("%s\t%s\t" % (protein, best["file"]))
                        f.write("%s\t%.4f\t%.4f\t%.6f\t" % (best_deepbind["model"], best_deepbind["auc"], best_deepbind["auc.mean"], best_deepbind["auc.std"]))
                        f.write("%s\t%.4f\t%.4f\t%.6f\t" % (best_rnac["model"],     best_rnac["auc"],     best_rnac["auc.mean"],     best_rnac["auc.std"]))
                        f.write("%s\t%.4f\t%.4f\t%.6f\n" % (best_trivial["model"],  abs_auc(best_trivial["auc"]),  abs_auc(best_trivial["auc.mean"]), best_trivial["auc.std"]))
                    

    elif args.mode in ("A","B"):
        # Generate predictions on PBM probes from test set (i.e. if mode="A", the training set was "A" so the test set is "B")
        print "Loading PBM data...",
        testfold = "A" if args.mode == "B" else "B"
        pbmdata = util.datasource.fromtxt("../data/rnac/sequences.tsv.gz", None, "../data/rnac/targets.tsv.gz", targetcols=[0], foldfilter=testfold,  maxrows=maxrows)
        print "done"

        util.makepath(args.testdir+"/pbm")
        for targetname in chunktargets:
            print targetname
            pbmdata.targetnames = [targetname]
            predictions = util.predict(pbmdata, args.finaldir, args.reportdir, include=[targetname])
            Z = predictions[targetname].ravel()
            with gzip.open(args.testdir+"/pbm/%s-DB-%s.txt.gz"%(targetname, testfold), "w") as f:
                for z in Z:
                    f.write("%.4f\n"%z)

    else:
        quit("Unrecognized mode")

#################################

def save_pfms(args):
    maxrows = 10000 if args.quick else None
    chunktargets, chunkcols = get_chunktargets(args)

    print "Loading PBM data...",
    if args.mode == "A":
        testfold = "B"
    elif args.mode == "B":
        testfold = "A"
    else:
        testfold = "AB"
    pbmdata = util.datasource.fromtxt("../data/rnac/sequences.tsv.gz", None, "../data/rnac/targets.tsv.gz", targetcols=chunkcols, foldfilter=testfold,  maxrows=maxrows)
    print "done"

    util.save_featuremaps(pbmdata, args.finaldir, args.reportdir)


if __name__=="__main__":
    #util.disable_multiprocessing()
    main()

