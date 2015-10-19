# Contents of the DeepFind data directory
# July 20, 2015
# Babak Alipanahi
# University of Toronto

Original SNVs are from "A general framework for estimating the relative pathogenicity
of human genetic variants" by Kircher et al. (http://dx.doi.org/10.1038/ng.2892)

derived: high-frequency human-derived allels
simulated: simulated mutations

AVINPUT files:
The SNVs mapped *only* to promoters by ANNOVAR (version: Mon, 20 May 2013) using this command:

annotate_variation.pl --geneanno -dbtype refGene -buildver hg19 [simulated|derived].avinput --outfile my_output --neargene 2000 -hgvs ~/humandb/

SEQ.GZ files:
For every SNV, two lines are added to SEQ files: 
1. The wildtype sequence: 51 bp around the position of SNV
2. The mutant sequence: the wildtype sequence with the middle base mutated as indicated
by the SNV

*_cons.npz:
Conservation information for every SNV:
1-3: phastCons (46way): mammalian, vertebrates, and primates
4-6: phastCons (46way): mammalian, vertebrates, and primates
7-9: phastCons (46way): mammalian, vertebrates, and primates (averaged over 51 bp centered at the SNV)

*_feats.npz:
For each SNV this file contains:
1. Whether the SNV is a transversion
2. The normalized distance to the closest splice site

tfs_* folders:
Contain the predictions made by DeepBind (10 TFs only) for each sequence in the .SEQ.GZ files.
The complete set of predictions are available at http://tools.genes.toronto.edu/deepbind/nbtcode
