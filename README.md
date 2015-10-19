DEEPBIND training scripts
=========================

This source tree was distributed as part of the Nature Biotechnology 
supplementary software release for DeepBind. Users of DeepBind
are encouraged to instead use the latest source code and binaries 
for scoring sequences at

   http://tools.genes.toronto.edu/deepbind/

The commands below assume starting from the 'code' subdirectory.



REQUIREMENTS
------------
CUDA 7.0+ SDK
   https://developer.nvidia.com/cuda-downloads

Python 2.7+ 64-bit
   http://www.python.org/getit/

Numpy 1.8+ linked with MKL
Matplotlib 1.2+
Scipy 0.11+
PIL 1.1.7+
   http://www.lfd.uci.edu/~gohlke/pythonlibs/  (if on Windows)

Visual C++ 2012 / GCC 4.8+



BUILDING (WINDOWS)
------------------
You must have the CUDA 7.0+ runtime DLLs somewhere in your PATH. 
You need Visual C++ 2012 to compile the internal libraries:
   libs/smat/vcproj/smat.sln          (open and build)
   libs/deepity/vcproj/deepity.sln    (open and build)
   libs/kangaroo/vcproj/kangaroo.sln  (open and build)



BUILDING (LINUX)
----------------
You need g++ 4.8+.
The library makefiles assume cuda is installed to /usr/local/cuda
but this can be overridden with environment variable CUDADIR.

Once you've unzipped the source, compile the sub-projects as follows:
   cd libs/smat/src
   make
   cd ../py
   python run_tests.py

   cd ../../deepity/deepity_smat
   make

   cd ../../kangaroo/kangaroo_smat
   make

Add absolute paths for the following directories to your LD_LIBRARY_PATH

   code/libs/deepity/build/release/bin
   code/libs/smat/build/release/bin
   code/libs/kangaroo/build/release/bin



GETTING THE FULL DATA SETS
--------------------------
Due to the 150MB size limit on Supplementary Software, this package contains only 
small subset of the training and testing data, and not all scripts will run 
without the missing data. Please visit 

   http://tools.genes.toronto.edu/deepbind/nbtcode

to download the full data sets and include them in the 'data' subdirectory.
The full DREAM5 data is however included in the supplement.


 

TRAINING RBPs ON RNACOMPETE
---------------------------

Train RBP models on RNAcompete Set A, then test on Set B

   python deepbind_train_rnac.py A calib,train,test,report
   >> out/rnac/A/report/index.html    (model info)
   >> out/rnac/test/pbm/              (model predictions on SetB)

Train RBP models on RNAcompete Set AB, then test on in vivo sequences

   python deepbind_train_rnac.py AB calib,train,test,report
   >> out/rnac/AB/report/index.html            (model info)
   >> out/rnac/test/invivo/deepbind_all.txt    (AUCs when applied different ways)

For more information on the RNAcompete training and testing data, please
refer to Ray et al., 2013 (Nature, doi:10.1038/nature12311)


TRAINING TFs ON DREAM5
----------------------

Train TF models on DREAM5 training array (ME or HK depending on the TF),
then test on the opposite array:

   python deepbind_train_dream5.py calib,train,test,report
   >> out/dream5/report/index.html               (model info)
   >> out/dream5/report/performance_pbm.txt      (performance on held-out array design)
   >> out/dream5/report/performance_chipseq.txt  (performance on ChIP-seq peaks against various types of background sequence)


For more information on the 2013 DREAM5 Motif Recognition Challenge protocols and evaluation,
please refer to Weirauch et al., 2013 (Nature Biotechnology, doi:10.1038/nbt.2486)



TRAINING TFs ON ENCODE CHIP-SEQ
-------------------------------

Train TF models on ENCODE ChIP-seq peaks, then test on held-out subset of peaks.
See supplementary information if training/testing set is not clear from descriptions below.
Use top 500 even to train, top 500 odd to test:

   python deepbind_train_encode.py top calib,train,test,report
   >> out/encode/top/report/index.html    (model info)


Using all non-test peaks to train (top 500 even + 1000-and-beyond), then test on top 500 odd:

   python deepbind_train_encode.py all calib,train,test,report
   >> out/encode/all/report/index.html    (model info)



TRAINING TFs ON HT-SELEX
------------------------

Train TF models on HT-SELEX sequences, then test on held-out subset of sequences.
Using the SELEX cycles selected by Jolma et al.:

   python deepbind_train_selex.py --jolma calib,train,test,report
   >> out/selex/jolma/report/index.html

Using the SELEX cycles selected by Aipanahi et al.:

   python deepbind_train_selex.py calib,train,test,report
   >> out/selex/best/report/index.html

