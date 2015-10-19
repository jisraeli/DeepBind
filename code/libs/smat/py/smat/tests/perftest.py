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
from testutil import *
import numpy as np
import smat        # want module name too
from smat import *
import timeit
import os,os.path
import matplotlib
matplotlib.use('Agg') # off-screen rendering
import matplotlib.pyplot as plt

#######################################################################

def _apply_unary(b,func,repeats,A,*args,**kwargs):
    for i in range(repeats):
        func(A,*args,**kwargs)
    b.sync()


def apply_unary(dt,b,n,m,repeats,func,*args,**kwargs):  # dtype to test, backend module to test
    A = b.rand(n,m,dt)
    _apply = lambda: _apply_unary(b,func,repeats,A,*args,**kwargs)
    _apply() # run once to stabilize running time
    b.sync()
    trials = [timeit.timeit(_apply,number=1)/repeats for i in range(5)]
    trials.sort()
    return trials[0]  # return best time
    

#######################################################################

def perftest_logistic(dt,b): return apply_unary(dt,b,128,1000,20,b.logistic),None
def perftest_exp(dt,b):      return apply_unary(dt,b,128,1000,20,b.exp),None
def perftest_tanh(dt,b):     return apply_unary(dt,b,128,1000,20,b.tanh),None
def perftest_softmax(dt,b):  return apply_unary(dt,b,1000,10,20,b.softmax),None
def perftest_repeat_x(dt,b): return apply_unary(dt,b,512,256,20,b.repeat,16,axis=1),None
def perftest_tile_x(dt,b):   return apply_unary(dt,b,512,256,20,b.tile,(1,16)),None

#######################################################################

def perftest_reduce_5Kx1(dt,b):   return apply_unary(dt,b,5000,1,100,b.sum,axis=None),None
def perftest_reducex_5Kx10(dt,b): return apply_unary(dt,b,5000,10,100,b.sum,axis=1),None
def perftest_reducey_5Kx10(dt,b): return apply_unary(dt,b,5000,10,100,b.sum,axis=0),None
def perftest_reducex_10x5K(dt,b): return apply_unary(dt,b,10,5000,100,b.sum,axis=1),None
def perftest_reducey_10x5K(dt,b): return apply_unary(dt,b,10,5000,100,b.sum,axis=0),None
def perftest_reduce_1Mx1(dt,b):   return apply_unary(dt,b,1000000,1,5,b.sum,axis=None),None
def perftest_reducex_1Mx10(dt,b): return apply_unary(dt,b,1000000,10,5,b.sum,axis=1),None
def perftest_reducey_1Mx10(dt,b): return apply_unary(dt,b,1000000,10,5,b.sum,axis=0),None
def perftest_reducex_10x1M(dt,b): return apply_unary(dt,b,10,1000000,5,b.sum,axis=1),None
def perftest_reducey_10x1M(dt,b): return apply_unary(dt,b,10,1000000,5,b.sum,axis=0),None


#######################################################################

def _apply_binary(b,func,repeats,A,B,*args,**kwargs):
    for i in range(repeats):
        func(A,B,*args,**kwargs)
    b.sync()


def apply_binary(dt,b,n,m,p,q,repeats,func,*args,**kwargs):  # dtype to test, backend module to test
    A = b.rand(n,m,dt)
    B = b.rand(p,q,dt)
    _apply = lambda: _apply_binary(b,func,repeats,A,B,*args,**kwargs)
    _apply() # run once to stabilize running time
    b.sync()
    trials = []
    for i in range(5):
        # push everything out of the cache, if any
        #X = b.ones((1024*1024,1))
        #X = None
        # do the performance test
        trials.append(timeit.timeit(_apply,number=1)/repeats)
        b.sync()
    trials.sort()
    return trials[0]  # return best time
    

#######################################################################

def mulsum(b,A,B):
    #return
    b.sum(A*B)


def perftest_mul(dt,b,N):  return apply_binary(dt,b,1,2**N,1,2**N,10,b.multiply),2**N
def perftest_dot(dt,b):    return apply_binary(dt,b,128,784,784,500,10,b.dot),128*784*500
def perftest_dot_nt(dt,b): return apply_binary(dt,b,128,784,500,784,10,b.dot_nt),128*784*500
def perftest_dot_tn(dt,b): return apply_binary(dt,b,784,128,784,500,10,b.dot_tn),128*784*500
def perftest_dot_tt(dt,b): return apply_binary(dt,b,784,128,500,784,10,b.dot_tt),128*784*500
def perftest_dot_nt_vec(dt,b): return apply_binary(dt,b,1,1024,1,1024,20,b.dot_nt),None
def perftest_mulsum_vec(dt,b): return apply_binary(dt,b,1,1024*1024,1,1024*1024,20,lambda A,B: mulsum(b,A,B)),None

#######################################################################

def perftest_bprop(dt,b):
    # Simulate training a 784-800-800-10 network on subset of MNIST
    trainsize = 2000
    batchsize = 200
    insize = 28*28
    hiddensize = 800
    outsize = 10

    dt_X = uint8 if uint8 in get_supported_dtypes() else float32

    times = {}
    X  = b.rand(trainsize,insize,dtype=dt_X)
    Y  = b.rand(trainsize,outsize,dt)
    W1 = b.rand(insize,hiddensize,dt)
    b1 = b.rand(1,hiddensize,dt)
    W2 = b.rand(hiddensize,hiddensize,dt)
    b2 = b.rand(1,hiddensize,dt)
    W3 = b.rand(hiddensize,outsize,dt)
    b3 = b.rand(1,outsize,dt)
    eta = 0.001

    num_epoch = 2
    b.sync()
    tic()
    for epoch in range(num_epoch):
        for i in range(trainsize/batchsize):
            Z0 = X[i*batchsize:i*batchsize+batchsize].astype(dt)
            Y0 = Y[i*batchsize:i*batchsize+batchsize]
            
            # forward pass
            A1 =  b.dot(Z0,W1) + b1
            Z1 =  b.logistic(A1)
            A2 =  b.dot(Z1,W2) + b2
            Z2 =  b.logistic(A2)
            A3 =  b.dot(Z2,W3) + b3
            A3 -= b.max(A3,axis=1).reshape((batchsize,1))               # for softmax stability
            Z3 =  b.exp(A3)/b.sum(exp(A3),axis=1).reshape((batchsize,1))  # calculate softmax 

            # backward pass
            D3 = (Z3-Y0)/trainsize
            dW3 = b.dot_tn(Z2,D3)
            db3 = sum(D3,axis=0)

            D2 = (Z2-Z2**2) * b.dot_nt(D3,W3)
            dW2 = b.dot_tn(Z1,D2)
            db2 = sum(D2,axis=0)

            D1 = (Z1-Z1**2) * b.dot_nt(D2,W2)
            dW1 = b.dot_tn(Z0,D1)
            db1 = sum(D1,axis=0)

            # Take gradient step
            W3 -= eta*dW3
            b3 -= eta*db3
            W2 -= eta*dW2
            b2 -= eta*db2
            W1 -= eta*dW1
            b1 -= eta*db1
    b.sync()
    return toc() / num_epoch, None

#######################################################################

class gridtest_reduce(object):

    def __init__(self,name,reduce,axis):
        self.name = name
        self.reduce = reduce
        self.A = None
        self.b = None
        self.axis = axis
        self.nrepeat = 1

    def configure(self,b,dt,n,m,nrepeat):
        self.A = b.rand(n,m,dt)
        self.b = b
        self.nrepeat = nrepeat

    def __call__(self):
        #print self.A.shape
        for i in range(self.nrepeat):
            x = self.reduce(self.A,axis=self.axis)
            '''
            y = np.sum(as_numpy(self.A),axis=self.axis)
            try:
                assert_close(x,y)
            except:
                print x.ravel()
                print y
                quit()
            '''
        self.b.sync()

    def nflop(self):
        n,m = self.A.shape
        if self.axis == 1:
            return (m-1)*n
        else:
            return (n-1)*m

#######################################################################

def run_perftest(log,dt,test,dtypes,argsets=None):
    testname = test.__name__.partition("_")[2]
    if dt not in dtypes:
        log.write(testname+"\n")
        return
    if argsets is None:
        argsets = [()]
    for args in argsets:
        print rpad("%s%s:%s..." % (testname,str(args),dtype_short_name[dt]),24),
        backends = [smat,np]
        best = { backend : np.inf for backend in backends }
        for backend in backends:
            flop = None
            for trial in range(3):
                runtime,flop = test(dt,backend,*args)
                best[backend] = min(best[backend],runtime)  # Take the best of three runs
            if flop is None:
                print(rpad("%s=%.4fms," % (backend.__package__,best[backend]*1000),17)), # print out the best milliseconds
            else:
                print(rpad("%s=%.3f GFLOPS," % (backend.__package__,flop/best[backend]/1e9),17)), # print out the best GFLOPS
        if best[np] > best[smat]:
            print("(%.1fx faster)" % (best[np]/best[smat]))
        else:
            print("(%.1fx SLOWER)" % (best[smat]/best[np]))
        log.write( rpad(testname,16)
                  +rpad("%.6f" % best[smat],10)
                  +rpad("%.6f" % best[np],10)
                  +"\n")

def run_gridtest(log,dt,gridtest,dtypes):
    if dt not in dtypes:
        log.write(gridtest.name+"\n")
        return
    #backends = [(smat,"smat"),(np,"numpy")]
    backends = [(smat,"smat")]
    base = 5L
    nsteps = 8
    nrepeat = 3
    max_size = 128*1024*1024
    for b,bname in backends:
        testname = "%s_%s_%s" % (bname,gridtest.name,dtype_short_name[dt])
        print rpad("%s..." % testname,24),
        gflops = np.zeros((nsteps,nsteps))
        #flops[:] = np.nan
        for mexp in range(nsteps):
            for nexp in range(nsteps):
                n,m = base**(nexp+1),base**(mexp+1)
                if n*m > max_size:
                    continue
                gridtest.configure(b,dt,n,m,nrepeat)
                b.sync()
                seconds = timeit.timeit(gridtest,number=1)/nrepeat
                gflops[nexp,mexp] = gridtest.nflop()/seconds/1000/1000/1000
        print
        
        msg = ""
        for row in gflops:
            for val in row:
                if not np.isnan(val):
                    msg += str(val)
                msg += "\t"
            msg.strip('\t')
            msg += "\n"
        

        log.write( rpad(testname,16) + "\n")
        log.write(msg)

        plt.figure(dpi=60)
        plt.title(testname + " performance (GFLOPS)")
        plt.xlabel('shape.x')
        plt.ylabel('shape.y')
        img = plt.imshow(gflops.squeeze(),origin='lower') #Needs to be in row,col order
        img.set_interpolation('nearest')
        plt.xticks(np.arange(nsteps),[base**(i+1) for i in range(nsteps)])
        plt.yticks(np.arange(nsteps),[base**(i+1) for i in range(nsteps)])
        plt.colorbar()
        #plt.show()
        plt.savefig(os.path.join("log",testname+".png"))

#######################################################################

def perftest():
    print '\n------------------- PERFORMANCE TESTS ----------------------\n'
    np.random.seed(42)
    set_backend_options(randseed=42,verbose=0,sanitycheck=False)
    if not os.path.exists("log"):
        os.makedirs("log")
    
    for dt in [float32,float64,int32,bool]:
        if dt not in get_supported_dtypes():
            continue

        # Record the performance results in a text file that can be 
        # imported into a spreadsheet if so desired.
        perflog = os.path.join("log","smatperf-%s.txt" % dt.__name__)
        print "----- Generating %s ------" % perflog
        with open(perflog,"w") as log:
            log.write( rpad("test",16)
                      +rpad("smat",10)
                      +rpad("numpy",10)
                      +"\n")
            
            # Performance tests with dead code elimination disabled
            reset_backend(sanitycheck=False,elimdeadcode=False) # ,verbose=1,log=["exec"]
            #run_perftest(log,dt,perftest_mul,dtypes_float,((i,) for i in range(4,25)))
            run_perftest(log,dt,perftest_mul,dtypes_float,((i,) for i in [5,10,20,26]))
            '''
            run_perftest(log,dt,perftest_logistic     ,dtypes_float)
            run_perftest(log,dt,perftest_exp          ,dtypes_float)
            run_perftest(log,dt,perftest_tanh         ,dtypes_float)
            run_perftest(log,dt,perftest_softmax      ,dtypes_float)
            run_perftest(log,dt,perftest_dot          ,dtypes_float)
            run_perftest(log,dt,perftest_dot_nt       ,dtypes_float)
            run_perftest(log,dt,perftest_dot_tn       ,dtypes_float)
            run_perftest(log,dt,perftest_dot_tt       ,dtypes_float)
            run_perftest(log,dt,perftest_dot_nt_vec   ,dtypes_float)
            '''
            #run_perftest(log,dt,perftest_mulsum_vec   ,dtypes_float)
            '''
            run_perftest(log,dt,perftest_repeat_x     ,dtypes_generic)
            run_perftest(log,dt,perftest_tile_x       ,dtypes_generic)
            run_perftest(log,dt,perftest_reduce_5Kx1  ,dtypes_generic)
            run_perftest(log,dt,perftest_reducex_5Kx10,dtypes_generic)
            run_perftest(log,dt,perftest_reducey_5Kx10,dtypes_generic)
            run_perftest(log,dt,perftest_reducex_10x5K,dtypes_generic)
            run_perftest(log,dt,perftest_reducey_10x5K,dtypes_generic)
            run_perftest(log,dt,perftest_reduce_1Mx1  ,dtypes_generic)
            run_perftest(log,dt,perftest_reducex_1Mx10,dtypes_generic)
            run_perftest(log,dt,perftest_reducey_1Mx10,dtypes_generic)
            run_perftest(log,dt,perftest_reducex_10x1M,dtypes_generic)
            run_perftest(log,dt,perftest_reducey_10x1M,dtypes_generic)
            

            # More performance tests, where dead code elimination is now allowed (the default)
            reset_backend(elimdeadcode=True)
            run_perftest(log,dt,perftest_bprop,dtypes_float)
            

            reset_backend(elimdeadcode=True)
            run_gridtest(log,dt,gridtest_reduce("sum",sum,None),dtypes_float)
            run_gridtest(log,dt,gridtest_reduce("sum_y",sum,0),dtypes_float)
            run_gridtest(log,dt,gridtest_reduce("sum_x",sum,1),dtypes_float)
            '''

