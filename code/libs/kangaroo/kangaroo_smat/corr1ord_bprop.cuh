// Copyright (c) 2015, Andrew Delong and Babak Alipanahi All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Author's note: 
//     This file was distributed as part of the Nature Biotechnology 
//     supplementary software release for DeepBind. Users of DeepBind
//     are encouraged to instead use the latest source code and binaries 
//     for scoring sequences at
//        http://tools.genes.toronto.edu/deepbind/
// 
#ifndef __KR_CORRELATE1ORD_BPROP_H__
#define __KR_CORRELATE1ORD_BPROP_H__

#include <smat_cuda/launch_util.h>
#include <smat/dtypes.h>
#include <base/util.h>
#include <base/assert.h>
using namespace sm;

//#define SM_TRACE(fmt,...)       { printf("%2d: " fmt,tx,__VA_ARGS__); }
#define SM_TRACE(fmt,...) 
#define SM_TRACE0(fmt,...)      { if (tx == 0) SM_TRACE(fmt,__VA_ARGS__); }

template <unsigned bdx,          // = # threads per block
          unsigned wdy,          // = process at most bdx*wdy input elements
          unsigned __m,          // = # filter elements to cache in registers for innermost loop
          unsigned __n,          // = # input  elements to cache in registers for innermost loop
          unsigned nchannel,     // = # input channels, e.g. 4 for RNA/DNA ordinals.
          bool check_m,          // is there a chance that innermost loop may read beyond end of a filter in W?
          bool loop_m,           // is __m == m so that we do not even need to loop over the filter length?
          bool check_n,          // is there a chance that innermost loop may read beyond end of X?
          typename float_t>
__global__ 
void corr1ord_bprop_W_kernel(      float_t* __restrict__ dW, unsigned m, unsigned nfilter,
                           const uint8_t* __restrict__  X, unsigned n, unsigned _n,
                           const float_t* __restrict__ dZ)
{
	// The inner workings of this function are essentially the same as 
	// for corr1ord_bprop, except the outer loop is over j instead of _i
	const unsigned tx  = threadIdx.x;
	const unsigned bx  = blockIdx.x;
	const unsigned by  = blockIdx.y;
	SM_TRACE0("start: n=%d, _n=%d, m=%d, nfilter=%d, check_n=%d, check_m=%d\n",n,_n,m,nfilter,(int)check_n,(int)check_m)
	__shared__ uint8_t _X[wdy*bdx*sizeof(uint32_t)];   // stores bdx*wdy*sizeof(uint32_t) sequence elements
	float_t __dW[__m][nchannel];

#ifdef _DEBUG
	for (unsigned j = 0; j < __m; ++j)
		for (unsigned k = 0; k < nchannel; ++k)
			__dW[j][k] = -100000;                  // fill __W with garbage
	for (unsigned y = 0; y < wdy; ++y)
		((uint32_t*)_X)[y*bdx+tx] = 0xcccccccc;   // fill _X  with garbage
	__syncthreads();
#endif
	unsigned i = _n*by;
	#pragma unroll
	for (unsigned y = 0; y < wdy; ++y) {
		unsigned t = y*bdx+tx;
		if (!check_n || (i + t*sizeof(uint32_t) < n+m-1)) {            SM_TRACE("_X[%d:%d]=X[%d:%d]\n",t*(unsigned)sizeof(uint32_t),(t+1)*(unsigned)sizeof(uint32_t),i+t*(unsigned)sizeof(uint32_t),i+(t+1)*(unsigned)sizeof(uint32_t));
			((uint32_t*)_X)[t] = ((uint32_t*)(X+i))[t];
		}
	}
	unsigned f = bdx*bx+tx;
	if (f >= nfilter)
		return;

	for (unsigned j = 0; j < (loop_m ? m : 1); j += __m) {                    SM_TRACE0("j=%d\n",j);
		// INIT __dW[:,:] = 0
		#pragma unroll
		for (unsigned __j = 0; __j<__m; ++__j) {
			#pragma unroll
			for (unsigned k = 0; k < nchannel; ++k)
				__dW[__j][k] = 0;
		}

		for (unsigned _i = 0; _i<_n && (!check_n || i+_i<n); _i += __n) {     SM_TRACE0("_i=%d\n",_i);

			// LOAD dZ -> __dZ
			// Thread-local array for accumulating sums.
			float_t  __dZ[__n];
			#pragma unroll
			for (unsigned __i = 0; __i < __n && (!check_n || i+_i+__i<n); ++__i) {  SM_TRACE("__dZ[%d]=dZ[%d,%d]\n",__i,i+_i+__i,f);
				__dZ[__i] = dZ[(i+_i+__i)*nfilter + f];
			}
			
			// INNERMOST LOOPS.
			// Accumulate into thread-local __dW array using elements of sensitivities __dZ indexed by ordinals in shared-mem _X
			//
			#pragma unroll
			for (unsigned __j = 0; __j < __m && (!check_m || j+__j<m); ++__j) {
				#pragma unroll
				for (unsigned __i = 0; __i<__n && (!check_n || i+_i+__i<n); ++__i) {
					uint8_t k = _X[_i+__i+j+__j];
					SM_TRACE0("TERM: i,j=(%d,%d), X[%d]=%d\n",i+_i+__i,j+__j,  i+_i+__i+j+__j, (unsigned)k);
					if (k < nchannel) {
						__dW[__j][k] += __dZ[__i];
					} else if (k != 254) {  // 254 magic number means "dropout this input"
						float_t z = __dZ[__i]*((float_t)1/nchannel);
						#pragma unroll
						for (unsigned __k = 0; __k < nchannel; ++__k)
							__dW[__j][__k] += z;
					}
				}
			}
		}

		// STORE __dW[0:__m,:] to dW[j:j+__m,:,f]
		#pragma unroll
			for (unsigned __j = 0; __j<__m && (!check_m || j+__j<m); ++__j) {    SM_TRACE("dW[%d,:,%d]=__dW[%d,:]\n",j+__j,f,__j);
			#pragma unroll
			for (unsigned k = 0; k < nchannel; ++k)
				atomicAdd(&dW[(nchannel*(j+__j) + k)*nfilter + f], __dW[__j][k]);
		}
	}
}




//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////







template <unsigned bdx,          // = # threads per block
          unsigned wdy,          // = process at most bdx*wdy input elements
          unsigned __m,          // = # filter elements to cache in registers for innermost loop
          unsigned __n,          // = # input  elements to cache in registers for innermost loop
          unsigned nchannel,     // = # input channels, e.g. 4 for RNA/DNA ordinals.
          bool check_m,          // is there a chance that innermost loop may read beyond end of a filter in W?
          bool loop_m,           // is __m == m so that we do not even need to loop over the filter length?
          bool check_n,          // is there a chance that innermost loop may read beyond end of X?
          typename float_t>
__global__ 
void corr1ord_bprop_X_kernel(    float_t* __restrict__ dX, unsigned n, unsigned _n,
                           const float_t* __restrict__  W, unsigned m, unsigned nfilter,
                           const float_t* __restrict__ dZ)
{
}









#undef SM_TRACE
#undef SM_TRACE0

#endif // __KR_CORRELATE1ORD_BPROP_H__
