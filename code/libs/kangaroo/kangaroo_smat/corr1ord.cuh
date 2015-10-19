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
#ifndef __KR_CORRELATE1ORD_H__
#define __KR_CORRELATE1ORD_H__

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
void corr1ord_kernel(const float_t* __restrict__ W, unsigned m, unsigned nfilter,
                     const uint8_t* __restrict__ X, unsigned n, unsigned _n,
                           float_t* __restrict__ Z)
{
	const unsigned tx  = threadIdx.x;
	const unsigned bx  = blockIdx.x;
	const unsigned by  = blockIdx.y;

	SM_TRACE0("start: n=%d, _n=%d, m=%d, nfilter=%d, check_n=%d, check_m=%d\n",n,_n,m,nfilter,(int)check_n,(int)check_m)

	// Shared array for holding a subset of sequence elements, used by all threads
	// simultaneously. Since 3.x has broadcast ability from shared mem, it makes 
	// sense to keep the sequence there.
	__shared__ uint8_t _X[wdy*bdx*sizeof(uint32_t)];   // stores bdx*wdy*sizeof(uint32_t) sequence elements

	// Dedicated thread-local register array for holding a 
	// subset of filter elements, specifically __m  at a time.
	float_t __W[__m][nchannel];

#ifdef _DEBUG
	for (unsigned j = 0; j < __m; ++j)
		for (unsigned k = 0; k < nchannel; ++k)
			__W[j][k] = -100000;                  // fill __W with garbage
	for (unsigned y = 0; y < wdy; ++y)
		((uint32_t*)_X)[y*bdx+tx] = 0xcccccccc;   // fill _X  with garbage
	__syncthreads();
#endif

	// LOAD X -> _X
	// Let i be the index of first element of X handled by this block.
	// Load global X[i:i+_n+m-1] into shared _X[0:_n+m-1].
	// Each thread is responsible for loading 4 consecutive bytes for each y=0:wdy.
	// Across all threads, this loads wdy*bdx*sizeof(uint32_t) total sequence elements into _X.
	// NOTE: the code below may load up to 3 bytes out of bounds of X, but this 
	//       should be ok since they will just sit unused in _X.
	unsigned i = _n*by;
	#pragma unroll
	for (unsigned y = 0; y < wdy; ++y) {
		unsigned t = y*bdx+tx;
		if (!check_n || (i + t*sizeof(uint32_t) < n+m-1)) {            SM_TRACE("_X[%d:%d]=X[%d:%d]\n",t*(unsigned)sizeof(uint32_t),(t+1)*(unsigned)sizeof(uint32_t),i+t*(unsigned)sizeof(uint32_t),i+(t+1)*(unsigned)sizeof(uint32_t));
			((uint32_t*)_X)[t] = ((uint32_t*)(X+i))[t];   // coalesced!: consecutive threads read consecutive 32-bit words
		}                                                 // word-aligned!: _n is guaranteed to be a multiple of sizeof(uint32_t) so memory access will be aligned.
	}                                                     // pitch-aligned?: only if _n is multiple of 256

	// Compute the global index of the filter that this thread (tx) will be responsible for.
	unsigned f = bdx*bx+tx; // each block uses only one thread per filter
	if (f >= nfilter)
		return;

	// OUTER LOOP: _i = 0:_n
	// This loop loads X[i:i+_n+m-1] (indirectly through _X, one subchunk at a time, with _i subchunk index, __n size of subchunk).
	// This loop stores Z[i:i+_n,f] directly to global memory, one subchunk at a time.
	for (unsigned _i = 0; _i<_n && (!check_n || i+_i<n); _i += __n) {    SM_TRACE0("_i=%d\n",_i);

		// INIT __Z[:] = 0
		// Thread-local array for accumulating sums.
		float_t  __Z[__n];
		#pragma unroll
		for (unsigned __i = 0; __i < __n; ++__i)
			__Z[__i] = 0;

		// INNER LOOP: j = 0:m
		// This thread loads __m consecutive elements from filter f,
		// computes all products involving that subset of filter elements, then moves
		// on to the next chunk of __m filter elements.
		for (unsigned j = 0; j < (loop_m ? m : 1); j += __m) {                    SM_TRACE0("j=%d\n",j);
			// LOAD W -> __W
			// Let j be the first element of W handled by pass through the outer loop.
			// If W[f,j,c] denotes filter f, element j, channel c, then we want to
			// load global W[j:j+__m,:,tx] into local __W[0:__m,:].
			// Note that W is stored in C-array order of [element,channel,filter].
			#pragma unroll
			for (unsigned __j = 0; __j<__m && (!check_m || j+__j<m); ++__j) {    SM_TRACE("__W[%d,:]=W[%d,:,%d]\n",__j,j+__j,f);
				#pragma unroll
				for (unsigned k = 0; k < nchannel; ++k)
					__W[__j][k] = W[(nchannel*(j+__j) + k)*nfilter + f]; // coalesced!: consecutive thread reads at consecutive f, and W stored with interleaved f.
			}                                                            // pitch-aligned?: only if __m*nchannel*sizeof(float_t) is multiple of 256


			// INNERMOST LOOPS.
			// Fill the thread-local __Z array using elements of thread-local filter __W indexed by ordinals in shared-mem _X
			//
			#pragma unroll
			for (unsigned __j = 0; __j < __m && (!check_m || j+__j<m); ++__j) {
				#pragma unroll
				for (unsigned __i = 0; __i<__n && (!check_n || i+_i+__i<n); ++__i) {
					uint8_t k = _X[_i+__i+j+__j];   // broadcasted: pull an individual ordinal from _X
					SM_TRACE0("TERM: i,j=(%d,%d), X[%d]=%d\n",i+_i+__i,j+__j,  i+_i+__i+j+__j, (unsigned)k);
					// As long as all threads in a warp are processing the same i+j, at the same time,
					// then the following branch won't cause divergence.
					if (k < nchannel) {
						__Z[__i] += __W[__j][k];
					} else if (k != 254) {  // 254 magic number means "dropout this input"
						float_t z = __W[__j][0];
						#pragma unroll
						for (unsigned __k = 1; __k < nchannel; ++__k)
							z += __W[__j][__k];
						__Z[__i] += z*((float_t)1/nchannel);
					}
				}
			} // innermost loops
		} // inner loop

		// Store __Z[0:__n] to Z[i+_i:i+_i+__n,f]
		#pragma unroll
		for (unsigned __i = 0; __i<__n && (!check_n || i+_i+__i<n); ++__i) {            SM_TRACE("Z[%d,%d]=__Z[%d]=%f\n",i+_i+__i,f,__i,__Z[__i]);
			Z[(i+_i+__i)*nfilter + f] = __Z[__i];  // coalesced!: consecutive threads write to consecutive filter index f
		}                                          // pitch-aligned?: only if bdx*sizeof(float_t) is multiple of 256
	} // outer loop
}

#undef SM_TRACE
#undef SM_TRACE0

#endif // __KR_CORRELATE1ORD_H__
