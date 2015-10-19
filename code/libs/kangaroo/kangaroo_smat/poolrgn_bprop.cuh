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
#ifndef __KR_POOLRGN_BPROP_H__
#define __KR_POOLRGN_BPROP_H__

#include <smat_cuda/launch_util.h>
#include <smat/dtypes.h>
#include <base/util.h>
#include <base/assert.h>
#include "kangaroo_smat.h"
using namespace sm;

template <typename float_t>
__global__ 
void poolrgn_bprop_max_kernel(      float_t*  unpooledmaps, usize_t n,
                              const float_t*  pooledmaps,
                              const uindex_t* pooledmaps_argmax)
{
	const unsigned tx  = threadIdx.x;
	const unsigned bx  = blockIdx.x;
	const unsigned bdx = blockDim.x;
	const unsigned gdx = gridDim.x;
	#pragma unroll
	for (usize_t i = (usize_t)bdx*bx+tx; i < n; i += bdx*gdx)
		atomicAdd(&unpooledmaps[pooledmaps_argmax[i]],pooledmaps[i]);
}


template <unsigned bdx,               // = 32 threads per block in "filters" dimension
          bool want_avg,
          typename float_t>       
__global__ 
void poolrgn_bprop_sum_kernel(      float_t*  unpooledmaps, usize_t nfeaturemap,
                              const uindex_t* regions, usize_t nregion, bool per_region_step,
                              const float_t*  pooledmaps)
{
	const unsigned tx  = threadIdx.x;
	const unsigned bx  = blockIdx.x;
	const unsigned by  = blockIdx.y;

	unsigned featuremaps_this_block = ::min(bdx,nfeaturemap-bdx*bx);
	if (tx >= featuremaps_this_block)
		return; // we're past the last channel, so this thread is inactive in current implementation

	regions += per_region_step ? by*3 : by*2;
	uindex_t first = regions[0]; // first element index of region
	uindex_t last  = regions[1]; // last  element index of region (+1)
	isize_t  size  = last-first;
	first  = first*nfeaturemap + bdx*bx + tx; // first element index of region
	last   = last *nfeaturemap + bdx*bx;      // last  element index of region (+1)
	uindex_t step = per_region_step ? regions[2] : 1;         // the step size of this pooling region
	float_t pval = pooledmaps[by*nfeaturemap + bdx*bx + tx]; // pooled value this thread is responsible for distributing
	if (want_avg)
		pval /= (size+step-1)/step; // divide by number of items being pooled

	#pragma unroll
	for (uindex_t i = first; i < last; i += nfeaturemap*step)
		atomicAdd(&unpooledmaps[i],pval);
}

template <unsigned bdx,               // = 32 threads per block in "filters" dimension
          typename float_t>       
__global__ 
void poolrgn_bprop_all_kernel(      float_t*  unpooledmaps, usize_t nfeaturemap,
                              const uindex_t* regions, usize_t nregion, bool per_region_step,
                              const float_t*  pooledmaps,
                              const uindex_t* pooledmaps_argmax)
{
	const unsigned tx  = threadIdx.x;
	const unsigned bx  = blockIdx.x;
	const unsigned by  = blockIdx.y;

	unsigned featuremaps_this_block = ::min(bdx,nfeaturemap-bdx*bx);
	if (tx >= featuremaps_this_block)
		return; // we're past the last channel, so this thread is inactive in current implementation

	regions += per_region_step ? by*3 : by*2;
	uindex_t first = regions[0]; // first element index of region
	uindex_t last  = regions[1]; // last  element index of region (+1)
	isize_t  size  = last-first;
	first  = first*nfeaturemap + bdx*bx + tx; // first element index of region
	last   = last *nfeaturemap + bdx*bx;      // last  element index of region (+1)
	uindex_t step = per_region_step ? regions[2] : 1;         // the step size of this pooling region
	uindex_t j = by*nfeaturemap + bdx*bx + tx;

	typename device_tuple<float_t>::float2 pm = ((typename device_tuple<float_t>::float2*)pooledmaps)[j];
	float_t dxval = pm.x; // max-pooled delta 
	float_t dyval = pm.y; // average-pooled delta this thread is responsible for distributing
	dyval /= (size+step-1)/step; // divide by number of items being pooled

	#pragma unroll
	for (uindex_t i = first; i < last; i += nfeaturemap*step)
		atomicAdd(&unpooledmaps[i],dyval);
	atomicAdd(&unpooledmaps[pooledmaps_argmax[j]],dxval);
}

#endif // __KR_POOLRGN_BPROP_H__
