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
#ifndef __KR_POOLRGN_H__
#define __KR_POOLRGN_H__

#include <smat/dtypes.h>
#include <base/util.h>
#include <base/assert.h>
#include "kangaroo_smat.h"
using namespace sm;

template <unsigned bdx,               // = 64 threads per block in "channels" dimension
          typename float_t>
__global__ 
void poolrgn_max_kernel(const float_t* unpooledmaps, usize_t nfeaturemap,
                        const uindex_t* regions, bool per_region_step,
                        float_t*  pooledmaps,
                        uindex_t* pooledmaps_argmax)
{
	const unsigned tx  = threadIdx.x;
	const unsigned bx  = blockIdx.x;
	const unsigned by  = blockIdx.y;

	unsigned featuremaps_this_block = ::min(bdx,nfeaturemap-bdx*bx);
	if (tx >= featuremaps_this_block)
		return; // we're past the last featuremap, so this thread is inactive in current implementation

	regions += per_region_step ? by*3 : by*2;
	uindex_t first = regions[0]*nfeaturemap + bdx*bx + tx; // first element index of region
	uindex_t last  = regions[1]*nfeaturemap + bdx*bx;      // last  element index of region (+1)
	uindex_t step  = per_region_step ? regions[2] : 1;     // the step size of this pooling region
	float_t  x    = unpooledmaps[first]; // first samples element for this thread
	uindex_t  xi   = first;

	#pragma unroll
	for (uindex_t i = first+nfeaturemap*step; i < last; i += nfeaturemap*step) {
		if (unpooledmaps[i] > x) {
			x = unpooledmaps[i];
			xi = i;
		}
	}

	uindex_t j = by*nfeaturemap + bdx*bx + tx;
	pooledmaps[j] = x;
	if (pooledmaps_argmax)
		pooledmaps_argmax[j] = xi;
}



template <unsigned bdx,               // = 64 threads per block in "channels" dimension
          bool want_avg,
          typename float_t>
__global__ 
void poolrgn_sum_kernel(const float_t*  unpooledmaps, usize_t nfeaturemap,
                        const uindex_t* regions, bool per_region_step,
                        float_t* pooledmaps)
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
	isize_t size  = last-first;
	first  = first*nfeaturemap + bdx*bx + tx; // first element index of region
	last   = last *nfeaturemap + bdx*bx;      // last  element index of region (+1)
	uindex_t step  = per_region_step ? regions[2] : 1;      // the step size of this pooling region
	float_t  x    = unpooledmaps[first]; // first input element for this thread

	#pragma unroll
	for (uindex_t i = first+nfeaturemap*step; i < last; i += nfeaturemap*step)
		x += unpooledmaps[i];
	if (want_avg)
		x /= (size+step-1)/step;

	pooledmaps[by*nfeaturemap + bdx*bx + tx] = x;
}

template <unsigned bdx,               // = 64 threads per block in "channels" dimension
          typename float_t>
__global__ 
void poolrgn_all_kernel(const float_t*  unpooledmaps, usize_t nfeaturemap,
                        const uindex_t* regions, bool per_region_step,
                        float_t* pooledmaps,
                        uindex_t* pooledmaps_argmax)
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
	isize_t size  = last-first;
	first  = first*nfeaturemap + bdx*bx + tx; // first element index of region
	last   = last *nfeaturemap + bdx*bx;      // last  element index of region (+1)
	uindex_t step  = per_region_step ? regions[2] : 1;      // the step size of this pooling region
	float_t  x  = unpooledmaps[first]; // first input element for this thread
	uindex_t xi = first; // index of max element
	float_t  y  = x; // average

	#pragma unroll
	for (uindex_t i = first+nfeaturemap*step; i < last; i += nfeaturemap*step) {
		if (unpooledmaps[i] > x) {
			x = unpooledmaps[i];
			xi = i;
		}
		y += unpooledmaps[i];
	}
	y /= (size+step-1)/step;

	uindex_t j = by*nfeaturemap + bdx*bx + tx;
	if (pooledmaps_argmax)
		pooledmaps_argmax[j] = xi;

	 // store (max,avg) at location j in a single coalesced write, at least for float type
	((typename device_tuple<float_t>::float2*)pooledmaps)[j] = device_tuple<float_t>::make_float2(x,y);
}

#endif // __KR_POOLRGN_H__
