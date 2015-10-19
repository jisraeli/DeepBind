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
#include "poolrgn.cuh"


template <typename float_t>
void launch_poolrgn_dtype(cudaStream_t stream,
						  const float_t* unpooledmaps, usize_t nfeaturemap,
						  const uindex_t* regions,     usize_t nregion, bool per_region_step,
						        float_t* pooledmaps,   uindex_t* pooledmaps_argmax, pooltype_t ptype)
{
	const unsigned bdx = 128;  // 128 channels per block, 1 region per block. (TODO: this is not optimal)
	dim3 bdim(bdx,1);
	dim3 gdim(divup(nfeaturemap,bdx),nregion);
	switch (ptype) {
		case pt_max: poolrgn_max_kernel<bdx>      <<<gdim,bdim,0,stream>>>(unpooledmaps,nfeaturemap,regions,per_region_step,pooledmaps,pooledmaps_argmax); break;
		case pt_avg: poolrgn_sum_kernel<bdx,true ><<<gdim,bdim,0,stream>>>(unpooledmaps,nfeaturemap,regions,per_region_step,pooledmaps); break;
		case pt_sum: poolrgn_sum_kernel<bdx,false><<<gdim,bdim,0,stream>>>(unpooledmaps,nfeaturemap,regions,per_region_step,pooledmaps); break;
		case pt_all: poolrgn_all_kernel<bdx>      <<<gdim,bdim,0,stream>>>(unpooledmaps,nfeaturemap,regions,per_region_step,pooledmaps,pooledmaps_argmax); break;
		default: SM_UNREACHABLE();
	}
}

void launch_poolrgn(cudaStream_t stream, dtype_t dtype,
					const void*     unpooledmaps, usize_t   nfeaturemap,
					const uindex_t* regions,      usize_t   nregion, bool per_region_step,
					      void*     pooledmaps,   uindex_t* pooledmaps_argmax, pooltype_t ptype)
{
	if (nfeaturemap == 0 || nregion == 0)
		return;
	switch (dtype) {
	case f32: launch_poolrgn_dtype<float >(stream,(const float* )unpooledmaps,nfeaturemap,regions,nregion,per_region_step,(float* )pooledmaps,pooledmaps_argmax,ptype); break;
	case f64: launch_poolrgn_dtype<double>(stream,(const double*)unpooledmaps,nfeaturemap,regions,nregion,per_region_step,(double*)pooledmaps,pooledmaps_argmax,ptype); break;
	default: SM_UNREACHABLE();
	}
}
