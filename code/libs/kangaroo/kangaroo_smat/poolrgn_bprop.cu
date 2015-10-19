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
#include "poolrgn_bprop.cuh"

template <typename float_t>
void launch_poolrgn_bprop_dtype(cudaStream_t stream,
                                      float_t* unpooledmaps, usize_t nfeaturemap,
                                const uindex_t* regions,     usize_t nregion, bool per_region_step,
                                const float_t* pooledmaps,   const uindex_t* pooledmaps_argmax, pooltype_t ptype)
{
	if (ptype == pt_max) {
		launchcfg cfg = make_elemwise_launchcfg(nregion*nfeaturemap);
		poolrgn_bprop_max_kernel<<<cfg.gdim,cfg.bdim,0,stream>>>(unpooledmaps,nfeaturemap*nregion,pooledmaps,pooledmaps_argmax);
	} else {
		const unsigned bdx = 32;  // each thread block handles (32 feature maps x 1 pooling region). (TODO: this is not optimal)
		dim3 bdim(bdx,1);
		dim3 gdim(divup(nfeaturemap,bdx),nregion);
		switch (ptype) {
		case pt_avg: poolrgn_bprop_sum_kernel<bdx,true ><<<gdim,bdim,0,stream>>>(unpooledmaps,nfeaturemap,regions,nregion,per_region_step,pooledmaps); break;
		case pt_sum: poolrgn_bprop_sum_kernel<bdx,false><<<gdim,bdim,0,stream>>>(unpooledmaps,nfeaturemap,regions,nregion,per_region_step,pooledmaps); break;
		case pt_all: poolrgn_bprop_all_kernel<bdx><<<gdim,bdim,0,stream>>>(unpooledmaps,nfeaturemap,regions,nregion,per_region_step,pooledmaps,pooledmaps_argmax); break;
		default: SM_UNREACHABLE();
		}
	}
}


void launch_poolrgn_bprop(cudaStream_t stream, dtype_t dtype,
                                void*    unpooledmaps, usize_t nfeaturemap,
                          const uindex_t* regions,     usize_t nregion, bool per_region_step,
                          const void*    pooledmaps,   const uindex_t* pooledmaps_argmax, pooltype_t ptype)
{
	if (nfeaturemap == 0 || nregion == 0)
		return;
	switch (dtype) {
	case f32: launch_poolrgn_bprop_dtype<float >(stream,(float* )unpooledmaps,nfeaturemap,regions,nregion,per_region_step,(const float* )pooledmaps,pooledmaps_argmax,ptype); break;
	case f64: launch_poolrgn_bprop_dtype<double>(stream,(double*)unpooledmaps,nfeaturemap,regions,nregion,per_region_step,(const double*)pooledmaps,pooledmaps_argmax,ptype); break;
	default: SM_UNREACHABLE();
	}
}
