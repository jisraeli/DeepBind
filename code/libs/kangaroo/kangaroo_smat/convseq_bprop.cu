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
#include "convseq_bprop.cuh"

template <unsigned bdx,  // threads per block in "filters" dimension
          unsigned bdy,  // threads per block in "samples" dimension
          unsigned spt,  // samples per thread
          unsigned fpt,  // filters per thread
          typename float_t, usize_t filter_size, usize_t nchannel>
void launch_convseq_bprop_final(cudaStream_t stream, 
                                const uint8_t*  samples,  usize_t nsample, 
                                      float_t*  filters,  usize_t nfilter,
                                const uindex_t* segments, usize_t nsegment,
                                const float_t*  deltamaps)
{
	if (nsample == 0 || nfilter == 0)
		return;
	dim3 bdim(bdx,bdy);
	dim3 gdim(divup(nfilter,bdx*fpt),divup(nsample,bdy*spt));
	convseq_bprop_kernel<bdx,bdy,spt,fpt,filter_size,nchannel><<<gdim,bdim,0,stream>>>(samples,nsample,filters,nfilter,deltamaps);
	if (segments && nsegment > 0) {
		const unsigned spb = 16; // segments per block
		gdim = dim3(divup(nfilter,bdx*fpt),divup(nsegment,spb));
		convseq_bprop_kernel_applysegs<bdx,filter_size,spb,fpt,nchannel><<<gdim,bdim,0,stream>>>(samples,nsample,filters,nfilter,segments,nsegment,deltamaps);
	}
}

template <typename float_t>
void launch_convseq_bprop_dtype(cudaStream_t stream, 
                                const uint8_t*  samples,  usize_t nsample, usize_t nchannel,
                                      float_t*  filters,  usize_t nfilter, usize_t filter_size,
                                const uindex_t* segments, usize_t nsegment,
                                const float_t*  deltamaps) { }


template <>
void launch_convseq_bprop_dtype<float>(cudaStream_t stream, 
                                const uint8_t* samples,  usize_t nsample, usize_t nchannel,
                                      float* filters,  usize_t nfilter, usize_t filter_size,
                                const uindex_t* segments, usize_t nsegment,
                                const float* deltamaps)
{
	switch (nchannel) {
	case 4:
		switch (filter_size) {
		case  1: launch_convseq_bprop_final<8,16,8,4,float, 1,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case  2: launch_convseq_bprop_final<8,16,8,4,float, 2,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case  8: launch_convseq_bprop_final<8,16,8,4,float, 8,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case 12: launch_convseq_bprop_final<8,16,8,4,float,12,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case 15: {
			if (nfilter <= 16) {
				if (nsample >= 5000)
					launch_convseq_bprop_final<8,16,8,4,float,15,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); 
				else
					launch_convseq_bprop_final<32,16,8,4,float,15,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); 
			} else if (nfilter <= 32) {
				launch_convseq_bprop_final<32,16,8,4,float,15,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); 
			} else {
				if (nsample >= 5000)
					launch_convseq_bprop_final<64,16,8,2,float,15,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); 
				else
					launch_convseq_bprop_final<64,8,8,2,float,15,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); 
			}
		} break;
		default: SM_ERROR("NotImplementedError: The given filter_size does not have a matching pre-compiled kernel.\n");
		}
		break;
	// TODO: can do 80 channel for proteins
	default: SM_ERROR("NotImplementedError: The given number of channels does not have a matching pre-compiled kernel.\n");
	}
}

template <>
void launch_convseq_bprop_dtype<double>(cudaStream_t stream, 
                                const uint8_t*  samples,  usize_t nsample, usize_t nchannel,
                                      double*   filters,  usize_t nfilter, usize_t filter_size,
                                const uindex_t* segments, usize_t nsegment,
                                const double*   deltamaps)
{
	switch (nchannel) {
	case 4:
		switch (filter_size) {
			/*
		case  1: launch_convseq_bprop_final<8,16,8,4,double, 1,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case  2: launch_convseq_bprop_final<8,16,8,4,double, 2,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case  8: launch_convseq_bprop_final<8,16,8,4,double, 8,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case 15: {
			if (nsample >= 5000)
				launch_convseq_bprop_final<8,16,8,4,double,15,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); 
			else
				launch_convseq_bprop_final<8,16,8,4,double,15,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); 
		} break;
		*/
		case  1: launch_convseq_bprop_final<1,8,8,1,double, 1,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case  2: launch_convseq_bprop_final<1,8,8,1,double, 2,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case  8: launch_convseq_bprop_final<1,8,8,1,double, 8,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case 12: launch_convseq_bprop_final<1,8,8,1,double,12,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;
		case 15: launch_convseq_bprop_final<1,8,8,1,double,15,4>(stream,samples,nsample,filters,nfilter,segments,nsegment,deltamaps); break;

		default: SM_ERROR("NotImplementedError: The given filter_size does not have a matching pre-compiled kernel.\n");
		}
		break;
	// TODO: can do 80 channel for proteins
	default: SM_ERROR("NotImplementedError: The given number of channels does not have a matching pre-compiled kernel.\n");
	}
}


void launch_convseq_bprop(cudaStream_t stream, dtype_t dtype,
                          const uint8_t*  samples,  usize_t nsample, usize_t nchannel,
                                void*     filters,  usize_t nfilter, usize_t filter_size,
                          const uindex_t* segments, usize_t nsegment,
                          const void*     deltamaps)
{
	switch (dtype) {
	case f32: launch_convseq_bprop_dtype<float >(stream,samples,nsample,nchannel,(float* )filters,nfilter,filter_size,segments,nsegment,(const float* )deltamaps); break;
	case f64: launch_convseq_bprop_dtype<double>(stream,samples,nsample,nchannel,(double*)filters,nfilter,filter_size,segments,nsegment,(const double*)deltamaps); break;
	default: SM_UNREACHABLE();
	}
}
