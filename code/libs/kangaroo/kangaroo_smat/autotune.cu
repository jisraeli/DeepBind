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
#include <smat/vm/util/autotune.h>
#include <smat_cuda/cuda_context.h>

const int nchannel = 4;

size_t convseq_bprop_setup_args(const autotune_query& query, argument* arg)
{
	isize_t nsample = query.q0;
	isize_t nfilter = query.q1;
	isize_t fsize   = query.q2;
	argument& samples   = arg[0];
	argument& filters   = arg[1];
	argument& deltamaps = arg[2];
	samples.shape.x = nsample;
	samples.dtype = u8;
	filters.shape.x = nchannel*fsize;
	filters.shape.y = nfilter;
	deltamaps.shape.x = nsample;
	deltamaps.shape.y = nfilter;
	return nsample*nfilter*fsize; // flop count
}


struct convseq_bprop_functor {

	template <unsigned bdx, unsigned bdy, unsigned spt, unsigned fpt>
	struct launch {
		static void execute(opcode_t opcode, const argument& samples,
                                             const argument& filters,
                                             const argument& deltamaps)
		{
			usize_t nsample = samples.shape.x;
			usize_t nfilter = filters.shape.y;
			usize_t filter_size = filters.shape.x/nchannel;
			dim3 bdim(bdx,bdy);
			dim3 gdim(divup(nfilter,bdx*fpt),divup(nsample,bdy*spt));
			switch (filter_size) {
			case  8: convseq_bprop_kernel<bdx,bdy,spt,fpt, 8,nchannel><<<gdim,bdim,0,thread_cudactx().stream()>>>(samples.get<const uint8_t*>(),nsample,filters.get<float*>(),nfilter,deltamaps.get<const float*>()); break;
			case 15: convseq_bprop_kernel<bdx,bdy,spt,fpt,15,nchannel><<<gdim,bdim,0,thread_cudactx().stream()>>>(samples.get<const uint8_t*>(),nsample,filters.get<float*>(),nfilter,deltamaps.get<const float*>()); break;
			default: SM_UNREACHABLE();
			}
		}

		static bool validate(opcode_t opcode, const argument& src, const argument& dst) { return true; }
	};

};


extern opcode_t oc_convseq;
extern opcode_t oc_convseq_bp;
extern opcode_t oc_poolrgn;
extern opcode_t oc_poolrgn_bp;


extern "C" SM_DLLEXPORT void api_autotune()
{
	typedef make_typelist<     // <bdx,bdy,spt,fpt>
		make_intlist4< 8, 16,8,4>::type,
		make_intlist4< 8, 16,8,8>::type,
		make_intlist4<16,  8,8,8>::type,
		make_intlist4<16, 16,8,8>::type,
		make_intlist4<32,  8,8,4>::type,
		make_intlist4<32, 16,8,4>::type,
		make_intlist4<64,  8,8,2>::type,
		make_intlist4<64, 16,8,2>::type
	>::type psets;

	autotune_queries queries;
	for (size_t i = 200; i <= 50000; i *= 2) // sequence length
		for (size_t j = 24; j <= 24; j *= 2)   // number of filters
			queries.push_back(autotune_query((index_t)i,(index_t)j,15));

	autotuner tuner(oc_convseq_bp);
	tuner.sample<convseq_bprop_functor::launch,psets>(queries,convseq_bprop_setup_args);
	tuner.print_all();
	tuner.print_best();
}

