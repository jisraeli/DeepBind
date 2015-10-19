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
#include <smat_cuda/reduce.cuh>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>

SM_NAMESPACE_BEGIN

// Determines good launch config for reduce operation.
launchcfg make_reduce_launchcfg(dtype_t dt, usize_t size)
{
	// TODO: add autotuning mechanism
	unsigned maxthread = 128;
	unsigned maxblock  = 2*8*thread_cudactx().deviceprop().multiProcessorCount;
	unsigned nthread   = (size >= maxthread*2) ? maxthread : divup_pow2((size+1)/2);
	unsigned nblock    = min(maxblock,(size + (nthread*2-1))/(nthread*2));
    unsigned smem = nthread*dtype_size(dt);

    // When there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds.
	if (nthread <= 32)
		smem *= 2;

	return launchcfg(nblock,nthread,smem,thread_cudactx().stream());
}

void execute_reduce(opcode_t opcode, const argument& src, const argument& dst)
{
	#define LAUNCH_CASE(typesets,f,matched) \
		if (opcode == oc_##f) { \
			DECL_SPECIALIZATION_TABLE(typesets,execute_fn2,execute_reduce_typed<reducer_##f>::matched); \
			specialization_table(src.dtype)(opcode,src,dst);  \
			return; \
		}

	LAUNCH_CASE(T_G,max,matched)
	LAUNCH_CASE(T_G,min,matched)
	LAUNCH_CASE(T_G,sum,promoted)
	LAUNCH_CASE(T_G,mean,asfloat)
	LAUNCH_CASE(T_G,nnz,asuindex)
	SM_UNIMPLEMENTED()
}

SM_NAMESPACE_END
