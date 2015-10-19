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
#include <smat_cuda/cuda_errors.h>
#include <smat_cuda/cuda_context.h>
#include <curand_kernel.h>
#include <smat_cuda/launch_util.h>
#include <smat/vm/instruction_db.h>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>

SM_NAMESPACE_BEGIN

// Kernel to mask out all but the least significant bit of each byte, 
// to convert random 0..255 values into random 0..1 values
__global__ void k_boolfixup(unsigned* dst, usize_t size)
{
	DECL_KERNEL_VARS
	#pragma unroll
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)
		dst[i] &= 0x01010101;  // mask out all bits but the first, four bools at a time.
}

void execute_rand(opcode_t opcode, const argument& dst)
{
	usize_t size = dst.size();
	if (size == 0)
		return;
	curandGenerator_t handle = thread_cudactx().curand();
	if (opcode == oc_rand) {
		// curandGenerate writes 4 bytes at a time, no matter what, so we may have to overwrite the end of an
		// array with number of bytes not divisible by 4.
		// We can do this safely because cuda_machine::alloc ensures there is padding in the allocated range.
		switch (dst.dtype) {
		case b8:  ccr(Generate,handle,dst.get<unsigned*>(),divup(size,4)); break;
		case i8:  ccr(Generate,handle,dst.get<unsigned*>(),divup(size,4)); break;
		case u8:  ccr(Generate,handle,dst.get<unsigned*>(),divup(size,4)); break;
		case i16: ccr(Generate,handle,dst.get<unsigned*>(),divup(size,2)); break;
		case u16: ccr(Generate,handle,dst.get<unsigned*>(),divup(size,2)); break;
		case i32: ccr(Generate,handle,dst.get<unsigned*>(),size); break;
		case u32: ccr(Generate,handle,dst.get<unsigned*>(),size); break;
		case i64: ccr(Generate,handle,dst.get<unsigned*>(),size*2); break;
		case u64: ccr(Generate,handle,dst.get<unsigned*>(),size*2); break;
		case f32: ccr(GenerateUniform,handle,dst.get<float*>(),size); break;
		case f64: ccr(GenerateUniformDouble,handle,dst.get<double*>(),size); break;
		default: SM_UNREACHABLE();
		}
		if (dst.dtype == b8) {
			launchcfg cfg = make_elemwise_launchcfg(size);
			k_boolfixup<<<cfg.gdim,cfg.bdim,0,cfg.stream>>>(dst.get<unsigned*>(),divup(size,4));
		}
	} else if (opcode == oc_randn) {
		// Round 'size' up to the nearest even value, because GenerateNormal requires it.
		// We can do this safely because cuda_machine::alloc ensures there is padding in the allocated range.
		switch (dst.dtype) {
		case f32: ccr(GenerateNormal      ,handle,dst.get<float* >(),rndup(size,2),0.0f,1.0f); break;
		case f64: ccr(GenerateNormalDouble,handle,dst.get<double*>(),rndup(size,2),0.0 ,1.0 ); break;
		default: SM_UNREACHABLE();
		}
	} else {
		SM_ERROR(format("AssertionError: Instruction '%s' has unrecognized argument configuration.\n",get_instruction_info(opcode).mnemonic).c_str());
	}
}

/////////////////////////////////////////////////////////////////////////


template <typename T>
__global__ void kernel_bernoulli(curandState_t* state, float p, T* dst, usize_t size)
{
	DECL_KERNEL_VARS
	unsigned tid = bdx*bx + tx;
	curandState local_state = state[tid];
	for (usize_t i = (usize_t)tid; i < size; i += bdx*gdx)
		dst[i] = (p >= curand_uniform(&local_state)) ? 1 : 0;
	state[tid] = local_state;
}

template <typename T>
struct execute_bernoulli_typed {
	static void execute(opcode_t opcode, const argument& p, const argument& dst)
	{
		usize_t size = (usize_t)dst.size();
		if (size > 0) {
			launchcfg cfg = make_elemwise_launchcfg(size);
			kernel_bernoulli<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(thread_cudactx().curand_state(),p.get<float>(),dst.get<T*>(),size);
		}
	}
};

// launches type-specific bernoulli kernel
void execute_bernoulli(opcode_t opcode, const argument& p, const argument& dst)
{
	DECL_SPECIALIZATION_TABLE(T_G,execute_fn2,execute_bernoulli_typed);
	specialization_table(dst.dtype)(opcode,p,dst);
}

__global__ void kernel_curand_init(curandState *state, int seed)
{
	// Each possible thread uses same seed, but different sequence number 
	// (as suggested by CURAND docs)
	int global_id = blockDim.x*blockIdx.x + threadIdx.x;
	curand_init(seed,global_id,0,&state[global_id]);
}

void execute_curand_init(cudaStream_t stream, curandState* state, int seed, unsigned gdim, unsigned bdim)
{
	kernel_curand_init<<<gdim,bdim,0,stream>>>(state,seed);
}

SM_NAMESPACE_END
