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
#include <smat_cuda/launch_util.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN


template <typename T>
__global__ void kernel_dropout_fp_tr(curandState_t* state, const T* X, T* Z, bool* M, T rate, usize_t n)
{
	DECL_KERNEL_VARS
	unsigned tid = bdx*bx + tx;
	curandState local_state = state[tid];
	for (usize_t i = (usize_t)tid; i < n; i += bdx*gdx) {
		bool mask = (curand_uniform(&local_state) >= rate);
		M[i] = mask;
		Z[i] = X[i]*(T)mask;
	}
	state[tid] = local_state;
}

template <typename T>
__global__ void kernel_dropout_bp_tr(const T* dZ, const bool* M, T* dX, usize_t n)
{
	DECL_KERNEL_VARS
	unsigned tid = bdx*bx + tx;
	for (usize_t i = (usize_t)tid; i < n; i += bdx*gdx) {
		dX[i] = dZ[i]*(T)M[i];
	}
}

void launch_dropout_fp_tr(cudaStream_t stream, dtype_t dtype,
                          const void* X, double rate, void* Z, bool* M,
                          usize_t n)
{
	launchcfg cfg = make_elemwise_launchcfg(n);
	if (dtype == f32)
		kernel_dropout_fp_tr<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(thread_cudactx().curand_state(),(const float*)X,(float*)Z,M,(float)rate,n);
	else
		kernel_dropout_fp_tr<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(thread_cudactx().curand_state(),(const double*)X,(double*)Z,M,(double)rate,n);
}

void launch_dropout_bp_tr(cudaStream_t stream, dtype_t dtype,
                       const void* dZ, const bool* M, void* dX, usize_t n)
{
	launchcfg cfg = make_elemwise_launchcfg(n);
	if (dtype == f32)
		kernel_dropout_bp_tr<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((const float*)dZ,M,(float*)dX,n);
	else
		kernel_dropout_bp_tr<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((const double*)dZ,M,(double*)dX,n);
}


void validate_dropout_fp_tr(opcode_t opcode, const argument& X, const argument& rate,
                                             const argument& Z, const argument& M)
{
	double _rate = rate.get<double>();
	SM_ASSERT(_rate >= 0.0 && _rate <= 1.0);
	SM_ASSERT(X.dtype == Z.dtype);
	SM_ASSERT(X.shape == Z.shape);
	SM_ASSERT(X.shape == M.shape);
}

void execute_dropout_fp_tr(opcode_t opcode, const argument& X, const argument& rate,
                                            const argument& Z, const argument& M)
{
	launch_dropout_fp_tr(thread_cudactx().stream(), X.dtype,
	                     X.get<const void*>(), rate.get<double>(), Z.get<void*>(), M.get<bool*>(),
	                     X.size());
}


void validate_dropout_bp_tr(opcode_t opcode, const argument& dZ, const argument& M, const argument& dX)
{
	SM_ASSERT(dX.dtype == dZ.dtype);
	SM_ASSERT(dX.size() == M.size());
	SM_ASSERT(dX.size() == dZ.size());
}

void execute_dropout_bp_tr(opcode_t opcode, const argument& dZ, const argument& M, const argument& dX)
{
	launch_dropout_bp_tr(thread_cudactx().stream(),dZ.dtype, dZ.get<const void*>(), M.get<const bool*>(), dX.get<void*>(), dX.size());
}

SM_NAMESPACE_END
