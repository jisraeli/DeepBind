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
using namespace sm;

template <typename T>
__global__ void kernel_dropout_fp_tr(curandState_t* state, const T* X, const T* rate, T* Z, bool* M,
                                     usize_t n, usize_t m, usize_t k)
{
	DECL_KERNEL_VARS
	unsigned tid = bdx*bx + tx;
	curandState local_state = state[tid];
	for (usize_t i = (usize_t)tid; i < n; i += bdx*gdx) {
		T p = rate[(i/k) % m];
		bool mask = (curand_uniform(&local_state) >= p);
		M[i] = mask;
		Z[i] = X[i]*(T)mask;
	}
	state[tid] = local_state;
}

template <typename T>
__global__ void kernel_dropout_fp_tr_match(curandState_t* state, const T* X, const T* rate, T* Z, bool* M,
                                           usize_t nrow, usize_t ncol, usize_t m, usize_t k)
{
	DECL_KERNEL_VARS
	unsigned tid = bdx*bx + tx;
	curandState local_state = state[tid];
	for (usize_t i = (usize_t)tid; i < nrow*ncol/2; i += bdx*gdx) {
		usize_t row = (i / ncol)*2;
		usize_t col = i % ncol;
		T p = rate[col/k];
		bool mask = (curand_uniform(&local_state) >= p);
		usize_t i0 = row*ncol+col;
		usize_t i1 = i0+ncol;
		M[i0] = mask;
		M[i1] = mask;
		Z[i0] = X[i0]*(T)mask;
		Z[i1] = X[i1]*(T)mask;
	}
	state[tid] = local_state;
}

template <typename T>
__global__ void kernel_dropout_fp_te(const T* X, const T* rate, T* Z, usize_t n, usize_t m, usize_t k)
{
	DECL_KERNEL_VARS
	unsigned tid = bdx*bx + tx;
	for (usize_t i = (usize_t)tid; i < n; i += bdx*gdx) {
		T scale = 1-rate[(i/k) % m];
		Z[i] = X[i]*scale;
	}
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

template <typename T>
__global__ void kernel_dropout_bp_te(const T* dZ, const T* rate, T* dX, usize_t n, usize_t m, usize_t k)
{
	DECL_KERNEL_VARS
	unsigned tid = bdx*bx + tx;
	for (usize_t i = (usize_t)tid; i < n; i += bdx*gdx) {
		T scale = 1-rate[(i/k) % m];
		dX[i] = dZ[i]*scale;
	}
}

void launch_dropout_fp_tr(cudaStream_t stream, dtype_t dtype,
                          const void* X, const void* rate, void* Z, bool* M,
                          usize_t n, usize_t m, usize_t k, bool matchrows)
{
	if (matchrows) {
		usize_t ncol = m*k;
		usize_t nrow = n/ncol;
		launchcfg cfg = make_elemwise_launchcfg(n/2);
		if (dtype == f32)
			kernel_dropout_fp_tr_match<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(thread_cudactx().curand_state(),(const float*)X,(const float*)rate,(float*)Z,M,nrow,ncol,m,k);
		else
			kernel_dropout_fp_tr_match<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(thread_cudactx().curand_state(),(const double*)X,(const double*)rate,(double*)Z,M,nrow,ncol,m,k);
	} else {
		launchcfg cfg = make_elemwise_launchcfg(n);
		if (dtype == f32)
			kernel_dropout_fp_tr<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(thread_cudactx().curand_state(),(const float*)X,(const float*)rate,(float*)Z,M,n,m,k);
		else
			kernel_dropout_fp_tr<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(thread_cudactx().curand_state(),(const double*)X,(const double*)rate,(double*)Z,M,n,m,k);
	}
}

void launch_dropout_fp_te(cudaStream_t stream, dtype_t dtype,
                          const void* X, const void* rate, void* Z,
                          usize_t n, usize_t m, usize_t k)
{
	launchcfg cfg = make_elemwise_launchcfg(n);
	if (dtype == f32)
		kernel_dropout_fp_te<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((const float*)X,(const float*)rate,(float*)Z,n,m,k);
	else
		kernel_dropout_fp_te<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((const double*)X,(const double*)rate,(double*)Z,n,m,k);
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

void launch_dropout_bp_te(cudaStream_t stream, dtype_t dtype,
                          const void* dZ, const void* rate, void* dX,
                          usize_t n, usize_t m, usize_t k)
{
	launchcfg cfg = make_elemwise_launchcfg(n);
	if (dtype == f32)
		kernel_dropout_bp_te<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((const float*)dZ,(const float*)rate,(float*)dX,n,m,k);
	else
		kernel_dropout_bp_te<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((const double*)dZ,(const double*)rate,(double*)dX,n,m,k);
}
