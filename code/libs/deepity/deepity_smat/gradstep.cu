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
__global__ void kernel_gradstep(T* P, const T* dP, const T* drate, 
                                            T* mP, const T* mrate, usize_t size)
{
	DECL_KERNEL_VARS
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx) {
		T delta = mP[i]*mrate[i] - dP[i]*drate[i];
		mP[i] = delta;
		P[i] += delta;
	}
}

template <typename T>
__global__ void kernel_gradstep_nesterov1(T* P, const T* mP, const T* mrate, usize_t size)
{
	DECL_KERNEL_VARS
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx) {
		P[i] = P[i] + mrate[i]*mP[i];
	}
}

template <typename T>
__global__ void kernel_gradstep_nesterov2(T* P, const T* dP, const T* drate, 
                                                      T* mP, const T* mrate, usize_t size)
{
	DECL_KERNEL_VARS
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx) {
		T delta = dP[i]*drate[i];
		mP[i] = mP[i]*mrate[i] - delta;
		P[i] -= delta;
	}
}

void launch_gradstep(cudaStream_t stream, dtype_t dtype, isize_t n,
                     void* P, const void* dP, const void* drate,
                                    void* mP, const void* mrate)
{
	launchcfg cfg = make_elemwise_launchcfg(n);
	if (dtype == f32)
		kernel_gradstep<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((float*)P,(const float*)dP,(const float*)drate,(float*)mP,(const float*)mrate,(usize_t)n);
	else
		kernel_gradstep<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((double*)P,(const double*)dP,(const double*)drate,(double*)mP,(const double*)mrate,(usize_t)n);
}

void launch_gradstep_nesterov1(cudaStream_t stream, dtype_t dtype, isize_t n,
                               void* P, const void* mP, const void* mrate)
{
	launchcfg cfg = make_elemwise_launchcfg(n);
	if (dtype == f32)
		kernel_gradstep_nesterov1<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((float*)P,(const float*)mP,(const float*)mrate,(usize_t)n);
	else
		kernel_gradstep_nesterov1<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((double*)P,(const double*)mP,(const double*)mrate,(usize_t)n);
}

void launch_gradstep_nesterov2(cudaStream_t stream, dtype_t dtype, isize_t n,
                               void* P, const void* dP, const void* drate,
                                              void* mP, const void* mrate)
{
	launchcfg cfg = make_elemwise_launchcfg(n);
	if (dtype == f32)
		kernel_gradstep_nesterov2<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((float*)P,(const float*)dP,(const float*)drate,(float*)mP,(const float*)mrate,(usize_t)n);
	else
		kernel_gradstep_nesterov2<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((double*)P,(const double*)dP,(const double*)drate,(double*)mP,(const double*)mrate,(usize_t)n);
}
