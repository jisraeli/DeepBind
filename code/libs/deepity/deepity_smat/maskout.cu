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
__global__ void kernel_maskout(const bool* M, T* A, usize_t n)
{
	DECL_KERNEL_VARS
	unsigned tid = bdx*bx + tx;
	for (usize_t i = (usize_t)tid; i < n; i += bdx*gdx) {
		if (!M[i])
			A[i] = (T)0;
	}
}

template <typename T>
__global__ void kernel_calc_zmask(const T* Z, bool* M, usize_t n, usize_t m)
{
	DECL_KERNEL_VARS
	unsigned tid = bdx*bx + tx;
	for (usize_t i = (usize_t)tid; i < (n*m)/2; i += bdx*gdx) {
		usize_t row = (i/m)*2;
		usize_t col = i%m;
		usize_t index = m*row+col;
		if (Z[index] >= Z[index+m]) {
			M[index] = true;
		} else {
			M[index+m] = true;
		}
	}
}

void launch_maskout(cudaStream_t stream, dtype_t dtype, const bool* M, void* A, usize_t n)
{
	launchcfg cfg = make_elemwise_launchcfg(n);
	if (dtype == f32)
		kernel_maskout<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(M,(float*)A,n);
	else
		kernel_maskout<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(M,(double*)A,n);
}

void launch_calc_zmask(cudaStream_t stream, dtype_t dtype, const void* Z, bool* M, usize_t n, usize_t m)
{
	launchcfg cfg = make_elemwise_launchcfg((n*m)/2);
	if (dtype == f32)
		kernel_calc_zmask<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((float* )Z,M,n,m);
	else
		kernel_calc_zmask<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((double*)Z,M,n,m);
}
