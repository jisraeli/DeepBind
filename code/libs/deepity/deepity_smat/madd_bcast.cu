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
__global__ void kernel_madd_bcast(const T* A, const T* b, T* dst, usize_t n, usize_t m, usize_t k)
{
	DECL_KERNEL_VARS
	for (usize_t i = (usize_t)bdx*bx+tx; i < n; i += bdx*gdx)
		dst[i] += A[i]*b[(i/k) % m];
}

void launch_madd_bcast(cudaStream_t stream, dtype_t dtype,
                        const void* A, const void* b, void* dst,
                        usize_t n, usize_t m, usize_t k)
{
	launchcfg cfg = make_elemwise_launchcfg(n);
	if (dtype == f32)
		kernel_madd_bcast<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((const float*)A,(const float*)b,(float*)dst,n,m,k);
	else
		kernel_madd_bcast<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>((const double*)A,(const double*)b,(double*)dst,n,m,k);
}
