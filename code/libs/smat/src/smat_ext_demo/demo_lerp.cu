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
#include <smat/dtypes.h>
using namespace sm;

template <typename T>
__global__ void lerp_kernel(const T* a, const T* b, T* c, T alpha, usize_t size)
{
	usize_t tx  = threadIdx.x, bx  = blockIdx.x;
	usize_t bdx = blockDim.x,  gdx = gridDim.x;

	#pragma unroll
	for (uindex_t i = bdx*bx+tx; i < size; i += bdx*gdx)
		c[i] = (1-alpha)*a[i] + alpha*b[i];
}

void launch_lerp(dim3 gdim, dim3 bdim, unsigned smem, cudaStream_t stream,
                 usize_t size, dtype_t dtype,
                 const void* a, 
                 const void* b,
                       void* c,
                 double alpha)
{
	switch (dtype) {
	case f32: lerp_kernel<<<gdim,bdim,smem,stream>>>((const float* )a,(const float* )b,(float* )c,(float )alpha,size); break;
	case f64: lerp_kernel<<<gdim,bdim,smem,stream>>>((const double*)a,(const double*)b,(double*)c,(double)alpha,size); break;
	}
}
