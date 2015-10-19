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
#ifndef __SM_CUDA_LAUNCH_UTIL_H__
#define __SM_CUDA_LAUNCH_UTIL_H__

#include <smat_cuda/config.h>
#include <smat/dtypes.h>
#include <cuda_runtime.h>

#ifdef __CUDACC__
// Declare shorthand versions of the standard cuda thread-local constants
#define DECL_KERNEL_VARS          \
	const unsigned& tx  = threadIdx.x; ((void)tx); \
	const unsigned& ty  = threadIdx.y; ((void)ty); \
    const unsigned& bx  = blockIdx.x;  ((void)bx); \
    const unsigned& by  = blockIdx.y;  ((void)by); \
    const unsigned& bdx  = blockDim.x;  ((void)bdx); \
    const unsigned& bdy  = blockDim.y;  ((void)bdy); \
    const unsigned& gdx  = gridDim.x;  ((void)gdx); \
    const unsigned& gdy  = gridDim.y;  ((void)gdy);
#else
// These variable declarations are provided so that Visual Studio source editor doesn't try to highlight intellisense errors
#define DECL_KERNEL_VARS    \
	const unsigned tx  = 1; ((void)tx); \
	const unsigned ty  = 1; ((void)ty); \
    const unsigned bx  = 1; ((void)bx); \
    const unsigned by  = 1; ((void)by); \
    const unsigned bdx = 1; ((void)bdx); \
    const unsigned bdy = 1; ((void)bdy); \
    const unsigned gdx = 1; ((void)gdx); \
    const unsigned gdy = 1; ((void)gdy);
#endif

#ifdef __CUDACC__
SM_DEVICE_INLINE double atomicAdd(double* address, double val)
{
	unsigned long long* address_as_ull = (unsigned long long*)address;
	unsigned long long old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
		__double_as_longlong(val +__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

SM_NAMESPACE_BEGIN

// KERNEL CONFIGURATION MACROS
struct launchcfg {
	launchcfg(const dim3& gdim, const dim3& bdim, unsigned smem, cudaStream_t stream): gdim(gdim), bdim(bdim), smem(smem), stream(stream) { }
	dim3 gdim;
	dim3 bdim;
	unsigned smem;
	cudaStream_t stream;
};

#define SM_CUDA_LAUNCH(kernel,cfg) kernel<<<(cfg).gdim,(cfg).bdim,(cfg).smem,(cfg).stream>>>

SM_CUDA_EXPORT launchcfg make_elemwise_launchcfg(usize_t size); // Determines good launch config for basic linear kernels.

SM_NAMESPACE_END

#endif // __SM_CUDA_LAUNCH_UTIL_H__
