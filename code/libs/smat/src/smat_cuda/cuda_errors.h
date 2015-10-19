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
#ifndef __SM_CUDA_ERRORS_H__
#define __SM_CUDA_ERRORS_H__

#include <smat_cuda/config.h>
#include <base/assert.h>
#include <base/util.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

// ccu(XXX,...) = checked call to cudaXXX(...)
// ccb(XXX,...) = checked call to cublasXXX(...)
// ccr(XXX,...) = checked call to curandXXX(...)
// cce() = check cudaGetLastError and, if there is an error, report it
#define ccu(func,...) { cudaError_t    error  =   cuda##func(__VA_ARGS__); if (error != cudaSuccess)            SM_ERROR(format("AssertionError: CUDA error in %s: %s.",#func,cudaGetErrorString(error)).c_str()); }
#define ccb(func,...) { cublasStatus_t status = cublas##func(__VA_ARGS__); if (status != CUBLAS_STATUS_SUCCESS) SM_ERROR(format("AssertionError: CUBLAS error in %s: %s.",#func,get_cublas_err_str(status)).c_str()); }
#define ccr(func,...) { curandStatus_t status = curand##func(__VA_ARGS__); if (status != CURAND_STATUS_SUCCESS) SM_ERROR(format("AssertionError: CURAND failed in %s: %s.",#func,get_curand_err_str(status)).c_str()); }
#define cce()         { cudaError_t    error  = cudaGetLastError();        if (error != cudaSuccess)            SM_ERROR(format("AssertionError: CUDA error: %s.",cudaGetErrorString(error)).c_str()); }
#ifdef _DEBUG
#define cce_dbsync()  { cudaDeviceSynchronize(); cce(); }
#else
#define cce_dbsync()  { }
#endif

SM_NAMESPACE_BEGIN

SM_CUDA_EXPORT const char* get_cublas_err_str(cublasStatus_t s);
SM_CUDA_EXPORT const char* get_curand_err_str(curandStatus_t s);

SM_NAMESPACE_END

#endif // __SM_CUDA_ERRORS_H__
