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

SM_NAMESPACE_BEGIN

const char* get_cublas_err_str(cublasStatus_t s)
{
	switch (s) {
	case CUBLAS_STATUS_SUCCESS:          return "SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:  return "NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:     return "ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:    return "INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:    return "ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:    return "MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:   return "INTERNAL_ERROR";
	}
	return "UNKNOWN";
}

const char* get_curand_err_str(curandStatus_t s)
{
	switch (s) {
	case CURAND_STATUS_SUCCESS:                   return "SUCCESS";
	case CURAND_STATUS_VERSION_MISMATCH:          return "VERSION_MISMATCH";
	case CUBLAS_STATUS_NOT_INITIALIZED:           return "NOT_INITIALIZED";
	case CURAND_STATUS_ALLOCATION_FAILED:         return "ALLOCATION_FAILED";
	case CURAND_STATUS_TYPE_ERROR:                return "TYPE_ERROR";
	case CURAND_STATUS_OUT_OF_RANGE:              return "OUT_OF_RANGE";
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:       return "LENGTH_NOT_MULTIPLE";
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "DOUBLE_PRECISION_REQUIRED";
	case CURAND_STATUS_LAUNCH_FAILURE:            return "LAUNCH_FAILURE";
	case CURAND_STATUS_PREEXISTING_FAILURE:       return "PREEXISTING_FAILURE";
	case CURAND_STATUS_INITIALIZATION_FAILED:     return "INITIALIZATION_FAILED";
	case CURAND_STATUS_ARCH_MISMATCH:             return "ARCH_MISMATCH";
	case CURAND_STATUS_INTERNAL_ERROR:            return "INTERNAL_ERROR";
	}
	return "UNKNOWN";
}

SM_NAMESPACE_END
