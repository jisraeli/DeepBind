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
#ifndef __SM_CUDA_CONTEXT_H__
#define __SM_CUDA_CONTEXT_H__

#include <smat/vm/context.h>
#include <smat_cuda/config.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

SM_NAMESPACE_BEGIN

class SM_CUDA_EXPORT cuda_context: public context {
public:
	cuda_context();
	virtual ~cuda_context();

	virtual void set_device(int device);
	virtual void set_randseed(size_t seed);         // override
	virtual void set_options(const optionset& opt); // override
	virtual bool is_supported(dtype_t dt) const;    // override
	virtual void sync();      // override
	virtual void autotune();  // override

	int                   device() const;
	cudaStream_t          stream() const;
	cublasHandle_t        cublas() const;
	curandGenerator_t     curand() const;
	curandState*          curand_state() const;
	const cudaDeviceProp& deviceprop() const;

private:
	virtual void ensure_initialized() const; // override
	void set_curand_seed() const;

	int                       _device;
	bool                      _want_stream;
	mutable int               _curr_device;
	mutable cudaDeviceProp    _deviceprop;
	mutable cudaStream_t      _stream;
	mutable cublasHandle_t    _cublas;
	mutable curandGenerator_t _curand;
	mutable curandState*      _curand_state;
	friend class cuda_block_allocator;
};

SM_INLINE cuda_context& thread_cudactx() { return (cuda_context&)thread_ctx(); } // casts to cuda_context, for convenience

SM_NAMESPACE_END

#endif // __SM_CUDA_CONTEXT_H__
