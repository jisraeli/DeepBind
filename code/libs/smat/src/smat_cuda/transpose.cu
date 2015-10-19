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
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN

const unsigned c_trans_tile_size = 16;

template <typename T> 
__global__ void kernel_trans(const T* src, T* dst, isize_t n, isize_t m) {
	DECL_KERNEL_VARS;
	unsigned i,j;
	__shared__ T tile[c_trans_tile_size][c_trans_tile_size+1];

	// Read the tile into shared memory.
	i = c_trans_tile_size*by + ty;
	j = c_trans_tile_size*bx + tx;
	if(i < n && j < m)
		tile[ty][tx] = src[m*i+j];

	__syncthreads();

	// Write the tile to global memory in transposed order
	i = c_trans_tile_size*bx + ty;
	j = c_trans_tile_size*by + tx;
	if(i < m && j < n)
		dst[n*i+j] = tile[tx][ty];
}

template <typename T>
struct execute_transpose_typed { // TODO: autotune this
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		if (src.size() > 0) {
			dim3 bdim(c_trans_tile_size,c_trans_tile_size);
			dim3 gdim(divup((unsigned)src.shape.x,c_trans_tile_size),
					  divup((unsigned)src.shape.y,c_trans_tile_size));
			kernel_trans<<<gdim,bdim,0,thread_cudactx().stream()>>>(src.get<const T*>(),dst.get<T*>(),src.shape.y,src.shape.x);
		}
	}
};

// Use NVIDIA BLAS extensions to do more highly-tuned transpose for float and double types.
// Use CUBLAS for float or double type.
template <>
struct execute_transpose_typed<float> {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		float alpha = 1, beta = 0;
		ccb(Sgeam,thread_cudactx().cublas(),CUBLAS_OP_T,CUBLAS_OP_T,(int)src.shape.y,(int)src.shape.x,
			&alpha,src.get<const float*>(),(int)src.shape.x,
			&beta ,src.get<const float*>(),(int)src.shape.x,
			dst.get<float*>(),(int)dst.shape.x)
	}
};

template <>
struct execute_transpose_typed<double> {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		double alpha = 1, beta = 0;
		ccb(Dgeam,thread_cudactx().cublas(),CUBLAS_OP_T,CUBLAS_OP_T,(int)src.shape.y,(int)src.shape.x,
			&alpha,src.get<const double*>(),(int)src.shape.x,
			&beta ,src.get<const double*>(),(int)src.shape.x,
			dst.get<double*>(),(int)dst.shape.x)
	}
};

void execute_transpose(opcode_t opcode, const argument& src, const argument& dst)
{
	DECL_SPECIALIZATION_TABLE(T_G,execute_fn2,execute_transpose_typed);
	specialization_table(src.dtype)(opcode,src,dst);
}

SM_NAMESPACE_END
