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
#ifndef __SM_SMAT_CUDA_REDUCE_X_H__
#define __SM_SMAT_CUDA_REDUCE_X_H__

#include <smat_cuda/reducers.cuh>
#include <smat_cuda/launch_util.h>
#include <smat_cuda/cuda_context.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN


#define MAX_REDUCE_R_SHORT_SHARED_MEM (6*1024)
#define MAX_REDUCE_R_SHORT_THREADS 256
#define MIN_REDUCE_R_SHORT_THREADS 128

// reduce_x_single_elem:
//   Called when the input is a column matrix, and so we just copy to the output, possibly with promoted type.
template <typename reducer, typename S, typename D>
__global__ void kernel_reduce_x_single_elem(const S* src, D* dst, usize_t n, usize_t m)
{
	DECL_KERNEL_VARS;
	#pragma unroll
	for (usize_t i = (usize_t)bdx*bx+tx; i < n; i += bdx*gdx) {
		reducer r;
		if (m > 0)
			r.element(src[i]);
		dst[i] = r.result;
	}
}

// reduce_x_short:
//   Reduce operation on n x m matrix with "short" rows, 
//   i.e. m is significantly smaller than the number of
//   threads that will be operating.
//   
// TODO: assert m >= bdx
template <int k, typename reducer, typename S, typename D>
__global__ void kernel_reduce_x_short(const S* src, D* dst, usize_t n, usize_t m)
{
	DECL_KERNEL_VARS;
	extern __shared__ char _smem[];  // size determined at kernel launch
	D* accum = (D*)(_smem+0);
	S* tile = (S*)(_smem+bdy*k*sizeof(D));
	int tid = k*ty+tx;
	
	// PHASE 1: Load tile memory as if it was declared as size tile[bdy][m+1].
	#pragma unroll
	for (int i = tid; i < bdy*m && by*bdy*m+i < n*m; i += k*bdy) // check bottom edge of matrix
		tile[i+i/m] = src[by*bdy*m+i];
	__syncthreads();

	// PHASE 2: Walk along the row, reducing every kth item.
	int i0 = (m+1)*ty;
	reducer reduce;
	#pragma unroll
	for (int i = tx; i < m; i += k)
		reduce.element(tile[i0+i]);
	accum[k*ty+tx] = reduce.result;  // write back to shared memory
	__syncthreads();
	
	// PHASE 3: First column of threads computes final reduction.
	if (tx == 0 && by*bdy+ty < n) {
		#pragma unroll
		for (int i = 1; i < k; i++)
			reduce.partial(accum[k*ty+i]);
		reduce.finalize(m);
		dst[by*bdy+ty] = reduce.result;
	}
}

// k = block dimension x  (number of thread columns per block)
// tdx = tile dimension x (NOT same as block dimension x)
// tdy = tile dimension y (same as block dimension y)
// x0 = initial value for reduce
// TODO: this will have horrible bank conflicts and other performance problems for types smaller than 4 bytes
template <int k, int tdx, int tdy, typename reducer, typename S, typename D>
__global__ void kernel_reduce_x(const S* src, D* dst, size_t n, size_t m)
{
	DECL_KERNEL_VARS;
	__shared__ S tile[tdy][tdx+1];
	__shared__ D accum[tdy][k];
	reducer reduce;
	int tid = k*ty+tx;

	// OUTER LOOP: Process several rows by loading them one tile at src time.
	//             Exclude last iteration where we have to check for boundary of matrix.
	isize_t j;
	for (j = 0; j+tdx <= m; j += tdx) {
		__syncthreads();

		// PHASE 1: Load tile memory 
		#pragma unroll
		for (int i = 0; i < tdy && by*tdy+i < n; ++i)  // check bottom edge of matrix
			tile[i][tid] = src[(by*tdy+i)*m+j+tid];
		__syncthreads();
	
		// PHASE 2: Walk along the tile rows, reducing every kth item.
		#pragma unroll
		for (int i = tx; i < tdx; i += k)
			reduce.element(tile[ty][i]);
	}
	__syncthreads();

	// Repeat the inner loop, but with checks to handle the bottom boundary of the matrix

	// PHASE 1: Load tile memory 
	if (j+tid < m) {
		#pragma unroll
		for (int i = 0; i < tdy && by*tdy+i < n; ++i) // check bottom edge of matrix
			tile[i][tid] = src[(by*tdy+i)*m+j+tid];
	}
	__syncthreads();
	
	// PHASE 2: Walk along the tile rows, reducing every kth item.
	#pragma unroll
	for (int i = tx; i < m-j; i += k)
		reduce.element(tile[ty][i]);

	accum[ty][tx] = reduce.result;  // write back to first columns of shared memory
	__syncthreads();

	// PHASE 3: First column of threads computes final reduction and stores it in output
	if (tx == 0 && by*tdy+ty < n) { // check bottom edge of matrix
		#pragma unroll
		for (int i = 1; i < k; i++)
			reduce.partial(accum[ty][i]);
		reduce.finalize(m);
		dst[by*bdy+ty] = reduce.result;
	}
}

template <template <typename value_type, typename result_type> class reducer>
struct execute_reduce_x_typed {
	template <typename S, typename D>
	struct general {
		static void execute(opcode_t opcode, const argument& src, const argument& dst)
		{
			typedef reducer<S,D> reducer_SD;
			isize_t  n = src.shape.y;
			isize_t  m = src.shape.x;
			if (n == 0)
				return;
			if (m <= 1) {
				// Handle degenerate case of reducing along rows of width 1 (i.e. input is a column vector)
				launchcfg cfg = make_elemwise_launchcfg(n);
				kernel_reduce_x_single_elem<reducer_SD><<<cfg.gdim,cfg.gdim,cfg.smem,cfg.stream>>>(src.get<const S*>(),dst.get<D*>(),n,m);
			} else {
				size_t src_dtsize = dtype_size(src.dtype); // input dtype size
				size_t dst_dtsize = dtype_size(dst.dtype); // dstput dtype size; for example, sum on int8 src is promoted to int32 dstput; nnz uses uint32/uint64 regardless of src type
				const unsigned k = 2; // k columns of threads in each block
				unsigned t = (unsigned)min(MAX_REDUCE_R_SHORT_SHARED_MEM/(dst_dtsize*k + src_dtsize*(m+1)),MAX_REDUCE_R_SHORT_THREADS);
				if (t >= MIN_REDUCE_R_SHORT_THREADS) {
					// CASE 1: Matrix is narrow enough that we should try to reduce with tall blocks, 
					//         so as to have a reasonably lsrce number of threads per block.
					//         Strategy: configure each block to have k columns and t/k rows, for t total threads.
					//                   each block is responsible for t/k rows of the input matrix.
					//         If we made it inside here, that means we can have at least t>=MIN_REDUCE_R_SHORT_THREADS
					//         withdst blowing our shared memory limit.
					SM_ASSERT(m >= k);
					unsigned smem = (unsigned)((t/k)*(dst_dtsize*k + src_dtsize*(m+1))); // __shared__ D accum[t/k][k]; __shared A tile[t/k][m+1]   where T is dtype of srcument
					dim3 bdim(k,t/k);
					dim3 gdim(1,(unsigned)divup(n,t/k));
					kernel_reduce_x_short<k,reducer_SD><<<gdim,bdim,smem,thread_cudactx().stream()>>>(src.get<const S*>(),dst.get<D*>(),n,m);
				} else if (m < 64) {
					// CASE 2: Matrix is not wide enough to do 64x16 blocks, but can do 32x16 blocks with half as many threads.
					const unsigned k = 2; // still two columns
					const unsigned tdx = 32, tdy = 16;
					dim3 bdim(k,tdy);
					dim3 gdim(1,(unsigned)divup(n,tdy));
					kernel_reduce_x<k,tdx,tdy,reducer_SD><<<gdim,bdim,0,thread_cudactx().stream()>>>(src.get<const S*>(),dst.get<D*>(),n,m);
				} else {
					// CASE 3: Matrix is wide enough to do 64x16 blocks with 4 columns of threads
					const unsigned k = 4; // still two columns
					const unsigned tdx = 64, tdy = 16;
					dim3 bdim(k,tdy);
					dim3 gdim(1,(unsigned)divup(n,tdy));
					kernel_reduce_x<k,tdx,tdy,reducer_SD><<<gdim,bdim,0,thread_cudactx().stream()>>>(src.get<const S*>(),dst.get<D*>(),n,m);
				}
			}
		}
	};

	template <typename T> struct matched  { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,T>::execute(opcode,src,dst); } };
	template <typename T> struct promoted { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,typename ctype2ptype(T)>::execute(opcode,src,dst); } };
	template <typename T> struct asfloat  { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,typename ctype2ftype(T)>::execute(opcode,src,dst); } };
	template <typename T> struct asuindex { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,reduce_index_t>::execute(opcode,src,dst); } };
};


SM_NAMESPACE_END

#endif // __SM_SMAT_CUDA_REDUCE_X_H__
