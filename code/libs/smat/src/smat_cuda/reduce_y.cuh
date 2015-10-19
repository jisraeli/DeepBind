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
#ifndef __SM_SMAT_CUDA_REDUCE_Y_H__
#define __SM_SMAT_CUDA_REDUCE_Y_H__

#include <smat_cuda/reducers.cuh>
#include <smat_cuda/launch_util.h>
#include <smat_cuda/cuda_context.h>
#include <smat_cuda/reduce_y.autotune.cuh>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN

/////////////////////////////////////////////////////////////////////
// KERNEL 0: Each column is reduced by gdy threads (1 thread per block).
//           If nx >= bdx this is efficient, otherwise gdx=1 and there 
//           will be idle threads "hanging off" the edge of each block.
/////////////////////////////////////////////////////////////////////

template <unsigned bdx,   // blockdim.x
          unsigned wdx,   // workdim.x
          unsigned wdy,   // workdim.y
          bool check_ny,  // does a block need to worry about falling off the bottom of the src array?
          typename reducer, typename S, typename D>
__global__ void kernel_reduce_y_0(const S* src, D* dst, unsigned nx, unsigned ny, unsigned n)
{
	unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned by = blockIdx.y;
	unsigned j  = wdx*bdx*bx+tx;  // j   = index of current src column being reduced
	unsigned top= wdy*bdx*by;     // top = index of top row that this block is responsible for
	src += top*nx + j;            // move src so that it points to the first element this block is responsible for
	dst += by*nx;                 // move dst so that it points to the row in the output that this block should write to
	for (unsigned wx = 0; wx < wdx; ++wx, j += bdx, src += bdx) {
		if (j >= nx)
			return;
		reducer reduce;
		#pragma unroll
		for (unsigned i = 0; i < bdx*wdy; ++i) {
			if (check_ny && top + i >= ny)
				break;
			reduce.element(src[i*nx]);  // reduce item at column j and row top+i
		}
		reduce.finalize(n);
		dst[j] = reduce.result;   // done this column, so store the result and move on to the next
	}
}

//////////////////////////////////////////////////
// KERNEL 1: When nx <= bdx/2, then it's more efficient
//           to have each block assign multiple threads
//           per column.
//////////////////////////////////////////////////

template <unsigned bdx,   // blockdim.x
          unsigned wdy,   // workdim.y
          bool check_ny,  // does a block need to worry about falling off the bottom of the src array?
          typename reducer, typename S, typename D>
__global__ void kernel_reduce_y_1(const S* src, D* dst, unsigned nx, unsigned ny, unsigned n, unsigned nrow)
{
	extern __shared__ char smem_raw[]; // requires nx*nrow elements of type D
	D* smem = (D*)smem_raw;
	unsigned tx = threadIdx.x;
    unsigned bx = blockIdx.x;
    unsigned by = blockIdx.y;
	unsigned trow = tx/nx;
	unsigned j    = bdx*bx + tx%nx;      // j   = index of current src column being reduced
	if (trow >= nrow)
		return;
	unsigned top= wdy*bdx*by; // top = index of top row that this block is responsible for
	src += top*nx + j;             // move src so that it points to the first element this block is responsible for
	dst += by*nx;                  // move dst so that it points to the row in the output that this block should write to

	// Reduce along the column, which this thread responsible for reducing every 'nrow'-th item
	reducer reduce;
	#pragma unroll
	for (unsigned i = trow; i < bdx*wdy; i += nrow) {
		if (check_ny && top + i >= ny)
			break;
		reduce.element(src[i*nx]);  // reduce item at column j and row top+i
	}

	// store this thread's partial result in the block's shared memory
	smem[tx] = reduce.result;
	__syncthreads();
	if (trow == 0) {
		// Threads in the first row within the block are responsible for collecting
		// the partial results from shared memory, and writing the final result to global mem.
		for (int i = 1; i < nrow; ++i)
			reduce.partial(smem[i*nx+tx]);
		reduce.finalize(n);
		dst[j] = reduce.result;   // done this column, so store the result and move on to the next
	}
}

///////////////////////////////////////////////////

const int reduce_y_partial_bdx = 128; // used when matrix has been substantially squished in the y dimension; use a kernel launch config that's reasonable for fat short matrices

SM_CUDA_EXPORT void reduce_y_launch(opcode_t opcode, const argument& src, const argument& dst, 
                                    unsigned bdx, unsigned wdx, unsigned wdy,
                                    void (*reduce )(unsigned gdx, unsigned gdy, const void*, void*, unsigned nx, unsigned ny, unsigned n),
                                    void (*partial)(unsigned gdx, unsigned gdy, const void*, void*, unsigned nx, unsigned ny, unsigned n));

template <template <typename value_type, typename result_type> class reducer>
struct execute_reduce_y_typed {
	template <typename S, typename D>
	struct general {
		typedef reducer<S,D> reducer_SD;
		static void execute(opcode_t opcode, const argument& src, const argument& dst)
		{
			if (src.shape.x > 0) {
				//static reduce_y_autotune_table<launch> autotune_table;
				//autotune_table(src.shape.x,src.shape.y)(opcode,src,dst);

				// Some rules hand-derived from auto-tuning plot; can auto generate
				// the table once the table can be made aware of each kernel's input
				// constraints; currently, the table might return a kernel that performs
				// well with similar query parameters, but in fact cannot be executed
				// with those exact query parameters.

				//     ...
				//     AAAADCCCCCCCC
				//     AAAADDCCCCCCC
				//     AABBDDDCCCCCC
				// ny  ABBBDDDDCCCCC
				//     BBBBDDDDDDCCC
				//     BBBBDDDDDDCCC
				//     BBBBDDDDDDCCC ...
				//           nx
				//
				isize_t nx = src.shape.x;
				isize_t ny = src.shape.x;
				if (nx <= 128) {
					if (ny >= (1<<18) || (ny >= (1<<15) && nx <= (1<<16))) {
						launch<1,128,4,1>::execute(opcode,src,dst); // A
					} else {
						launch<1,128,1,1>::execute(opcode,src,dst); // B
					}
				} else {
					if (nx >= (1<<20) || nx*ny >= (1<<25)) {
						launch<0,128,4,1>::execute(opcode,src,dst); // C
					} else {
						launch<0,128,1,1>::execute(opcode,src,dst); // D
					}
				}
			}
		}

		template <unsigned kernel, unsigned bdx, unsigned wdx, unsigned wdy> struct launch { };
		template <unsigned bdx, unsigned wdx, unsigned wdy>
		struct launch<0,bdx,wdx,wdy> {
			static void reduce(unsigned gdx, unsigned gdy, const void* src, void* dst, unsigned nx, unsigned ny, unsigned n)
			{
				kernel_reduce_y_0<bdx,wdx,wdy,true,reducer_SD><<<dim3(gdx,gdy),bdx,0,thread_cudactx().stream()>>>((const S*)src,(D*)dst,nx,ny,n);
			}
			static void partial(unsigned gdx, unsigned gdy, const void* src, void* dst, unsigned nx, unsigned ny, unsigned n)
			{
				typedef reducer_partial_results<reducer_SD> reducer_partial;
				kernel_reduce_y_0<reduce_y_partial_bdx,1,1,true,reducer_partial><<<dim3(gdx,gdy),reduce_y_partial_bdx,0,thread_cudactx().stream()>>>((const D*)src,(D*)dst,nx,ny,n);
			}
			static void execute(opcode_t opcode, const argument& src, const argument& dst)
			{
				reduce_y_launch(opcode,src,dst,bdx,wdx,wdy,&reduce,&partial);
			}
			static bool validate(opcode_t opcode, const argument& src, const argument& dst)
			{
				unsigned gdx = divup((unsigned)src.shape.x,bdx*wdx);
				unsigned gdy = divup((unsigned)src.shape.y,bdx*wdy);
				const int* limits = thread_cudactx().deviceprop().maxGridSize;
				return gdx <= (unsigned)limits[0] && gdy <= (unsigned)limits[1];
			}
		};
		template <unsigned bdx, unsigned wdx, unsigned wdy>
		struct launch<1,bdx,wdx,wdy> {
			static void reduce(unsigned gdx, unsigned gdy, const void* src, void* dst, unsigned nx, unsigned ny, unsigned n)
			{
				unsigned nrow = ::max(1,bdx/nx);
				unsigned smem = nx*nrow*sizeof(D);
				SM_ASSERT(smem <= 48*1024); // shared memory limit
				kernel_reduce_y_1<bdx,wdy,true,reducer_SD><<<dim3(gdx,gdy),bdx,smem,thread_cudactx().stream()>>>((const S*)src,(D*)dst,nx,ny,n,nrow);
			}
			static void partial(unsigned gdx, unsigned gdy, const void* src, void* dst, unsigned nx, unsigned ny, unsigned n)
			{
				unsigned nrow = ::max(1,bdx/nx);
				unsigned smem = nx*nrow*sizeof(D);
				SM_ASSERT(smem <= 48*1024); // shared memory limit
				typedef reducer_partial_results<reducer_SD> reducer_partial;
				kernel_reduce_y_1<reduce_y_partial_bdx,1,true,reducer_partial><<<dim3(gdx,gdy),reduce_y_partial_bdx,smem,thread_cudactx().stream()>>>((const D*)src,(D*)dst,nx,ny,n,nrow);
			}
			static void execute(opcode_t opcode, const argument& src, const argument& dst)
			{
				SM_ASSERT(src.shape.x <= bdx); // kernel 1 is only for tall matrices
				reduce_y_launch(opcode,src,dst,bdx,wdx,wdy,&reduce,&partial);
			}
			static bool validate(opcode_t opcode, const argument& src, const argument& dst) { return src.shape.x <= bdx; }
		};
	};

	template <typename T> struct matched  { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,T>::execute(opcode,src,dst); } };
	template <typename T> struct promoted { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,typename ctype2ptype(T)>::execute(opcode,src,dst); } };
	template <typename T> struct asfloat  { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,typename ctype2ftype(T)>::execute(opcode,src,dst); } };
	template <typename T> struct asuindex { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,reduce_index_t>::execute(opcode,src,dst); } };
};


SM_NAMESPACE_END

#endif // __SM_SMAT_CUDA_REDUCE_Y_H__
