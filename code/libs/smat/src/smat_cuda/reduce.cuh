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
#ifndef __SM_SMAT_CUDA_REDUCE_H__
#define __SM_SMAT_CUDA_REDUCE_H__

#include <smat_cuda/reducers.cuh>
#include <smat_cuda/launch_util.h>
#include <smat_cuda/cuda_context.h>
#include <smat/vm/instruction_db.h>
#include <smat/vm/heap.h>

SM_NAMESPACE_BEGIN

// Reduce kernel based on "kernel version 6" in the CUDA SDK Samples.
// The kernel need max(64,blocksize)*sizeof(out_t) bytes of shared memory.
template <unsigned blocksize, typename reducer, typename S, typename D>
__global__ void kernel_reduce(const S* src, D* dst, unsigned n, unsigned original_n)
{
	DECL_KERNEL_VARS;
	extern __shared__ char smem_raw[];
	D* smem = (D*)smem_raw;
	reducer reduce;

	// Step 1. Reduce the array from 'n' elements down to gridsize elements.
	// Thread "tx" reduces elements 2*bx+tx and 2*(bx+1)+tx, then
	// increments by 2*bdx*gdx to move on to the next grid of blocks.
	for (unsigned i = bx*blocksize*2 + tx; i < n; i += 2*blocksize*gdx) {
		reduce.element(src[i]);
		if (i + blocksize < n)
			reduce.element(src[i+blocksize]);
	}
	smem[tx] = reduce.result; // Store the reduce result for values visited by this thread
	__syncthreads();

	// Step 2. Reduce from gridsize elements down to gdx 
	//         elements (i.e. one element per block).
	if (blocksize >= 512) { if (tx < 256) reduce.partial(smem[tx+256]); smem[tx] = reduce.result; __syncthreads(); }
	if (blocksize >= 256) { if (tx < 128) reduce.partial(smem[tx+128]); smem[tx] = reduce.result; __syncthreads(); }
	if (blocksize >= 128) { if (tx <  64) reduce.partial(smem[tx+ 64]); smem[tx] = reduce.result; __syncthreads(); }
	if (tx < 32) {
		// All threads with tx < 32 belong to the same warp (TODOL as of 3.5, but what about after?),
		// so we have the opportunity to avoid calling __syncthreads for the remaining 
		// steps (i.e. "warp-synchronous programming"). For that to work, we need to 
		// declare our shared memory volatile so that the compiler knows not to 
		// reorder stores!
		volatile D* vsmem = smem;
		if (blocksize >=  64) { reduce.partial(vsmem[tx+32]); vsmem[tx] = reduce.result; }
		if (blocksize >=  32) { reduce.partial(vsmem[tx+16]); vsmem[tx] = reduce.result; }
		if (blocksize >=  16) { reduce.partial(vsmem[tx+8]);  vsmem[tx] = reduce.result; }
		if (blocksize >=  8)  { reduce.partial(vsmem[tx+4]);  vsmem[tx] = reduce.result; }
		if (blocksize >=  4)  { reduce.partial(vsmem[tx+2]);  vsmem[tx] = reduce.result; }
		if (blocksize >=  2)  { reduce.partial(vsmem[tx+1]);  vsmem[tx] = reduce.result; }

		// The final reduced array has gdx contiguous elements (one element per block)
		if (tx == 0) {
			reduce.result = smem[0];
			reduce.finalize(original_n);  // The final output should always call reduce.finalize, since (for example) reducer_mean divides by m in this case
			dst[bx] = reduce.result;
		}
	}
}

template <typename T> 
T divup_pow2(T x)
{
	--x;
	for (T y = x; y; y >>= 1)
		x |= y;
	return ++x;
}

SM_CUDA_EXPORT launchcfg make_reduce_launchcfg(dtype_t dt, usize_t size);

template <template <typename value_type, typename result_type> class reducer>
struct execute_reduce_typed {
	template <typename S, typename D>
	struct general {
		static void execute(opcode_t opcode, const argument& src, const argument& dst)
		{
			// First pass over the data 
			usize_t original_size = src.size();
			usize_t size = src.size();
			launchcfg cfg = make_reduce_launchcfg(dst.dtype,size);
			D* dst_ptr = dst.get<D*>();
			heap_alloc buffer;

			if (cfg.gdim.x > 1) {
				// There is more than one output block, so we cannot directly write to "dst".
				// Instead, allocate temporary memory to store the partial reductions.
				buffer = thread_cudactx().heap().alloc(cfg.gdim.x*sizeof(D));
				dst_ptr = (D*)buffer.addr;
			}
			typedef reducer<S,D> reducer_SD;
			switch (cfg.bdim.x) {
			#define LAUNCH_CASE(bdx) case bdx: kernel_reduce<bdx,reducer_SD><<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(src.get<const S*>(),dst_ptr,size,cfg.gdim.x > 1 ? 0 : original_size); break;
			LAUNCH_CASE(512)
			LAUNCH_CASE(256)
			LAUNCH_CASE(128)
			LAUNCH_CASE( 64)
			LAUNCH_CASE( 32)
			LAUNCH_CASE( 16)
			LAUNCH_CASE(  8)
			LAUNCH_CASE(  4)
			LAUNCH_CASE(  2)
			LAUNCH_CASE(  1)
			default: SM_UNREACHABLE();
			#undef LAUNCH_CASE
			}

			while (cfg.gdim.x > 1) {
				size = cfg.gdim.x;  // new number of elements is equal to number of blocks in the first pass
				cfg = make_reduce_launchcfg(dst.dtype,size);
				unsigned nthread = cfg.bdim.x;
				D* src_ptr = dst_ptr; // read from the temporary buffer, possibly also writing to it if this is not the last iteration.
				if (cfg.gdim.x == 1)
					dst_ptr = dst.get<D*>();  // we've come down to the last iteration, so pass it along to the final output destination
				typedef reducer_partial_results<reducer_SD> reducer_partial;
				switch (nthread) {
				#define LAUNCH_CASE(bdx) case bdx: kernel_reduce<bdx,reducer_partial><<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(src_ptr,dst_ptr,size,cfg.gdim.x > 1 ? 0 : original_size); break;
				LAUNCH_CASE(512)
				LAUNCH_CASE(256)
				LAUNCH_CASE(128)
				LAUNCH_CASE( 64)
				LAUNCH_CASE( 32)
				LAUNCH_CASE( 16)
				LAUNCH_CASE(  8)
				LAUNCH_CASE(  4)
				LAUNCH_CASE(  2)
				LAUNCH_CASE(  1)
				#undef LAUNCH_CASE
				}
				size = (size + (nthread*2-1))/(nthread*2); // *2 because each thread sums two elements in its main loop
			}
			
			// If we had to allocate temporary memory for storing intermediate results, free it now
			if (buffer.addr)
				thread_cudactx().heap().free(buffer);
		}
	};

	template <typename T> struct matched  { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,T>::execute(opcode,src,dst); } };
	template <typename T> struct promoted { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,typename ctype2ptype(T)>::execute(opcode,src,dst); } };
	template <typename T> struct asfloat  { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,typename ctype2ftype(T)>::execute(opcode,src,dst); } };
	template <typename T> struct asuindex { static void execute(opcode_t opcode, const argument& src, const argument& dst) { general<T,reduce_index_t>::execute(opcode,src,dst); } };
};


SM_NAMESPACE_END

#endif // __SM_SMAT_CUDA_REDUCE_H__
