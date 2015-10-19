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
#ifndef __KR_CONVSEQ_H__
#define __KR_CONVSEQ_H__

#include <smat/dtypes.h>
#include <base/util.h>
#include <base/assert.h>
using namespace sm;



template <unsigned bdx,               // =  8 threads per block in "filter" dimension
          unsigned bdy,               // = 16 threads per block in "input samples" dimension
          unsigned samples_per_thread,// =  8 input samples per thread
          unsigned filters_per_thread,// =  4 filters per thread 
          unsigned filter_size,       // =  ? number of elements in filter
          unsigned nchannel,          // =  4 input channels per sample (therefore per filter)
          typename float_t>
__global__ 
void convseq_kernel(const uint8_t* samples, usize_t nsample,
                    const float_t* filters, usize_t nfilter,
                          float_t* featuremaps)
{
	const unsigned tx  = threadIdx.x;
	const unsigned ty  = threadIdx.y;
	const unsigned bx  = blockIdx.x;
	const unsigned by  = blockIdx.y;
	const unsigned tid = bdx*ty+tx;

	const unsigned nthread               = bdx*bdy;
	const unsigned samples_per_block     = bdy*samples_per_thread;      // Each block ultimately convolves this many input samples with each filter.
	const unsigned filters_per_block     = bdx*filters_per_thread;      // Each block ultimately convolves each input sample with this many filters.
	const unsigned filter_elem           = filter_size*nchannel;        // floating point elements per filter
	const unsigned filter_elem_per_block = filters_per_block*filter_elem;
	
	// Shared memory.
	// The input samples are loaded one contiguous block at a time
	// into s_samples, and are the convolved with whatever filters were
	// loaded into s_filters.
	__shared__ uint32_t s_samples[nthread+filter_size-1]; // +filter_size-1 samples that overlap with next block
	__shared__ float_t  s_filters[filters_per_block][filter_size][nchannel]; // same memory layout as filters argument

	/*
#ifdef _DEBUG
	// Fill shared memory with something that will trigger the debugger if we accidentally read an uninitialized value
	if (tid == 0) {
		for (int i = tid; i < nthread+filter_size-1; i += nthread)
			s_samples[i] = 1000000000;
		for (int i = tid; i < filter_size*filters_per_block*nchannel; i += nthread)
			((float_t*)s_filters)[i] = -100000;
	}
	__syncthreads();
#endif
	*/

	// Setup: Load all filters.
	// For each filter in our block (not exceeding nfilter),
	// load its weights into shared memory.
	filters += filter_elem_per_block*bx; // Jump to start of filters for this block.
	unsigned filters_this_block = ::min(filters_per_block,nfilter-filters_per_block*bx);
	#pragma unroll
	for (unsigned i = tid; i < filters_this_block*filter_elem; i += nthread)
		((float_t*)s_filters)[i] = filters[i];

	// Main loop.
	// For each consecutive block of input handled by this thread...
	samples     += samples_per_block*by;             // Jump to start of inputs/outputs for this block. Each sample is one byte, regardless of nchannel.
	featuremaps += samples_per_block*by*nfilter + filters_per_block*bx; // However, each block writes nfilter outputs for every input sample.
	unsigned samples_this_block = ::min(samples_per_block,nsample-samples_per_block*by);
	for (uindex_t s = 0; s < samples_this_block; s += nthread) {

		///////////////////////////////////////////////////////////////////////
		// Step 1.
		// Load block of inputs into shared memory. Even though inputs are
		// 8 bytes each, the L1 cache on later GPUs makes it reasonably fast
		// for each consecutive thread to read a consecutive byte, so we let
		// all threads participate in the load.
		
		// First load the single element corresponding to this thread id.
		// Then, a subset of threads load the trailing inputs that overlap with the next block.
		s_samples[tid] = (uint32_t)((s+tid < samples_this_block) ? samples[s+tid] : 0xffffffff);
		if (tid < filter_size-1)
			s_samples[nthread+tid] = (uint32_t)((nthread+s+tid < nsample-samples_per_block*by) ? samples[nthread+s+tid] : 0xffffffff);
		__syncthreads();

		///////////////////////////////////////////////////////////////////////
		// Step 2.
		// Convolve all these inputs with the pre-loaded filters for this block.
		//
		// This thread (tid) should convolve every a particular subset of samples (every bdy'th)
		// with a particular subset of filters (every bdx'th)
		// 
		for (unsigned i = ty; s+i < samples_this_block; i += bdy) {
			#pragma unroll
			for (unsigned f = tx; f < filters_this_block; f += bdx) {
				float_t sum = 0;
				#pragma unroll
				for (unsigned t = 0; t < filter_size; ++t) {
					// Step 2 inner loop.
					//  f  is the particular filter this thread currently uses to convolve.
					//  t  is the index of the current filter element 
					//  i  is the index of the current start-sample within s_input
					// So filter[f] gets its element t multiplied with input[i+t] for t=0..filter_size-1.
					// We do not have to worry about i taking us past boundary of s_samples, because
					// we added "apron" elements from the next block to the end of s_samples.
					uint32_t q = s_samples[i+t];
					if (q < nchannel)
						sum += s_filters[f][t][q];
				}
				featuremaps[(s+i)*nfilter+f] = sum; // this will be coalesced if bdx is large enough
			}
		}
		__syncthreads(); // wait for everyone to finish before moving on to the next block of input samples
	}
}




template <unsigned bdx,                // =  8 threads per block in "filter" dimension
          unsigned bdy,                // =  filter size
          unsigned segments_per_block, // = 16 segment junctions handled by each block
          unsigned filters_per_thread, // =  4 filters per thread (all samples)
          unsigned nchannel,      // =  4 input channels per sample (therefore per filter)
          typename float_t>
__global__ 
void convseq_kernel_applysegs(const uint8_t*  samples, usize_t nsample,
                              const float_t*  filters, usize_t nfilter,
                              const uindex_t* segments, usize_t nsegment,
                              float_t* featuremaps)
{
	const unsigned tx  = threadIdx.x;
	const unsigned ty  = threadIdx.y;
	const unsigned bx  = blockIdx.x;
	const unsigned by  = blockIdx.y;
	const unsigned tid = bdx*ty+tx;

	const unsigned num_threads        = bdx*bdy;
	const unsigned filter_size        = bdy;
	const unsigned filters_per_block  = bdx*filters_per_thread;          // Each block ultimately convolves each input sample with this many filters.
	const unsigned filter_elem        = filter_size*nchannel;        // floating point elements per filter
	const unsigned filter_elem_per_block = filters_per_block*filter_elem;
	
	__shared__ uindex_t s_segments[segments_per_block];
	__shared__ uint32_t s_samples[filter_size > 1 ? filter_size-1 : 1];
	__shared__ float_t  s_filters[filters_per_block][filter_size][nchannel]; // same memory layout as filters argument

	/*
#ifdef _DEBUG
	// Fill shared memory with something that will trigger the debugger if we accidentally read an uninitialized value
	for (int i = tid; i < segments_per_block; i+=bdx*bdy)
		s_segments[i] = 1000000000;
	for (int i = tid; i < filter_size-1; i+=bdx*bdy)
		s_samples[i] = 1000000000;
	for (int i = tid; i < filter_size*filters_per_block*nchannel; i+=bdx*bdy)
		((float_t*)s_filters)[i] = -100000;
	__syncthreads();
#endif
	*/

	filters += filter_elem_per_block*bx; // Jump to start of filters for this block.
	unsigned filters_this_block = ::min(filters_per_block,nfilter-filters_per_block*bx);
	#pragma unroll
	for (unsigned i = tid; i < filters_this_block*filter_elem; i += num_threads)
		((float_t*)s_filters)[i] = filters[i];

	// Main loop.
	// For each segment assigned to this block, loop over the indices and start convolving.
	segments += segments_per_block*by;
	unsigned segments_this_block = ::min(segments_per_block,nsegment-segments_per_block*by);
	if (tid < segments_this_block)
		s_segments[tid] = segments[tid]; // Pre-load segments into shared memory -- threads will have bank conflicts, but oh well

	uindex_t j0 = by > 0 ? segments[-1] : 0;
	uindex_t j1;

	for (uindex_t s = 0; s < segments_this_block; ++s, j0=j1) {
		__syncthreads();

		// Compute starting address for this segment's tail convolution.
		j1 = s_segments[s];                   // j1 = index of the input sample that is just past the end of this segment.
		j0 = ::max(j1-(isize_t)filter_size+1, j0);   // j0 = index of first input sample that needs to be re-convolved.
		
		// Load up to filter_size samples from the input.
		const int samples_this_seg = j1-j0;
		if (tid < samples_this_seg) {
			// This thread has a unique sample associated with it, so load it from the input
			const uint8_t* segment_input = samples + j0;
			s_samples[tid] = (uint32_t)segment_input[tid];
		}
		__syncthreads();

		// Only convolve sample s_input[ty+...] if our thread id is low enough to be
		// responsible for a sample.
		if (ty < samples_this_seg) {
			float_t* seg_out = featuremaps + (j0+ty)*nfilter;
			// Now for each filter this thread is responsible for (based on tx), 
			// convolve input at our assigned offset (based on ty).
			// The main difference with the normal convolve is that here
			// we explicitly terminate the innermost loop when we hit the
			// end of the segment.
			#pragma unroll
			for (unsigned f = tx; f < filters_this_block; f += bdx) {
				float_t sum = 0;
				#pragma unroll
				for (unsigned t = 0; ty+t < samples_this_seg; ++t) {
					uint32_t q = s_samples[ty+t];
					if (q < nchannel)
						sum += s_filters[f][t][q];
				}
				seg_out[f] = sum; // this will be coalesced if bdx is large enough
			}
		}
	} // done current segment, move on to the next
}

#endif // __KR_CONVSEQ_H__
