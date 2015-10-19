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
#include <smat_cuda/launch_util.h>
#include <smat_cuda/cuda_context.h>
#include <base/util.h>

SM_NAMESPACE_BEGIN

launchcfg make_elemwise_launchcfg(usize_t size)
{
	auto& dprop = thread_cudactx().deviceprop();
	unsigned  min_threads = dprop.warpSize;
	unsigned  max_threads = 256;
	unsigned  resident_blocks_per_multiprocessor = 8;
	unsigned  max_blocks = 4*resident_blocks_per_multiprocessor*dprop.multiProcessorCount;
	unsigned  block_size = max_threads;
	unsigned  grid_size  = max_blocks;

	if (size < min_threads) {
		// Array can be handled by the smallest block size.
		block_size = min_threads;
		grid_size = 1;
	} else if (size < max_blocks*min_threads) {
		// Array can be handled by several blocks of the smallest size.
		block_size = min_threads;
		grid_size = divup(size,block_size);
	} else if (size < max_blocks*max_threads) {
		// Array must be handled by max number of blocks, each of 
		// larger-than-minimal size, but still a multiple of warp size.
		// In this case, each thread within a block should handle 
		// multiple elements, looping until the entire grid has
		// processed 'size' unique elements.
		block_size = divup(divup(size,min_threads),max_blocks)*min_threads;
		grid_size = max_blocks;
	} else {
		// do nothing
	}

	return launchcfg(grid_size,block_size,0,thread_cudactx().stream());
}

SM_NAMESPACE_END
