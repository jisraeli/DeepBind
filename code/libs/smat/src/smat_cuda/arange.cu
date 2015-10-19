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

template <typename T>
__global__ void kernel_arange(T start, T* dst, usize_t size)
{
	DECL_KERNEL_VARS
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)
		dst[i] = start + (T)i;
}

template <typename T>
struct execute_arange_typed { // TODO: autotune this
	static void execute(opcode_t opcode, const argument& start, const argument& dst)
	{
		usize_t size = (usize_t)dst.size();
		if (size > 0) {
			launchcfg cfg = make_elemwise_launchcfg(size);
			kernel_arange<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(start.get<T>(),dst.get<T*>(),size);
		}
	}
};

void execute_arange(opcode_t opcode, const argument& start, const argument& dst)
{
	DECL_SPECIALIZATION_TABLE(T_N,execute_fn2,execute_arange_typed);
	specialization_table(dst.dtype)(opcode,start,dst);
}

SM_NAMESPACE_END
