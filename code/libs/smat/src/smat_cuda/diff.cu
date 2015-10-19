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
__global__ void kernel_diff_y(const T* src, T* dst, usize_t m, usize_t size)
{
	DECL_KERNEL_VARS
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)
		dst[i] = src[i+m]-src[i];  // could be implemented by oc_sub operation on two views of arg, but this should be marginally faster.
}


template <typename T>
struct execute_diff_typed {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		usize_t size = (usize_t)dst.size();
		if (size == 0)
			return;
		if (opcode == oc_diff_x) {
			SM_UNIMPLEMENTED();
		} else if (opcode == oc_diff_y) {
			launchcfg cfg = make_elemwise_launchcfg(size);
			kernel_diff_y<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(src.get<const T*>(),dst.get<T*>(),dst.shape.x,size);
		} else {
			SM_UNREACHABLE();
		}
	}
};

void execute_diff(opcode_t opcode, const argument& src, const argument& dst)
{
	DECL_SPECIALIZATION_TABLE(T_G,execute_fn2,execute_diff_typed);
	specialization_table(src.dtype)(opcode,src,dst);
}

SM_NAMESPACE_END
