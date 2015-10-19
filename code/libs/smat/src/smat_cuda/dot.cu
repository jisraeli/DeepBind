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
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>
#include <smat/vm/instruction_db.h>

SM_NAMESPACE_BEGIN

SM_INLINE cublasOperation_t cublas_opA(opcode_t opcode) { return (opcode == oc_dottn || opcode == oc_dottt) ? CUBLAS_OP_T : CUBLAS_OP_N; }
SM_INLINE cublasOperation_t cublas_opB(opcode_t opcode) { return (opcode == oc_dotnt || opcode == oc_dottt) ? CUBLAS_OP_T : CUBLAS_OP_N; }

template <typename T>
struct execute_dot_typed { };  // Only implemented for float types supported by CUBLAS.

// Use CUBLAS for float or double type.
template <>
struct execute_dot_typed<float> {
	static void execute(opcode_t opcode, const argument& a, const argument& b, const argument& c)
	{
		float alpha = 1, beta = 0;
		int n = (int)c.shape.y;
		int m = (int)c.shape.x;
		int k = (int)(cublas_opA(opcode) == CUBLAS_OP_T ? a.shape.y : a.shape.x);
		if (n > 0 && m > 0 && k > 0) {
			ccb(Sgemm,thread_cudactx().cublas(),
				cublas_opB(opcode),cublas_opA(opcode),
				m,n,k,&alpha,
				b.get<const float*>(),(int)b.shape.x,
				a.get<const float*>(),(int)a.shape.x,&beta,
				c.get<      float*>(),(int)c.shape.x);
		}
	}
};

template <>
struct execute_dot_typed<double> {
	static void execute(opcode_t opcode, const argument& a, const argument& b, const argument& c)
	{
		double alpha = 1, beta = 0;
		int n = (int)c.shape.y;
		int m = (int)c.shape.x;
		int k = (int)(cublas_opA(opcode) == CUBLAS_OP_T ? a.shape.y : a.shape.x);
		if (n > 0 && m > 0 && k > 0) {
			ccb(Dgemm,thread_cudactx().cublas(),
				cublas_opB(opcode),cublas_opA(opcode),
				m,n,k,&alpha,
				b.get<const double*>(),(int)b.shape.x,
				a.get<const double*>(),(int)a.shape.x,&beta,
				c.get<      double*>(),(int)c.shape.x);
		}
	}
};

void execute_dot(opcode_t opcode, const argument& a, const argument& b, const argument& c)
{
	DECL_SPECIALIZATION_TABLE(T_F,execute_fn3,execute_dot_typed);
	specialization_table(a.dtype)(opcode,a,b,c);
}

SM_NAMESPACE_END
