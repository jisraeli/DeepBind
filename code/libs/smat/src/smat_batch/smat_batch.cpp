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
#include <smat/vm/instruction_db.h>
#include <smat/smat.h>
using namespace sm;

void launch_batch_dot_general(const void* A, const void* B, void* C, dtype_t dt,
                              int n, int m, int k, int p, bool broadcast_A,
                              double alpha, double beta, 
                              bool transA, bool transB);

// Each block A[i,:] is n x k
// Each block B[i,:] is k x m
// Each block C[i,:] is n x m

void validate_batch_dot(opcode_t opcode, const argument& A, const argument& B, const argument& C)
{
	int k = A.shape.x;
	int m = B.shape.x;
	int p = B.shape.y / k;
	int n = C.shape.y / p;
	SM_ASSERT(A.shape.y == p*n || A.shape.y == n);
	SM_ASSERT(B.shape.y == p*k);
	SM_ASSERT(C.shape.y == p*n);
	SM_ASSERT(C.shape.x == B.shape.x);
}

void execute_batch_dot(opcode_t opcode, const argument& A, const argument& B, const argument& C)
{
	int k = A.shape.x;
	int m = B.shape.x;
	int p = B.shape.y / k;
	int n = C.shape.y / p;
	launch_batch_dot_general(A.get<const void*>(), B.get<const void*>(), C.get<void*>(), A.dtype,
	                         n, m, k, p, (A.shape.y == n), 0.0, 1.0, false, false);
}

void validate_batch_dot_nt(opcode_t opcode, const argument& A, const argument& B, const argument& C)
{
	int k = A.shape.x;
	int m = B.shape.x;
	int p = B.shape.y / k;
	int n = C.shape.y / p;

	SM_ASSERT(A.shape.y == p*n || A.shape.y == n);
	SM_ASSERT(B.shape.y == p*k);
	SM_ASSERT(C.shape.y == p*n);
	SM_ASSERT(C.shape.x == B.shape.x);

	SM_ASSERT(nblock.dtype == i32);
	SM_ASSERT(nblock.vtype == vt_carray);
	int _nblock = nblock.get<int>();
	SM_ASSERT(dZ.dtype == W.dtype);
	SM_ASSERT(dZ.dtype == dX.dtype);
	SM_ASSERT(dX.shape.y == dZ.shape.y);
	SM_ASSERT(dZ.shape.x == W.shape.x*_nblock);
	SM_ASSERT(W.shape.y % _nblock == 0);
	SM_ASSERT(dZ.shape.x % _nblock == 0);
	SM_ASSERT(dX.shape.x == W.shape.y);
}

void execute_batch_dot_nt(opcode_t opcode, const argument& nblock, const argument& dZ, const argument& W, const argument& dX)
{
	// dX[:,i] is m x k
	//  W[i,:] is k x n
	// dZ[:,i] is m x n
	// ... so ...
	// dX is m x (k*nblock)
	//  W is (k*nblock) x n
	// dZ is m x (n*nblock)
	int _nblock = nblock.get<int>();
	int n = dX.shape.y;
	int k = W.shape.y / _nblock;
	int m = W.shape.x;
	launch_batch_dot_nt(thread_cudactx().stream(), dZ.dtype,
	                        _nblock, n, m, k, dZ.get<const void*>(), W.get<const void*>(), dX.get<void*>());
}

void validate_batch_dot_tn(opcode_t opcode, const argument& nblock, const argument& X, const argument& dZ, const argument& dW)
{
	SM_ASSERT(nblock.dtype == i32);
	SM_ASSERT(nblock.vtype == vt_carray);
	int _nblock = nblock.get<int>();
	SM_ASSERT(dZ.dtype == dW.dtype);
	SM_ASSERT(dZ.dtype == X.dtype);
	SM_ASSERT(X.shape.y == dZ.shape.y);
	SM_ASSERT(dZ.shape.x == dW.shape.x*_nblock);
	SM_ASSERT(dW.shape.y % _nblock == 0);
	SM_ASSERT(dZ.shape.x % _nblock == 0);
	SM_ASSERT(X.shape.x == dW.shape.y || X.shape.x*_nblock == dW.shape.y);
}

void execute_batch_dot_tn(opcode_t opcode, const argument& nblock, const argument& X, const argument& dZ, const argument& dW)
{
	//  X[:,i] is m x k
	// dW[i,:] is k x n
	// dZ[:,i] is m x n
	// ... so ...
	//  X is m x (k*nblock)
	// dW is (k*nblock) x n
	// dZ is m x (n*nblock)
	int _nblock = nblock.get<int>();
	int n = X.shape.y;
	int k = dW.shape.y / _nblock;
	int m = dW.shape.x;
	launch_batch_dot_tn(thread_cudactx().stream(), X.dtype,
	                        _nblock, n, m, k, X.shape.x*_nblock == dW.shape.y,
							X.get<const void*>(), dZ.get<const void*>(), dW.get<void*>());
}





opcode_t oc_bw_dot = -1;
opcode_t oc_bw_dot_nt = -1;
opcode_t oc_bw_dot_tn = -1;

#pragma warning(disable : 4190 4297)  // disable warning about C linkage of shape_t, and about throwing exceptions from C functions

extern "C" {

SM_DLLEXPORT void api_batch_dot(int nblock, const smat* X, const smat* W, smat* Z)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_bw_dot, carray(nblock), X->as_arg(), W->as_arg(), Z->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_batch_dot_nt(int nblock, const smat* dZ, const smat* W, smat* dX)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_bw_dot_nt, carray(nblock), dZ->as_arg(), W->as_arg(), dX->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_batch_dot_tn(int nblock, const smat* X, const smat* dZ, smat* dW)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_bw_dot_tn, carray(nblock), X->as_arg(), dZ->as_arg(), dW->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void register_ext()
{
	oc_bw_dot = add_instruction("bw_dot",
			iprop_none,
			aprop_in |          aprop_int,        // (nblock)
			aprop_in |          aprop_float,      // (X)
			aprop_in |          aprop_float,      // (W)
			          aprop_out|aprop_float       // (Z)
		);

	oc_bw_dot_nt = add_instruction("bw_dot_nt",
			iprop_none,
			aprop_in |          aprop_int,        // (nblock)
			aprop_in |          aprop_float,      // (dZ)
			aprop_in |          aprop_float,      // (W)
			          aprop_out|aprop_float       // (dX)
		);

	oc_bw_dot_tn = add_instruction("bw_dot_tn",
			iprop_none,
			aprop_in |          aprop_int,        // (nblock)
			aprop_in |          aprop_float,      // (X)
			aprop_in |          aprop_float,      // (dZ)
			          aprop_out|aprop_float       // (dW)
		);

	add_instruction_impl(cuda_uuid,oc_bw_dot    ,execute_batch_dot   ,validate_batch_dot);
	add_instruction_impl(cuda_uuid,oc_bw_dot_nt ,execute_batch_dot_nt,validate_batch_dot_nt);
	add_instruction_impl(cuda_uuid,oc_bw_dot_tn ,execute_batch_dot_tn,validate_batch_dot_tn);
}

}


