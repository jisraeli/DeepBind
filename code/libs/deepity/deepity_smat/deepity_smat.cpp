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
#include <smat/vm/instruction_db.h>
#include <smat/smat.h>
#include <cstring>
using namespace sm;

void launch_gradstep(cudaStream_t stream, dtype_t dtype, isize_t n,
                     void* P, const void* dP, const void* drate,
                                    void* mP, const void* mrate);

void launch_gradstep_nesterov1(cudaStream_t stream, dtype_t dtype, isize_t n,
                               void* P, const void* mP, const void* mrate);

void launch_gradstep_nesterov2(cudaStream_t stream, dtype_t dtype, isize_t n,
                               void* P, const void* dP, const void* drate,
                                              void* mP, const void* mrate);

void launch_madd_bcast(cudaStream_t stream, dtype_t dtype,
                        const void* A, const void* b, void* dst,
                        usize_t n, usize_t m, usize_t k);

void launch_maskout(cudaStream_t stream, dtype_t dtype, const bool* M, void* A, usize_t n);

void launch_calc_zmask(cudaStream_t stream, dtype_t dtype, const void* Z, bool* M, usize_t n, usize_t m);

void launch_dropout_fp_tr(cudaStream_t stream, dtype_t dtype,
                          const void* X, const void* rate, void* Z, bool* M,
                          usize_t n, usize_t m, usize_t k, bool matchrows);

void launch_dropout_fp_te(cudaStream_t stream, dtype_t dtype,
                          const void* X, const void* rate, void* Z,
                          usize_t n, usize_t m, usize_t k);

void launch_dropout_bp_tr(cudaStream_t stream, dtype_t dtype,
                          const void* dZ, const bool* M, void* dX, usize_t n);

void launch_dropout_bp_te(cudaStream_t stream, dtype_t dtype,
                          const void* dZ, const void* rate, void* dX, usize_t n, usize_t m, usize_t k);

void launch_blockwise_dot(cudaStream_t stream, dtype_t dtype,
                          int nblock, int n, int m, int k, bool broadcast_x,
                          const void* X, const void* W, void* Z);

void launch_blockwise_dot_nt(cudaStream_t stream, dtype_t dtype,
                             int nblock, int n, int m, int k, 
                             const void* dZ, const void* W, void* dX);

void launch_blockwise_dot_tn(cudaStream_t stream, dtype_t dtype,
                             int nblock, int n, int m, int k, bool broadcast_x,
                             const void* X, const void* dZ, void* dW);

void launch_blockwise_dot_combined(cudaStream_t stream, dtype_t dtype,
                                   int nblock, int Xoffset, bool Xbroadcast, 
                                   int n, int m, int k, int p,
                                   const void* X, const void* W, void* Z);

void launch_blockwise_dot_nt_combined(cudaStream_t stream, dtype_t dtype,
                                      int nblock, int Xoffset,
                                      int n, int m, int k, int p,
                                      void* dX, const void* W, const void* dZ);

void launch_blockwise_dot_tn_combined(cudaStream_t stream, dtype_t dtype,
                                      int nblock, int Xoffset, bool Xbroadcast,
                                      int n, int m, int k, int p,
                                      const void* dX, void* dW, const void* dZ);

void validate_gradstep(opcode_t opcode, const argument& P,
                                        const argument& dP, const argument& drate,
                                        const argument& mP, const argument& mrate)
{
	auto n = P.shape.y;
	SM_ASSERT(dP.shape.y == n);
	SM_ASSERT(mP.shape.y == n);
	SM_ASSERT(drate.shape.y == n);
	SM_ASSERT(mrate.shape.y == n);
}


void execute_gradstep(opcode_t opcode, const argument& P,
                                       const argument& dP, const argument& drate,
                                       const argument& mP, const argument& mrate)
{
	launch_gradstep(thread_cudactx().stream(),P.dtype,P.shape.y,
	                 P.get<      void*>(),
	                dP.get<const void*>(),drate.get<const void*>(),
	                mP.get<      void*>(),mrate.get<const void*>());
}

void validate_gradstep_nesterov1(opcode_t opcode, const argument& P,                                        
                                                  const argument& mP, const argument& mrate)
{
	auto n = P.shape.y;
	SM_ASSERT(mP.shape.y == n);
	SM_ASSERT(mrate.shape.y == n);
}


void execute_gradstep_nesterov1(opcode_t opcode, const argument& P,                                                
                                                 const argument& mP, const argument& mrate)
{
	launch_gradstep_nesterov1(thread_cudactx().stream(),P.dtype,P.shape.y,
	                  P.get<void*>(),mP.get<const void*>(),mrate.get<const void*>());
}

void validate_gradstep_nesterov2(opcode_t opcode, const argument& P,
                                                  const argument& dP, const argument& drate,
                                                  const argument& mP, const argument& mrate)
{
	auto n = P.shape.y;
	SM_ASSERT(dP.shape.y == n);
	SM_ASSERT(mP.shape.y == n);
	SM_ASSERT(drate.shape.y == n);
	SM_ASSERT(mrate.shape.y == n);
}


void execute_gradstep_nesterov2(opcode_t opcode, const argument& P,
                                                 const argument& dP, const argument& drate,
                                                 const argument& mP, const argument& mrate)
{
	launch_gradstep_nesterov2(thread_cudactx().stream(),P.dtype,P.shape.y,
	                          P.get<      void*>(),
	                          dP.get<const void*>(),drate.get<const void*>(),
	                          mP.get<      void*>(),mrate.get<const void*>());
}

void validate_madd_bcast(opcode_t opcode, const argument& A, const argument& b,
                                          const argument& k, const argument& dst)
{
	SM_ASSERT(A.dtype == dst.dtype);
	SM_ASSERT(A.dtype == b.dtype);
	SM_ASSERT(k.dtype == ctype2dtype(usize_t));
	SM_ASSERT(A.size() == dst.size());
	SM_ASSERT(A.size() % b.size() == 0);
}


void execute_madd_bcast(opcode_t opcode, const argument& A, const argument& b,
                                         const argument& k, const argument& dst)
{
	launch_madd_bcast(thread_cudactx().stream(),A.dtype,
	                 A.get<const void*>(), b.get<const void*>(), dst.get<void*>(),
	                 A.size(), b.size(), k.get<usize_t>());
}

void validate_maskout(opcode_t opcode, const argument& M, const argument& A)
{
	SM_ASSERT(A.size() == M.size());
}

void execute_maskout(opcode_t opcode, const argument& M, const argument& A)
{
	launch_maskout(thread_cudactx().stream(),A.dtype, M.get<const bool*>(), A.get<void*>(), A.size());
}

void validate_calc_zmask(opcode_t opcode, const argument& Z, const argument& M)
{
	SM_ASSERT(Z.size() == M.size());
}

void execute_calc_zmask(opcode_t opcode, const argument& Z, const argument& M)
{
	launch_calc_zmask(thread_cudactx().stream(),Z.dtype, Z.get<const void*>(), M.get<bool*>(), Z.shape.y, Z.shape.x);
}

void validate_dropout_fp_tr(opcode_t opcode, const argument& X, const argument& rate,
                                             const argument& Z, const argument& M, const argument& matchrows)
{
	SM_ASSERT(X.dtype == rate.dtype);
	SM_ASSERT(X.dtype == Z.dtype);
	SM_ASSERT(X.size() == Z.size());
	SM_ASSERT(X.size() == M.size());
	SM_ASSERT(X.size() % rate.size() == 0);
}

void execute_dropout_fp_tr(opcode_t opcode, const argument& X, const argument& rate,
                                            const argument& Z, const argument& M, const argument& matchrows)
{
	launch_dropout_fp_tr(thread_cudactx().stream(),X.dtype,
	                     X.get<const void*>(), rate.get<const void*>(), Z.get<void*>(), M.get<bool*>(),
	                     X.size(), rate.size(), X.shape.x/rate.size(), matchrows.get<bool>());
}

void validate_dropout_fp_te(opcode_t opcode, const argument& X, const argument& rate, const argument& Z)
{
	SM_ASSERT(X.dtype == rate.dtype);
	SM_ASSERT(X.dtype == Z.dtype);
	SM_ASSERT(X.size() == Z.size());
	SM_ASSERT(X.size() % rate.size() == 0);
}

void execute_dropout_fp_te(opcode_t opcode, const argument& X, const argument& rate, const argument& Z)
{
	launch_dropout_fp_te(thread_cudactx().stream(),X.dtype,
	                     X.get<const void*>(), rate.get<const void*>(), Z.get<void*>(),
	                     X.size(), rate.size(), X.shape.x/rate.size());
}


void validate_dropout_bp_tr(opcode_t opcode, const argument& dZ, const argument& M, const argument& dX)
{
	SM_ASSERT(dX.dtype == dZ.dtype);
	SM_ASSERT(dX.size() == M.size());
	SM_ASSERT(dX.size() == dZ.size());
}

void execute_dropout_bp_tr(opcode_t opcode, const argument& dZ, const argument& M, const argument& dX)
{
	launch_dropout_bp_tr(thread_cudactx().stream(),dZ.dtype, dZ.get<const void*>(), M.get<const bool*>(), dX.get<void*>(), dX.size());
}

void validate_dropout_bp_te(opcode_t opcode, const argument& dZ, const argument& rate, const argument& dX)
{
	SM_ASSERT(dX.dtype == dZ.dtype);
	SM_ASSERT(dX.size() == dZ.size());
	SM_ASSERT(dX.size() % rate.size() == 0);
}

void execute_dropout_bp_te(opcode_t opcode, const argument& dZ, const argument& rate, const argument& dX)
{
	launch_dropout_bp_te(thread_cudactx().stream(),dZ.dtype,
	                     dZ.get<const void*>(), rate.get<const void*>(), dX.get<void*>(),
	                     dX.size(), rate.size(), dX.shape.x/rate.size());
}

void validate_blockwise_dot(opcode_t opcode, const argument& nblock, const argument& X, const argument& W, const argument& Z)
{
	SM_ASSERT(nblock.dtype == i32);
	SM_ASSERT(nblock.vtype == vt_carray);
	int _nblock = nblock.get<int>();
	SM_ASSERT(X.dtype == W.dtype);
	SM_ASSERT(X.dtype == Z.dtype);
	SM_ASSERT(X.shape.y == Z.shape.y);
	SM_ASSERT(Z.shape.x == W.shape.x*_nblock);
	SM_ASSERT(W.shape.y % _nblock == 0);
	SM_ASSERT(Z.shape.x % _nblock == 0);
	SM_ASSERT(X.shape.x == W.shape.y || X.shape.x*_nblock == W.shape.y);
}

void execute_blockwise_dot(opcode_t opcode, const argument& nblock, const argument& X, const argument& W, const argument& Z)
{
	// X[:,i] is m x k
	// W[i,:] is k x n
	// Z[:,i] is m x n
	// ... so ...
	// X is m x (k*nblock)
	// W is (k*nblock) x n
	// Z is m x (n*nblock)
	int _nblock = nblock.get<int>();
	int n = X.shape.y;
	int k = W.shape.y / _nblock;
	int m = W.shape.x;
	launch_blockwise_dot(thread_cudactx().stream(), X.dtype,
	                     _nblock, n, m, k, X.shape.x*_nblock == W.shape.y,
	                     X.get<const void*>(), W.get<const void*>(), Z.get<void*>());
}

void validate_blockwise_dot_nt(opcode_t opcode, const argument& nblock, const argument& dZ, const argument& W, const argument& dX)
{
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

void execute_blockwise_dot_nt(opcode_t opcode, const argument& nblock, const argument& dZ, const argument& W, const argument& dX)
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
	launch_blockwise_dot_nt(thread_cudactx().stream(), dZ.dtype,
	                        _nblock, n, m, k, dZ.get<const void*>(), W.get<const void*>(), dX.get<void*>());
}

void validate_blockwise_dot_tn(opcode_t opcode, const argument& nblock, const argument& X, const argument& dZ, const argument& dW)
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

void execute_blockwise_dot_tn(opcode_t opcode, const argument& nblock, const argument& X, const argument& dZ, const argument& dW)
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
	launch_blockwise_dot_tn(thread_cudactx().stream(), X.dtype,
	                        _nblock, n, m, k, X.shape.x*_nblock == dW.shape.y,
							X.get<const void*>(), dZ.get<const void*>(), dW.get<void*>());
}

void validate_blockwise_dot_combined(opcode_t opcode, const argument& nblock, const argument& Xoffset_Xbroadcast, 
                                     const argument& X, const argument& W, const argument& Z)
{
	SM_ASSERT(nblock.dtype == i32);
	SM_ASSERT(nblock.vtype == vt_carray);
	SM_ASSERT(Xoffset_Xbroadcast.dtype == i32);
	SM_ASSERT(Xoffset_Xbroadcast.vtype == vt_carray);
	int _nblock = nblock.get<int>();
	SM_ASSERT(X.dtype == W.dtype);
	SM_ASSERT(X.dtype == Z.dtype);
	SM_ASSERT(X.shape.y == Z.shape.y);
	SM_ASSERT(Z.shape.x == W.shape.x*_nblock);
	SM_ASSERT(W.shape.y % _nblock == 0);
	SM_ASSERT(Z.shape.x % _nblock == 0);
}

void execute_blockwise_dot_combined(opcode_t opcode, const argument& nblock, const argument& Xoffset_Xbroadcast, 
                                     const argument& X, const argument& W, const argument& Z)
{
	// X[:,i] is m x k
	// W[i,:] is k x n
	// Z[:,i] is m x n
	// ... so ...
	// X is m x (k*nblock)    (or m x k if broadcasting)
	// W is (p*nblock) x n
	// Z is m x (n*nblock)
	int _nblock  = nblock.get<int>();
	int Xoffset = Xoffset_Xbroadcast.get<int>() >> 1;
	bool Xbroadcast = (Xoffset_Xbroadcast.get<int>() & 1) != 0;
	int n = X.shape.y;
	int k = Xbroadcast ? X.shape.x : X.shape.x / _nblock;
	int m = W.shape.x;
	int p = W.shape.y / _nblock; // number of W rows to jump to reach next chunk of weights for consecutive "blocks" being multiplied
	launch_blockwise_dot_combined(thread_cudactx().stream(), X.dtype,
	                              _nblock, Xoffset, Xbroadcast, 
	                              n, m, k, p,
	                              X.get<const void*>(), W.get<const void*>(), Z.get<void*>());
}


void validate_blockwise_dot_nt_combined(opcode_t opcode, const argument& nblock, const argument& Xoffset_Xbroadcast, 
                                        const argument& dX, const argument& W, const argument& dZ)
{
	SM_ASSERT(nblock.dtype == i32);
	SM_ASSERT(nblock.vtype == vt_carray);
	SM_ASSERT(Xoffset_Xbroadcast.dtype == i32);
	SM_ASSERT(Xoffset_Xbroadcast.vtype == vt_carray);
	int _nblock = nblock.get<int>();
	bool Xbroadcast = (Xoffset_Xbroadcast.get<int>() & 1) != 0;
	SM_ASSERT(!Xbroadcast);
	SM_ASSERT(dX.dtype == W.dtype);
	SM_ASSERT(dX.dtype == dZ.dtype);
	SM_ASSERT(dX.shape.y == dZ.shape.y);
	SM_ASSERT(dZ.shape.x == W.shape.x*_nblock);
	SM_ASSERT(W.shape.y % _nblock == 0);
	SM_ASSERT(dZ.shape.x % _nblock == 0);
	SM_ASSERT(dX.shape.x % _nblock == 0);
}

void execute_blockwise_dot_nt_combined(opcode_t opcode, const argument& nblock, const argument& Xoffset_Xbroadcast, 
                                       const argument& dX, const argument& W, const argument& dZ)
{
	// X[:,i] is m x k
	// W[i,:] is k x n
	// Z[:,i] is m x n
	// ... so ...
	// X is m x (k*nblock)    (or m x k if broadcasting)
	// W is (p*nblock) x n
	// Z is m x (n*nblock)
	int _nblock  = nblock.get<int>();
	int Xoffset = Xoffset_Xbroadcast.get<int>() >> 1;
	int n = dX.shape.y;
	int k = dX.shape.x / _nblock;
	int m = W.shape.x;
	int p = W.shape.y / _nblock; // number of W rows to jump to reach next chunk of weights for consecutive "blocks" being multiplied
	launch_blockwise_dot_nt_combined(thread_cudactx().stream(), dX.dtype,
	                                 _nblock, Xoffset, n, m, k, p,
	                                 dX.get<void*>(), W.get<const void*>(), dZ.get<const void*>());
}


void validate_blockwise_dot_tn_combined(opcode_t opcode, const argument& nblock, const argument& Xoffset_Xbroadcast, 
                                        const argument& X, const argument& dW, const argument& dZ)
{
	SM_ASSERT(nblock.dtype == i32);
	SM_ASSERT(nblock.vtype == vt_carray);
	SM_ASSERT(Xoffset_Xbroadcast.dtype == i32);
	SM_ASSERT(Xoffset_Xbroadcast.vtype == vt_carray);
	int _nblock = nblock.get<int>();
	SM_ASSERT(X.dtype == dW.dtype);
	SM_ASSERT(X.dtype == dZ.dtype);
	SM_ASSERT(X.shape.y == dZ.shape.y);
	SM_ASSERT(dZ.shape.x == dW.shape.x*_nblock);
	SM_ASSERT(dW.shape.y % _nblock == 0);
	SM_ASSERT(dZ.shape.x % _nblock == 0);
}

void execute_blockwise_dot_tn_combined(opcode_t opcode, const argument& nblock, const argument& Xoffset_Xbroadcast, 
                                       const argument& X, const argument& dW, const argument& dZ)
{
	// X[:,i] is m x k
	// W[i,:] is k x n
	// Z[:,i] is m x n
	// ... so ...
	// X is m x (k*nblock)    (or m x k if broadcasting)
	// W is (p*nblock) x n
	// Z is m x (n*nblock)
	int _nblock  = nblock.get<int>();
	int Xoffset = Xoffset_Xbroadcast.get<int>() >> 1;
	bool Xbroadcast = (Xoffset_Xbroadcast.get<int>() & 1) != 0;
	int n = X.shape.y;
	int k = Xbroadcast ? X.shape.x : X.shape.x / _nblock;
	int m = dW.shape.x;
	int p = dW.shape.y / _nblock; // number of W rows to jump to reach next chunk of weights for consecutive "blocks" being multiplied
	launch_blockwise_dot_tn_combined(thread_cudactx().stream(), X.dtype,
	                                 _nblock, Xoffset, Xbroadcast, n, m, k, p,
	                                 X.get<const void*>(), dW.get<void*>(), dZ.get<const void*>());
}


opcode_t oc_gstep   = -1;
opcode_t oc_gstepn1 = -1;
opcode_t oc_gstepn2 = -1;
opcode_t oc_maddbc  = -1;
opcode_t oc_maskout = -1;
opcode_t oc_calczmask = -1;
opcode_t oc_drop_fp_tr = -1;
opcode_t oc_drop_fp_te = -1;
opcode_t oc_drop_bp_tr = -1;
opcode_t oc_drop_bp_te = -1;
opcode_t oc_bw_dot = -1;
opcode_t oc_bw_dot_nt = -1;
opcode_t oc_bw_dot_tn = -1;
opcode_t oc_bw_dot_c = -1;
opcode_t oc_bw_dot_nt_c = -1;
opcode_t oc_bw_dot_tn_c = -1;

#pragma warning(disable : 4190 4297)  // disable warning about C linkage of shape_t, and about throwing exceptions from C functions

extern "C" {

SM_DLLEXPORT void api_gradstep(smat* P, const smat* dP, const smat* drate, smat* mP, const smat* mrate)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_gstep,P->as_arg(),dP->as_arg(),drate->as_arg(),mP->as_arg(),mrate->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_gradstep_nesterov1(const smat* P, const smat* mP, const smat* mrate)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_gstepn1,P->as_arg(),mP->as_arg(),mrate->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_gradstep_nesterov2(smat* P, const smat* dP, const smat* drate, smat* mP, const smat* mrate)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_gstepn2,P->as_arg(),dP->as_arg(),drate->as_arg(),mP->as_arg(),mrate->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_madd_bcast(const smat* A, const smat* b, usize_t k, smat* dst)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_maddbc,A->as_arg(),b->as_arg(),carray(k),dst->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_maskout(const smat* M, smat* A)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_maskout,M->as_arg(),A->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_calc_zmask(const smat* Z, smat* M)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_calczmask,Z->as_arg(),M->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_dropout_fp_tr(const smat* X, const smat* rate, smat* Z, smat* M, int matchrows)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_drop_fp_tr,X->as_arg(),rate->as_arg(),Z->as_arg(),M->as_arg(),carray((bool)matchrows));
	SM_API_CATCH
}

SM_DLLEXPORT void api_dropout_fp_te(const smat* X, const smat* rate, smat* Z)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_drop_fp_te,X->as_arg(),rate->as_arg(),Z->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_dropout_bp_tr(const smat* dZ, const smat* M, smat* dX)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_drop_bp_tr,dZ->as_arg(),M->as_arg(),dX->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_dropout_bp_te(const smat* dZ, const smat* rate, smat* dX)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_drop_bp_te,dZ->as_arg(),rate->as_arg(),dX->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_blockwise_dot(int nblock, const smat* X, const smat* W, smat* Z)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_bw_dot, carray(nblock), X->as_arg(), W->as_arg(), Z->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_blockwise_dot_nt(int nblock, const smat* dZ, const smat* W, smat* dX)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_bw_dot_nt, carray(nblock), dZ->as_arg(), W->as_arg(), dX->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_blockwise_dot_tn(int nblock, const smat* X, const smat* dZ, smat* dW)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_bw_dot_tn, carray(nblock), X->as_arg(), dZ->as_arg(), dW->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_blockwise_dot_combined(int nblock, int Xoffset, int Xbroadcast, const smat* X, const smat* W, smat* Z)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_bw_dot_c, carray(nblock), carray((Xoffset << 1) | Xbroadcast), X->as_arg(), W->as_arg(), Z->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_blockwise_dot_nt_combined(int nblock, int Xoffset, int Xbroadcast, smat* dX, const smat* W, const smat* dZ)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_bw_dot_nt_c, carray(nblock), carray((Xoffset << 1) | Xbroadcast), dX->as_arg(), W->as_arg(), dZ->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void api_blockwise_dot_tn_combined(int nblock, int Xoffset, int Xbroadcast, const smat* X, smat* dW, const smat* dZ)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_bw_dot_tn_c, carray(nblock), carray((Xoffset << 1) | Xbroadcast), X->as_arg(), dW->as_arg(), dZ->as_arg());
	SM_API_CATCH
}

SM_DLLEXPORT void register_ext()
{
	oc_gstep = add_instruction("gstep",
			iprop_match_all,
			aprop_in |aprop_out|aprop_float,      // (P)
			aprop_in |          aprop_float,      // (dP)
			aprop_in |          aprop_float,      // (drate)
			aprop_in |aprop_out|aprop_float,      // (mP)
			aprop_in |          aprop_float       // (mrate)
		);

	oc_gstepn1 = add_instruction("gstepn1",
			iprop_match_all,
			aprop_in |aprop_out|aprop_float,      // (P)
			aprop_in |          aprop_float,      // (mP)
			aprop_in |          aprop_float       // (mrate)
		);

	oc_gstepn2 = add_instruction("gstepn2",
			iprop_match_all,
			aprop_in |aprop_out|aprop_float,      // (P)
			aprop_in |          aprop_float,      // (dP)
			aprop_in |          aprop_float,      // (drate)
			aprop_in |aprop_out|aprop_float,      // (mP)
			aprop_in |          aprop_float       // (mrate)
		);

	oc_maddbc = add_instruction("maddbc",
			iprop_none,
			aprop_in |          aprop_float,      // (A)
			aprop_in |          aprop_float,      // (b)
			aprop_in |          aprop_uint,       // (k)
			aprop_in |aprop_out|aprop_float       // (dst)
		);

	oc_maskout = add_instruction("maskout",
			iprop_none,
			aprop_in |          aprop_bool,       // (M)
			aprop_in |aprop_out|aprop_float       // (A)
		);

	oc_calczmask = add_instruction("calczmask",
			iprop_none,
			aprop_in |          aprop_float,     // (Z)
			aprop_in |aprop_out|aprop_bool       // (M)
		);

	oc_drop_fp_tr = add_instruction("drop_fp_tr",
			iprop_none,
			aprop_in |          aprop_float,      // (X)
			aprop_in |          aprop_float,      // (rate)
			          aprop_out|aprop_float,      // (Z)
			          aprop_out|aprop_bool,       // (M)
			aprop_in |          aprop_bool        // (matchrows)
		);

	oc_drop_fp_te = add_instruction("drop_fp_te",
			iprop_none,
			aprop_in |          aprop_float,      // (X)
			aprop_in |          aprop_float,      // (rate)
			          aprop_out|aprop_float       // (Z)
		);

	oc_drop_bp_tr = add_instruction("drop_bp_tr",
			iprop_none,
			aprop_in |          aprop_float,      // (dZ)
			aprop_in |          aprop_bool,       // (M)
			          aprop_out|aprop_float       // (dX)
		);

	oc_drop_bp_te = add_instruction("drop_bp_te",
			iprop_match_all,
			aprop_in |          aprop_float,      // (dZ)
			aprop_in |          aprop_float,      // (rate)
			          aprop_out|aprop_float       // (dX)
		);

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

	oc_bw_dot_c = add_instruction("bw_dot_c",
			iprop_none,
			aprop_in |          aprop_int,        // (nblock)
			aprop_in |          aprop_int,        // (Xoffset,Xbroadcast)
			aprop_in |          aprop_float,      // (X)
			aprop_in |          aprop_float,      // (W)
			aprop_in |aprop_out|aprop_float       // (Z)
		);

	oc_bw_dot_nt_c = add_instruction("bw_dot_nt_c",
			iprop_none,
			aprop_in |          aprop_int,        // (nblock)
			aprop_in |          aprop_int,        // (Xoffset,Xbroadcast)
			          aprop_out|aprop_float,      // (dX)
			aprop_in |          aprop_float,      // (W)
			aprop_in |          aprop_float       // (dZ)
		);
	
	oc_bw_dot_tn_c = add_instruction("bw_dot_tn_c",
			iprop_none,
			aprop_in |          aprop_int,        // (nblock)
			aprop_in |          aprop_int,        // (Xoffset,Xbroadcast)
			aprop_in |          aprop_float,      // (X)
			          aprop_out|aprop_float,      // (dW)
			aprop_in |          aprop_float       // (dZ)
		);

	add_instruction_impl(cuda_uuid,oc_gstep  ,execute_gradstep          ,validate_gradstep);
	add_instruction_impl(cuda_uuid,oc_gstepn1,execute_gradstep_nesterov1,validate_gradstep_nesterov1);
	add_instruction_impl(cuda_uuid,oc_gstepn2,execute_gradstep_nesterov2,validate_gradstep_nesterov2);
	add_instruction_impl(cuda_uuid,oc_maddbc ,execute_madd_bcast        ,validate_madd_bcast);
	add_instruction_impl(cuda_uuid,oc_maskout,execute_maskout           ,validate_maskout);
	add_instruction_impl(cuda_uuid,oc_calczmask,execute_calc_zmask      ,validate_calc_zmask);
	add_instruction_impl(cuda_uuid,oc_drop_fp_tr,execute_dropout_fp_tr,validate_dropout_fp_tr);
	add_instruction_impl(cuda_uuid,oc_drop_fp_te,execute_dropout_fp_te,validate_dropout_fp_te);
	add_instruction_impl(cuda_uuid,oc_drop_bp_tr,execute_dropout_bp_tr,validate_dropout_bp_tr);
	add_instruction_impl(cuda_uuid,oc_drop_bp_te,execute_dropout_bp_te,validate_dropout_bp_te);
	add_instruction_impl(cuda_uuid,oc_bw_dot    ,execute_blockwise_dot   ,validate_blockwise_dot);
	add_instruction_impl(cuda_uuid,oc_bw_dot_nt ,execute_blockwise_dot_nt,validate_blockwise_dot_nt);
	add_instruction_impl(cuda_uuid,oc_bw_dot_tn ,execute_blockwise_dot_tn,validate_blockwise_dot_tn);
	add_instruction_impl(cuda_uuid,oc_bw_dot_c  ,execute_blockwise_dot_combined,validate_blockwise_dot_combined);
	add_instruction_impl(cuda_uuid,oc_bw_dot_nt_c,execute_blockwise_dot_nt_combined,validate_blockwise_dot_nt_combined);
	add_instruction_impl(cuda_uuid,oc_bw_dot_tn_c,execute_blockwise_dot_tn_combined,validate_blockwise_dot_tn_combined);
}

} // extern "C"
