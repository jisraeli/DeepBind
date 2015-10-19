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
#include <smat/vm/instruction_db.h>
#include <vector>
using namespace sm;
using namespace std;

void launch_blockwise_op(const char** ptrs, dtype_t dtype, int nblock, 
						 int m, int n, int k,
						 int ldA, int ldB, int ldC, 
						 double alpha, double beta, 
						 cublasOperation_t opA, cublasOperation_t opB)
{
	if (opA == CUBLAS_OP_N && opB == CUBLAS_OP_T)
		swap(m,k);
	else if (opA == CUBLAS_OP_T && opB == CUBLAS_OP_N)
		swap(n,k);
	SM_ASSERTMSG(opA != CUBLAS_OP_T || opB != CUBLAS_OP_T, "Not implemented")

	// Now copy the subarray pointers to the 
	heap_alloc dev_arrays_alloc = thread_cudactx().heap().alloc(sizeof(void*) * nblock * 3);
	void** _ptrs = (void**)dev_arrays_alloc.addr;
	cudaMemcpyAsync(_ptrs, &ptrs[0], 3*nblock*sizeof(void*), cudaMemcpyHostToDevice);

	if (dtype == f32) {
		float _alpha = alpha, _beta = beta;
		ccb(SgemmBatched, thread_cudactx().cublas(), opB, opA, m, n, k, &_alpha,
			(const float**)&_ptrs[1*nblock], ldB,
			(const float**)&_ptrs[0*nblock], ldA, &_beta,
			(      float**)&_ptrs[2*nblock], ldC, nblock);
	} else {
		ccb(DgemmBatched, thread_cudactx().cublas(), opB, opA, m, n, k, &alpha,
			(const double**)&_ptrs[1*nblock], ldB,
			(const double**)&_ptrs[0*nblock], ldA, &beta,
			(      double**)&_ptrs[2*nblock], ldC, nblock);
	}

	thread_cudactx().heap().free(dev_arrays_alloc);
}

void launch_blockwise_dot(cudaStream_t stream, dtype_t dtype,
						  int nblock, int n, int m, int k, bool broadcast_x,
						  const void* X, const void* W, void* Z)
{
	if (n == 0 || m == 0 || k == 0)
		return;

	// X[:,i] is n x k
	// W[i,:] is k x m
	// Z[:,i] is n x m
	// ... so ...
	// X is n x (k*nblock)
	// W is (k*nblock) x m
	// Z is n x (m*nblock)
	// ... and so our strides are ...
	int ldx = k*(broadcast_x ? 1 : nblock);
	int ldw = m;
	int ldz = m*nblock;

	// Now copy the subarray pointer addresses into device memory
	vector<const char*> ptrs(3*nblock);
	for (int b = 0; b < nblock; ++b) {
		ptrs[0*nblock+b] = (const char*)X + dtype_size(dtype) * b * (k) * (broadcast_x ? 0 : 1); // if broadcast_x, X is actually just n x k, and we need to re-use X
		ptrs[1*nblock+b] = (const char*)W + dtype_size(dtype) * b * (k*m);
		ptrs[2*nblock+b] = (      char*)Z + dtype_size(dtype) * b * (m);
	}

	launch_blockwise_op(&ptrs[0], dtype, nblock,
			m, n, k, ldx, ldw, ldz,
			1.0, 0.0, CUBLAS_OP_N, CUBLAS_OP_N);
}


/////////////////////////////////

void launch_blockwise_dot_nt(cudaStream_t stream, dtype_t dtype,
						     int nblock, int n, int m, int k,
						     const void* dZ, const void* W, void* dX)
{
	if (n == 0 || m == 0 || k == 0)
		return;

	int ldx = k*nblock;
	int ldw = m;
	int ldz = m*nblock;

	vector<const char*> ptrs(3*nblock);
	for (int b = 0; b < nblock; ++b) {
		ptrs[2*nblock+b] = (const char*)dX + dtype_size(dtype) * b * (k);
		ptrs[1*nblock+b] = (const char*)W  + dtype_size(dtype) * b * (k*m);
		ptrs[0*nblock+b] = (      char*)dZ + dtype_size(dtype) * b * (m);
	}

	launch_blockwise_op(&ptrs[0], dtype, nblock,
			m, n, k, ldz, ldw, ldx,
			1.0, 0.0, CUBLAS_OP_N, CUBLAS_OP_T);
}

void launch_blockwise_dot_tn(cudaStream_t stream, dtype_t dtype,
						     int nblock, int n, int m, int k, bool broadcast_x,
						     const void* X, const void* dZ, void* dW)
{
	if (n == 0 || m == 0 || k == 0)
		return;

	int ldx = k*(broadcast_x ? 1 : nblock);
	int ldw = m;
	int ldz = m*nblock;

	vector<const char*> ptrs(3*nblock);
	for (int b = 0; b < nblock; ++b) {
		ptrs[0*nblock+b] = (const char*)X  + dtype_size(dtype) * b * (k) * (broadcast_x ? 0 : 1); // if broadcast_x, X is actually just n x k, and we need to re-use X
		ptrs[2*nblock+b] = (const char*)dW + dtype_size(dtype) * b * (k*m);
		ptrs[1*nblock+b] = (      char*)dZ + dtype_size(dtype) * b * (m);
	}

	launch_blockwise_op(&ptrs[0], dtype, nblock,
			m, n, k, ldx, ldz, ldw,
			1.0, 0.0, CUBLAS_OP_T, CUBLAS_OP_N);
}


void launch_blockwise_dot_combined(cudaStream_t stream, dtype_t dtype,
                                   int nblock, int Xoffset, bool Xbroadcast,
                                   int n, int m, int k, int p,
                                   const void* X, const void* W, void* Z)
{
	if (n == 0 || m == 0 || k == 0)
		return;

	// X[:,i] is n x k
	// W[i,:] is k x m
	// Z[:,i] is n x m
	// ... so ...
	// X is n x (k*nblock)
	// W is (p*nblock) x m
	// Z is n x (m*nblock)
	// ... and so our strides are ...
	int ldx = k*(Xbroadcast ? 1 : nblock);
	int ldw = m;
	int ldz = m*nblock;

	// Now copy the subarray pointer addresses into device memory
	vector<const char*> ptrs(3*nblock);
	for (int b = 0; b < nblock; ++b) {
		ptrs[0*nblock+b] = (const char*)X + dtype_size(dtype) * (b*k*(Xbroadcast ? 0 : 1)); // if broadcast_x, X is actually just n x k, and we need to re-use X
		ptrs[1*nblock+b] = (const char*)W + dtype_size(dtype) * (b*p*m + Xoffset*m);
		ptrs[2*nblock+b] = (      char*)Z + dtype_size(dtype) * (b*m);
	}

	double beta = Xoffset == 0 ? 0.0 : 1.0;
	launch_blockwise_op(&ptrs[0], dtype, nblock, 
		m, n, k, ldx, ldw, ldz, 
		1.0, beta, CUBLAS_OP_N, CUBLAS_OP_N);
}


void launch_blockwise_dot_nt_combined(cudaStream_t stream, dtype_t dtype,
                                      int nblock, int Xoffset, 
                                      int n, int m, int k, int p,
                                      void* dX, const void* W, const void* dZ)
{
	if (n == 0 || m == 0 || k == 0)
		return;

	// X[:,i] is n x k
	// W[i,:] is k x m
	// Z[:,i] is n x m
	// ... so ...
	// X is n x (k*nblock)
	// W is (p*nblock) x m
	// Z is n x (m*nblock)
	// ... and so our strides are ...
	int ldx = k*nblock;
	int ldw = m;
	int ldz = m*nblock;

	// Now copy the subarray pointer addresses into device memory
	vector<const char*> ptrs(3*nblock);
	for (int b = 0; b < nblock; ++b) {
		ptrs[2*nblock+b] = (      char*)dX + dtype_size(dtype) * (b*k);
		ptrs[1*nblock+b] = (const char*)W  + dtype_size(dtype) * (b*p*m + Xoffset*m);
		ptrs[0*nblock+b] = (const char*)dZ + dtype_size(dtype) * (b*m);
	}

	launch_blockwise_op(&ptrs[0], dtype, nblock, 
		m, n, k, ldz, ldw, ldx, 
		1.0, 0.0, CUBLAS_OP_N, CUBLAS_OP_T);
}





void launch_blockwise_dot_tn_combined(cudaStream_t stream, dtype_t dtype,
                                      int nblock, int Xoffset,  bool Xbroadcast,
                                      int n, int m, int k, int p,
                                      const void* X, void* dW, const void* dZ)
{
	if (n == 0 || m == 0 || k == 0)
		return;

	// X[:,i] is n x k
	// W[i,:] is k x m
	// Z[:,i] is n x m
	// ... so ...
	// X is n x (k*nblock)
	// W is (p*nblock) x m
	// Z is n x (m*nblock)
	// ... and so our strides are ...
	int ldx = k*(Xbroadcast ? 1 : nblock);
	int ldw = m;
	int ldz = m*nblock;

	// Now copy the subarray pointer addresses into device memory
	vector<const char*> ptrs(3*nblock);
	for (int b = 0; b < nblock; ++b) {
		ptrs[0*nblock+b] = (const char*)X  + dtype_size(dtype) * (b*k*(Xbroadcast ? 0 : 1)); // if broadcast_x, X is actually just n x k, and we need to re-use X
		ptrs[2*nblock+b] = (      char*)dW + dtype_size(dtype) * (b*p*m + Xoffset*m);
		ptrs[1*nblock+b] = (const char*)dZ + dtype_size(dtype) * (b*m);
	}

	launch_blockwise_op(&ptrs[0], dtype, nblock, 
		m, n, k, ldx, ldz, ldw,
		1.0, 0.0, CUBLAS_OP_T, CUBLAS_OP_N);
}

















