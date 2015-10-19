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
#include <vector>
#include <list>
using namespace sm;
using namespace std;


struct batch_dot_args {
	batch_dot_args(const void* A, const void* B, void* C, dtype_t dt,
		int n, int m, int k, int p, bool broadcast_A)
		: A(A), B(B), C(C), dt(dt),
		  n(n), m(m), k(k), p(p), broadcast_A(A),
		  _ptrs(0)
	{ }

	void free_ptrs()
	{
		// This is deliberately not in destructor, so that a program shutting down doesn't try to get
		// a handle to cuda context, since device memory will be freed when program exits anyway.
		if (_ptrs_alloc.addr)
			thread_cudactx().heap().free(_ptrs_alloc);
	}

	void** A_ptrs() { ensure_ptrs_allocated(); return &_ptrs[0*p]; }
	void** B_ptrs() { ensure_ptrs_allocated(); return &_ptrs[1*p]; }
	void** C_ptrs() { ensure_ptrs_allocated(); return &_ptrs[2*p]; }

private:

	void ensure_ptrs_allocated() {
		if (_ptrs)
			return;

		int dtsize = (int)dtype_size(dt);

		// First, construct the array of device pointers, storing the pointers
		// in a contiguous chunk of host memory.
		_ptrs_host.resize(3*p);
		for (int i = 0; i < p; ++i) {
			_ptrs_host[0*p+i] = (void*)((size_t)A + dtsize * i * (n*k) * (broadcast_A ? 0 : 1)); // if broadcast_A, A is actually just a single n x k, and we need to re-use the base address of A for all allocations
			_ptrs_host[1*p+i] = (void*)((size_t)B + dtsize * i * (k*m));
			_ptrs_host[2*p+i] = (void*)((size_t)C + dtsize * i * (n*m));
		}

		heap_alloc _ptrs_alloc = thread_cudactx().heap().alloc(sizeof(void*) * p * 3);
		_ptrs = (void**)_ptrs_alloc.addr;

		// Now copy _ptrs_host to the device and leave it there until this batch_dot_args instance is destroyed
		cudaMemcpyAsync(_ptrs, &_ptrs_host[0], 3*p*sizeof(void*), cudaMemcpyHostToDevice);
	}



	const void* A;
	const void* B;
	      void* C;
	dtype_t dt;
	int m, n, k, p;
	bool broadcast_A;
	
	heap_alloc _ptrs_alloc; // bookkeeping for device heap allocation of _ptrs
	void** _ptrs; // device pointer to a small block of memory containing
	              // pointers (3*p pointers to be exact) to all the 
	              // individual A[i], B[i], C[i] matrices for i=0..p-1
	vector<void*> _ptrs_host; // local host copy 

	friend bool operator==(const batch_dot_args& a, const batch_dot_args& b);
};


bool operator==(const batch_dot_args& a, const batch_dot_args& b)
{
	return a.A == b.A && a.B == b.B && a.C == b.C &&
		   a.m == b.m && a.n == b.n && a.k == b.k && a.p == b.p &&
		   a.dt == b.dt && a.broadcast_A == b.broadcast_A;
}

list<batch_dot_args> g_batch_dot_arg_cache;

// Registers 
batch_dot_args& cache_batch_dot_args(const void* A, const void* B, void* C, dtype_t dt,
		int n, int m, int k, int p, bool broadcast_A)
{
	batch_dot_args args(A, B, C, dt, n, m, k, p, broadcast_A);
	for (auto& entry : g_batch_dot_arg_cache) {
		if (args == entry)
			return entry;
	}
	g_batch_dot_arg_cache.push_front(args);
	if (g_batch_dot_arg_cache.size() > 30) {
		g_batch_dot_arg_cache.back().free_ptrs();
		g_batch_dot_arg_cache.pop_back();
	}
	return g_batch_dot_arg_cache.front();
}


void launch_batch_dot_general(const void* A, const void* B, void* C, dtype_t dt,
                              int n, int m, int k, int p, bool broadcast_A,
                              double alpha, double beta, 
                              bool transA, bool transB)
{
	if (n == 0 || m == 0 || k == 0)
		return;

	// Cublas expects strides to be *before* tranpose operation
	int ldA = k;
	int ldB = m;
	int ldC = m;

	// Cublas expects m,n,k to be the dimensions *after* transpose operation, not dimensions before.
	cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
	if      (opA == CUBLAS_OP_N && opB == CUBLAS_OP_T) swap(n, k);
	else if (opA == CUBLAS_OP_T && opB == CUBLAS_OP_N) swap(m, k);
	SM_ASSERTMSG(opA != CUBLAS_OP_T || opB != CUBLAS_OP_T, "Double transpose not implemented")

	batch_dot_args& args = cache_batch_dot_args(A, B, C, dt, n, m, k, p, broadcast_A);

	if (dt == f32) {
		float _alpha = (float)alpha, _beta = (float)beta;
		ccb(SgemmBatched, thread_cudactx().cublas(), opB, opA, n, m, k, &_alpha,
			(const float**)args.B_ptrs(), ldB,
			(const float**)args.A_ptrs(), ldA, &_beta,
			(      float**)args.C_ptrs(), ldC, p);
	} else {
		ccb(DgemmBatched, thread_cudactx().cublas(), opB, opA, n, m, k, &alpha,
			(const double**)args.B_ptrs(), ldB,
			(const double**)args.A_ptrs(), ldA, &beta,
			(      double**)args.C_ptrs(), ldC, p);
	}
}



