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
#include "corr1ord_bprop.cuh"

unsigned corr1ord_compute_n(unsigned bdx, unsigned wdy, unsigned m, unsigned __n);

template <usize_t nchannel, unsigned __m, bool check_m, bool loop_m, typename float_t>
void launch_corr1ord_bprop_W_spec2(cudaStream_t stream, 
                                       float_t* dW,  usize_t m, usize_t nfilter,
                                 const uint8_t*  X,  usize_t n, 
                                 const float_t* dZ)
{
	const unsigned bdx = 48;
	const unsigned wdy = 1;
	const unsigned __n = 32;
	const unsigned _n  = corr1ord_compute_n(bdx,wdy,m,__n);
	
	static bool s_was_cache_setup = false;
	if (!s_was_cache_setup) {
		s_was_cache_setup = true;
		cudaFuncSetCacheConfig(&corr1ord_bprop_W_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,false,float_t>,cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(&corr1ord_bprop_W_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,true ,float_t>,cudaFuncCachePreferL1);
	}

	// TODO: assert no corner cases, like very small filters, too-small block size, etc.
	dim3 bdim(bdx,1);
	dim3 gdim(divup(nfilter,bdx),divup(n,_n));
	if (gdim.y == 1 || (n % _n == 0)) {
		corr1ord_bprop_W_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,true ><<<gdim,bdim,0,stream>>>(dW,m,nfilter,X,n,_n,dZ);
	} else {
		dim3 gdim0(gdim.x,gdim.y-1); // all but the last row of blocks
		dim3 gdim1(gdim.x,1);        // only the last row of blocks
		uindex_t i0 = gdim0.y*_n;    // index of first sequence element processed by final row of blocks
		corr1ord_bprop_W_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,false><<<gdim0,bdim,0,stream>>>(dW,m,nfilter,X,n,_n,dZ);
		corr1ord_bprop_W_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,true ><<<gdim1,bdim,0,stream>>>(dW,m,nfilter,X+i0,n-i0,_n,dZ+i0*nfilter);
	}
}


template <usize_t nchannel, typename float_t>
void launch_corr1ord_bprop_W_spec1(cudaStream_t stream, 
                                float_t* dW,  usize_t m, usize_t nfilter,
                          const uint8_t*  X,  usize_t n, 
                          const float_t* dZ)
{
	const unsigned __m = 4;
	if (m % __m == 0) launch_corr1ord_bprop_W_spec2<nchannel,__m,false,true>(stream,dW,m,nfilter,X,n,dZ);
	else              launch_corr1ord_bprop_W_spec2<nchannel,__m,true ,true>(stream,dW,m,nfilter,X,n,dZ);
}

template <typename float_t>
void launch_corr1ord_bprop_W_spec0(cudaStream_t stream, 
                                       float_t* dW,  usize_t m, usize_t nfilter,
                                 const uint8_t*  X,  usize_t n, usize_t nchannel, 
                                 const float_t* dZ)
{
	switch (nchannel) {
	case 4: launch_corr1ord_bprop_W_spec1<4>(stream,dW,m,nfilter,X,n,dZ); break;
	// TODO: can do 80 channel for proteins
	default: SM_ERROR("NotImplementedError: The given number of channels does not have a matching pre-compiled kernel.\n");
	}
}

void launch_corr1ord_bprop_W(cudaStream_t stream, dtype_t dtype,
                                 void*    dW,  usize_t m, usize_t nfilter,
                           const uint8_t*  X,  usize_t n, usize_t nchannel, 
                           const void*    dZ)
{
	if (n == 0)
		return;
	switch (dtype) {
	case f32: launch_corr1ord_bprop_W_spec0(stream,(float* )dW,m,nfilter,X,n,nchannel,(const float* )dZ); break;
	case f64: launch_corr1ord_bprop_W_spec0(stream,(double*)dW,m,nfilter,X,n,nchannel,(const double*)dZ); break;
	default: SM_UNREACHABLE();
	}
}








////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


template <usize_t nchannel, unsigned __m, bool check_m, bool loop_m, typename float_t>
void launch_corr1ord_bprop_X_spec2(cudaStream_t stream, 
                                       float_t* dX,  usize_t n,
                                 const float_t*  W,  usize_t m, usize_t nfilter,
                                 const float_t* dZ)
{
	const unsigned bdx = 48;
	const unsigned wdy = 1;
	const unsigned __n = 32;
	const unsigned _n  = corr1ord_compute_n(bdx,wdy,m,__n);
	
	static bool s_was_cache_setup = false;
	if (!s_was_cache_setup) {
		s_was_cache_setup = true;
		cudaFuncSetCacheConfig(&corr1ord_bprop_X_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,false,float_t>,cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(&corr1ord_bprop_X_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,true ,float_t>,cudaFuncCachePreferL1);
	}

	// TODO: assert no corner cases, like very small filters, too-small block size, etc.
	dim3 bdim(bdx,1);
	dim3 gdim(divup(nfilter,bdx),divup(n,_n));
	if (gdim.y == 1 || (n % _n == 0)) {
		corr1ord_bprop_X_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,true ><<<gdim,bdim,0,stream>>>(dX,n,_n,W,m,nfilter,dZ);
	} else {
		dim3 gdim0(gdim.x,gdim.y-1); // all but the last row of blocks
		dim3 gdim1(gdim.x,1);        // only the last row of blocks
		uindex_t i0 = gdim0.y*_n;    // index of first sequence element processed by final row of blocks
		corr1ord_bprop_X_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,false><<<gdim0,bdim,0,stream>>>(dX,n,_n,W,m,nfilter,dZ);
		corr1ord_bprop_X_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,true ><<<gdim1,bdim,0,stream>>>(dX+i0,n-i0,_n,W,m,nfilter,dZ+i0*nfilter);
	}
}


template <usize_t nchannel, typename float_t>
void launch_corr1ord_bprop_X_spec1(cudaStream_t stream, 
                                float_t* dX,  usize_t n, 
                          const float_t*  W,  usize_t m, usize_t nfilter,
                          const float_t* dZ)
{
	const unsigned __m = 4;
	if (m % __m == 0) launch_corr1ord_bprop_X_spec2<nchannel,__m,false,true>(stream,dX,n,W,m,nfilter,dZ);
	else              launch_corr1ord_bprop_X_spec2<nchannel,__m,true ,true>(stream,dX,n,W,m,nfilter,dZ);
}



template <typename float_t>
void launch_corr1ord_bprop_X_spec0(cudaStream_t stream, 
                                       float_t* dX,  usize_t n, usize_t nchannel, 
                                 const float_t*  W,  usize_t m, usize_t nfilter,
                                 const float_t* dZ)
{
	/*
	switch (nchannel) {
	case 4: launch_corr1ord_bprop_X_spec1<4>(stream,dX,n,W,m,nfilter,dZ); break;
	// TODO: can do 80 channel for proteins
	default: SM_ERROR("NotImplementedError: The given number of channels does not have a matching pre-compiled kernel.\n");
	}*/

	usize_t dXsize = n*nchannel;
	usize_t Wsize  = m*nfilter*nchannel;
	usize_t dZsize = n*nfilter;
	float_t* _dX = new float_t[dXsize];
	float_t* _W  = new float_t[Wsize];
	float_t* _dZ = new float_t[dZsize];
	cudaMemcpyAsync(_dX,dX,dXsize*sizeof(float_t),cudaMemcpyDeviceToHost,stream);
	cudaMemcpyAsync( _W, W, Wsize*sizeof(float_t),cudaMemcpyDeviceToHost,stream);
	cudaMemcpyAsync(_dZ,dZ,dZsize*sizeof(float_t),cudaMemcpyDeviceToHost,stream);
	cudaDeviceSynchronize();

	for (usize_t i = 0; i < n; ++i) {
		for (usize_t j = 0; j < nfilter; ++j) {
			for (usize_t k = 0; k < m; ++k) {
				if (i < k) 
					continue;
				for (usize_t c = 0; c < nchannel; ++c) {
					_dX[i*nchannel + c] += _dZ[(i-k)*nfilter + j] * _W[(nchannel*k+c)*nfilter + j];
				}
			}
		}
	}

	cudaMemcpyAsync(dX,_dX,dXsize*sizeof(float_t),cudaMemcpyHostToDevice,stream);
	cudaDeviceSynchronize();

	delete[] _dZ;
	delete[] _W;
	delete[] _dX;
}


void launch_corr1ord_bprop_X(cudaStream_t stream, dtype_t dtype,
                                 void*  dX, usize_t n, usize_t nchannel, 
                           const void*  W, usize_t m, usize_t nfilter,
                           const void*  dZ)
{
	if (n == 0)
		return;
	switch (dtype) {
	case f32: launch_corr1ord_bprop_X_spec0(stream,(float* )dX,n,nchannel,(float* )W,m,nfilter,(const float* )dZ); break;
	case f64: launch_corr1ord_bprop_X_spec0(stream,(double*)dX,n,nchannel,(double*)W,m,nfilter,(const double*)dZ); break;
	default: SM_UNREACHABLE();
	}
}
