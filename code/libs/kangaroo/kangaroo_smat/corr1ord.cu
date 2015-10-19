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
#include "corr1ord.cuh"

// The quantity _n in the corr1ord kernel indicates the number of outputs that each
// block is responsible for in the *sequence* dimension. I.e. if _n=32 then each block
// generates outputs at 32 feature map positions.
//
// So how should _n be determined?...
// Fact 1a:
//    Each block allocates bdx*wdy*sizeof(uint32_t) bytes of shared memory in _X
//    to store a chunk of the sequence, so clearly _n <= bdx*wdy*sizeof(uint32_t).
// Fact 1b:
//    Since the output at position i needs sequence elements X[i:i+m-1], 
//    we cannot generate outputs for the last few positions in _X
//    then we actually require _n <= bdx*sizeof(uint32_t)-m+1
// Fact 1c:
//    Because sizeof(uint32_t) bytes of _X are loaded at a time, 
//    _n must be a multiple of sizeof(uint32_t).
// Fact 2a:
//    The innermost loop operates on sub-chunks of size __n, while 
//    loading __n+__m-1 sequence elements into register memory __X.
// Fact 2b:
//    If the number of outputs per block is a multiple of __n, then
//    only the last block will need to check for boundary conditions in n.
//
// Conclusion:
//    The number _n of outputs per block should be the largest multiple
//    of __n that does not violate the following constraint:
//       _n <= bdx*wdy*sizeof(uint32_t) - m + 1 
//    The largest multiple of both __n and sizeof(uint32_t) satisfying this is:
//       _n = ((bdx*wdy*sizeof(uint32_t) - m + 1)/lcm) * lcm
//    where lcm is the largest common multiple of __n and sizeof(uint32_t).
//
unsigned corr1ord_compute_n(unsigned bdx, unsigned wdy, unsigned m, unsigned __n)
{
	unsigned max_n = bdx*wdy*sizeof(uint32_t)-m+1;
	unsigned lcm = _SM::lcm(sizeof(uint32_t),__n);
	unsigned _n = (max_n/lcm)*lcm; // find largest integer <= len that is a multiple of both sizeof(uint32_t) and of __n
	return _n;
}



template <usize_t nchannel, unsigned __m, bool check_m, bool loop_m, typename float_t>
void launch_corr1ord_spec2(cudaStream_t stream, 
                          const float_t*  W,  usize_t m, usize_t nfilter,
                          const uint8_t*  X,  usize_t n, 
                                float_t*  Z)
{
	const unsigned bdx = 48;
	const unsigned wdy = 1;
	const unsigned __n = 32;
	//const unsigned bdx = 16;     // 16,32 for small; 32,32 for large
	//const unsigned wdy = 1;
	//const unsigned __n = 32;
	const unsigned _n  = corr1ord_compute_n(bdx,wdy,m,__n);

	static bool s_was_cache_setup = false;
	if (!s_was_cache_setup) {
		s_was_cache_setup = true;
		cudaFuncSetCacheConfig(&corr1ord_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,false,float_t>,cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(&corr1ord_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,true ,float_t>,cudaFuncCachePreferL1);
	}

	// TODO: assert no corner cases, like very small filters, too-small block size, etc.
	dim3 bdim(bdx,1);
	dim3 gdim(divup(nfilter,bdx),divup(n,_n));
	if (gdim.y == 1 || (n % _n == 0)) {
		corr1ord_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,true ><<<gdim,bdim,0,stream>>>(W,m,nfilter,X,n,_n,Z);
	} else {
		dim3 gdim0(gdim.x,gdim.y-1); // all but the last row of blocks
		dim3 gdim1(gdim.x,1);        // only the last row of blocks
		uindex_t i0 = gdim0.y*_n;    // index of first sequence element processed by final row of blocks
		corr1ord_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,false><<<gdim0,bdim,0,stream>>>(W,m,nfilter,X,n,_n,Z);
		corr1ord_kernel<bdx,wdy,__m,__n,nchannel,check_m,loop_m,true ><<<gdim1,bdim,0,stream>>>(W,m,nfilter,X+i0,n-i0,_n,Z+i0*nfilter);
	}
}

template <usize_t nchannel, typename float_t>
void launch_corr1ord_spec1(cudaStream_t stream, 
                          const float_t*  W,  usize_t m, usize_t nfilter,
                          const uint8_t*  X,  usize_t n, 
                                float_t*  Z)
{
	const unsigned __m = 4;
	if (m % __m == 0) launch_corr1ord_spec2<nchannel,__m,false,true>(stream,W,m,nfilter,X,n,Z);
	else              launch_corr1ord_spec2<nchannel,__m,true ,true>(stream,W,m,nfilter,X,n,Z);
}

template <typename float_t>
void launch_corr1ord_spec0(cudaStream_t stream, 
                           const float_t*  W,  usize_t m, usize_t nfilter,
                           const uint8_t*  X,  usize_t n, usize_t nchannel, 
                                 float_t*  Z)
{
	switch (nchannel) {
	case 4: launch_corr1ord_spec1<4>(stream,W,m,nfilter,X,n,Z); break;
	// TODO: can do 80 channel for proteins
	default: SM_ERROR("NotImplementedError: The given number of channels does not have a matching pre-compiled kernel.\n");
	}
}

void launch_corr1ord(cudaStream_t stream, dtype_t dtype,
                     const void*     W,  usize_t m, usize_t nfilter,
                     const uint8_t*  X,  usize_t n, usize_t nchannel, 
                           void*     Z)
{
	if (n == 0)
		return;
	switch (dtype) {
	case f32: launch_corr1ord_spec0(stream,(const float* )W,m,nfilter,X,n,nchannel,(float* )Z); break;
	case f64: launch_corr1ord_spec0(stream,(const double*)W,m,nfilter,X,n,nchannel,(double*)Z); break;
	default: SM_UNREACHABLE();
	}
}
