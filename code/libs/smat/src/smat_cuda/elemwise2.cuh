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
#ifndef __SM_CUDA_ELEMWISE2_H__
#define __SM_CUDA_ELEMWISE2_H__

#include <smat_cuda/launch_util.h>
#include <smat_cuda/cuda_context.h>
#include <smat/vm/instruction.h>
#include <base/util.h>

SM_NAMESPACE_BEGIN

#define DEF_AB_GENERIC(name,f) \
	template <typename T0, typename T1>                                                  \
	struct k_##name {                                                                    \
		typedef T0 A;                                                                    \
		typedef T1 B;                                                                    \
		SM_DEVICE_INLINE static void apply(const A* a, uindex_t i, B* b, uindex_t j) { f; } \
	};

#define _DECL_AB(name)      template <typename T0, typename T1> struct k_##name { };
#define _DEF_AB(name,T0,T1,f) \
	template <> struct k_##name<T0,T1> {                                                 \
		typedef T0 A;                                                                    \
		typedef T1 B;                                                                    \
		SM_DEVICE_INLINE static void apply(const A* a, uindex_t i, B* b, uindex_t j) { f; } \
	};

// MATCHING TYPES
#if SM_WANT_BOOL || SM_WANT_INT || SM_WANT_UINT
#define _DEF_LL(name,f)     _DEF_AB(name,ct_logical,ct_logical,f)
#else
#define _DEF_LL(name,f)     
#endif
#define _DEF_II(name,f)     _DEF_AB(name,int8_t,int8_t,f) \
                            _DEF_AB(name,int16_t,int16_t,f) \
                            _DEF_AB(name,int32_t,int32_t,f) \
                            _DEF_AB(name,int64_t,int64_t,f) 
#define _DEF_UU(name,f)     _DEF_AB(name,uint8_t,uint8_t,f) \
                            _DEF_AB(name,uint16_t,uint16_t,f) \
                            _DEF_AB(name,uint32_t,uint32_t,f) \
                            _DEF_AB(name,uint64_t,uint64_t,f) 
#define _DEF_FF(name,ff,fd) _DEF_AB(name,float,float,ff) \
                            _DEF_AB(name,double,double,fd)
#define _DEF_ZZ(name,fb,fi,fu)       _DEF_LL(name,fb)    _DEF_II(name,fi)       _DEF_UU(name,fu)
#define _DEF_SS(name,fi,ff,fd)       _DEF_II(name,fi)    _DEF_FF(name,ff,fd)
#define _DEF_NN(name,fi,fu,ff,fd)    _DEF_II(name,fi)    _DEF_UU(name,fu)       _DEF_FF(name,ff,fd)
#define _DEF_GG(name,fb,fi,fu,ff,fd) _DEF_LL(name,fb)    _DEF_NN(name,fi,fu,ff,fd)

// LOGICAL OUTPUT
#define _DEF_IL(name,f)     _DEF_AB(name,int8_t,ct_logical,f) \
                            _DEF_AB(name,int16_t,ct_logical,f) \
                            _DEF_AB(name,int32_t,ct_logical,f) \
                            _DEF_AB(name,int64_t,ct_logical,f) 
#define _DEF_UL(name,f)     _DEF_AB(name,uint8_t,ct_logical,f) \
                            _DEF_AB(name,uint16_t,ct_logical,f) \
                            _DEF_AB(name,uint32_t,ct_logical,f) \
                            _DEF_AB(name,uint64_t,ct_logical,f) 
#define _DEF_FL(name,ff,fd) _DEF_AB(name,float,ct_logical,ff) \
                            _DEF_AB(name,double,ct_logical,fd)
#define _DEF_ZL(name,fb,fi,fu)       _DEF_IL(name,fb)    _DEF_IL(name,fi)       _DEF_UL(name,fu)
#define _DEF_SL(name,fi,ff,fd)       _DEF_IL(name,fi)    _DEF_FL(name,ff,fd)
#define _DEF_NL(name,fi,fu,ff,fd)    _DEF_IL(name,fi)    _DEF_UL(name,fu)       _DEF_FL(name,ff,fd)
#define _DEF_GL(name,fb,fi,fu,ff,fd) _DEF_LL(name,fb)    _DEF_NL(name,fi,fu,ff,fd)

// PROMOTED TYPES
#define _DEF_LP(name,f)     _DEF_AB(name,bool,uint32_t,f)
#define _DEF_IP(name,f)     _DEF_AB(name,int8_t,int32_t,f) \
                            _DEF_AB(name,int16_t,int32_t,f) \
                            _DEF_AB(name,int32_t,int32_t,f) \
                            _DEF_AB(name,int64_t,int64_t,f) 
#define _DEF_UP(name,f)     _DEF_AB(name,uint8_t,uint32_t,f) \
                            _DEF_AB(name,uint16_t,uint32_t,f) \
                            _DEF_AB(name,uint32_t,uint32_t,f) \
                            _DEF_AB(name,uint64_t,uint64_t,f) 
#define _DEF_FP(name,ff,fd) _DEF_FF(name,ff,fd)
#define _DEF_ZP(name,fb,fi,fu)       _DEF_LP(name,fb)    _DEF_IP(name,fi)       _DEF_UP(name,fu)
#define _DEF_SP(name,fi,ff,fd)       _DEF_IP(name,fi)    _DEF_FP(name,ff,fd)
#define _DEF_NP(name,fi,fu,ff,fd)    _DEF_IP(name,fi)    _DEF_UP(name,fu)       _DEF_FP(name,ff,fd)
#define _DEF_GP(name,fb,fi,fu,ff,fd) _DEF_LP(name,fb)    _DEF_NP(name,fi,fu,ff,fd)

// Final macros used in .cpp code
#define DEF_LL(name,f)             _DECL_AB(name) _DEF_LL(name,f)
#define DEF_II(name,f)             _DECL_AB(name) _DEF_II(name,f)
#define DEF_UU(name,f)             _DECL_AB(name) _DEF_UU(name,f)
#define DEF_FF(name,ff,fd)         _DECL_AB(name) _DEF_FF(name,ff,fd)
#define DEF_ZZ(name,fb,fi,fu)      _DECL_AB(name) _DEF_ZZ(name,fb,fi,fu)
#define DEF_SS(name,fi,ff,fd)      _DECL_AB(name) _DEF_SS(name,fi,ff,fd)
#define DEF_NN(name,fi,fu,ff,fd)   _DECL_AB(name) _DEF_NN(name,fi,fu,ff,fd)
#define DEF_GG(name,fb,fi,fu,ff,fd)_DECL_AB(name) _DEF_GG(name,fb,fi,fu,ff,fd)

#define DEF_IL(name,f)             _DECL_AB(name) _DEF_IL(name,f)
#define DEF_UL(name,f)             _DECL_AB(name) _DEF_UL(name,f)
#define DEF_FL(name,ff,fd)         _DECL_AB(name) _DEF_FL(name,ff,fd)
#define DEF_ZL(name,fb,fi,fu)      _DECL_AB(name) _DEF_ZL(name,fb,fi,fu)
#define DEF_SL(name,fi,ff,fd)      _DECL_AB(name) _DEF_SL(name,fi,ff,fd)
#define DEF_NL(name,fi,fu,ff,fd)   _DECL_AB(name) _DEF_NL(name,fi,fu,ff,fd)
#define DEF_GL(name,fb,fi,fu,ff,fd)_DECL_AB(name) _DEF_GL(name,fb,fi,fu,ff,fd)

#define DEF_IP(name,f)             _DECL_AB(name) _DEF_IP(name,f)
#define DEF_UP(name,f)             _DECL_AB(name) _DEF_UP(name,f)
#define DEF_FP(name,ff,fd)         _DECL_AB(name) _DEF_FP(name,ff,fd)
#define DEF_ZP(name,fb,fi,fu)      _DECL_AB(name) _DEF_ZP(name,fb,fi,fu)
#define DEF_SP(name,fi,ff,fd)      _DECL_AB(name) _DEF_SP(name,fi,ff,fd)
#define DEF_NP(name,fi,fu,ff,fd)   _DECL_AB(name) _DEF_NP(name,fi,fu,ff,fd)
#define DEF_GP(name,fb,fi,fu,ff,fd)_DECL_AB(name) _DEF_GP(name,fb,fi,fu,ff,fd)

/////////////////////////////////////////////////////////////////////////////

template <typename functor> // functor = operation to perform on each index
__global__ void kernel_elemwise2(const typename functor::A* a, typename functor::B* b, usize_t size)
{
	DECL_KERNEL_VARS
	#pragma unroll
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)
		functor::apply(a,i,b,i);
}

template <typename functor> // functor = operation to perform on each index
__global__ void kernel_elemwise2_inplace(typename functor::A* a, usize_t size)
{
	DECL_KERNEL_VARS
	#pragma unroll
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)
		functor::apply(a,i,a,i);
}

template <typename functor, bool try_inplace>
struct cuda_elemwise2_launcher {
	static void launch(const argument& a, const argument& b, usize_t size, const launchcfg& cfg)
	{
		kernel_elemwise2<functor><<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(
			a.get<const typename functor::A*>(),
			b.get<      typename functor::B*>(),
			size);
	}
};

template <typename functor>
struct cuda_elemwise2_launcher<functor,true> {
	static void launch(const argument& a, const argument& b, usize_t size, const launchcfg& cfg)
	{
		if (a.get<void*>() == b.get<void*>()) {
			kernel_elemwise2_inplace<functor><<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(
				a.get<typename functor::A*>(), // only pass first argument
				size);
		} else {
			cuda_elemwise2_launcher<functor,false>::launch(a,b,size,cfg);
		}
	}
};

template <template <typename, typename> class functor, bool try_inplace>
struct execute_elemwise2_typed {
	template <typename A, typename B>
	struct general {
		static void execute(opcode_t opcode, const argument& a, const argument& b)
		{
			usize_t size = (usize_t)b.size();
			if (size > 0) {
				// Launch one of many versions of functor, each specialized to the given argument types
				cuda_elemwise2_launcher<functor<A,B>,try_inplace>::launch(a,b,size,make_elemwise_launchcfg(size));
			}
		}
	};

	template <typename T> struct match   { static void execute(opcode_t opcode, const argument& a, const argument& b) { general<T,T   >::execute(opcode,a,b); } };
	template <typename T> struct logical { static void execute(opcode_t opcode, const argument& a, const argument& b) { general<T,ct_logical>::execute(opcode,a,b); } };
};

SM_NAMESPACE_END

#endif // __SM_CUDA_ELEMWISE2_H__
