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
#ifndef __SM_CUDA_ELEMWISE3_H__
#define __SM_CUDA_ELEMWISE3_H__

#include <smat_cuda/launch_util.h>
#include <smat_cuda/cuda_context.h>
#include <smat/vm/instruction.h>
#include <base/util.h>

SM_NAMESPACE_BEGIN

#define DEF_ABC_GENERIC(name,f) \
	template <typename T0, typename T1, typename T2>                                     \
	struct k_##name {                                                                    \
		typedef T0 A;                                                                    \
		typedef T1 B;                                                                    \
		typedef T2 C;                                                                    \
		SM_DEVICE_INLINE static void apply(const A* a, uindex_t i, const B* b, uindex_t j, C* c, uindex_t k) { f; } \
	};

#define _DECL_ABC(name)      template <typename T0, typename T1, typename T2> struct k_##name { };
#define _DEF_ABC(name,T0,T1,T2,f) \
	template <> struct k_##name<T0,T1,T2> {                                              \
		typedef T0 A;                                                                    \
		typedef T1 B;                                                                    \
		typedef T2 C;                                                                    \
		SM_DEVICE_INLINE static void apply(const A* a, uindex_t i, const B* b, uindex_t j, C* c, uindex_t k) { f; } \
	};

// MATCHING TYPES
#if SM_WANT_BOOL || SM_WANT_INT || SM_WANT_UINT
#define _DEF_LLL(name,f)     _DEF_ABC(name,ct_logical,ct_logical,ct_logical,f)
#else
#define _DEF_LLL(name,f)     
#endif
#define _DEF_III(name,f)     _DEF_ABC(name,int8_t,int8_t,int8_t,f) \
                             _DEF_ABC(name,int16_t,int16_t,int16_t,f) \
                             _DEF_ABC(name,int32_t,int32_t,int32_t,f) \
                             _DEF_ABC(name,int64_t,int64_t,int64_t,f) 
#define _DEF_UUU(name,f)     _DEF_ABC(name,uint8_t,uint8_t,uint8_t,f) \
                             _DEF_ABC(name,uint16_t,uint16_t,uint16_t,f) \
                             _DEF_ABC(name,uint32_t,uint32_t,uint32_t,f) \
                             _DEF_ABC(name,uint64_t,uint64_t,uint64_t,f) 
#define _DEF_FFF(name,ff,fd) _DEF_ABC(name,float,float,float,ff) \
                             _DEF_ABC(name,double,double,double,fd)
#define _DEF_ZZZ(name,fb,fi,fu)       _DEF_LLL(name,fb)    _DEF_III(name,fi)       _DEF_UUU(name,fu)
#define _DEF_SSS(name,fi,ff,fd)       _DEF_III(name,fi)    _DEF_FFF(name,ff,fd)
#define _DEF_NNN(name,fi,fu,ff,fd)    _DEF_III(name,fi)    _DEF_UUU(name,fu)       _DEF_FFF(name,ff,fd)
#define _DEF_GGG(name,fb,fi,fu,ff,fd) _DEF_LLL(name,fb)    _DEF_NNN(name,fi,fu,ff,fd)

// LOGICAL OUTPUT
#define _DEF_IIL(name,f)     _DEF_ABC(name,int8_t,int8_t,ct_logical,f) \
                             _DEF_ABC(name,int16_t,int16_t,ct_logical,f) \
                             _DEF_ABC(name,int32_t,int32_t,ct_logical,f) \
                             _DEF_ABC(name,int64_t,int64_t,ct_logical,f) 
#define _DEF_UUL(name,f)     _DEF_ABC(name,uint8_t,uint8_t,ct_logical,f) \
                             _DEF_ABC(name,uint16_t,uint16_t,ct_logical,f) \
                             _DEF_ABC(name,uint32_t,uint32_t,ct_logical,f) \
                             _DEF_ABC(name,uint64_t,uint64_t,ct_logical,f) 
#if SM_WANT_DOUBLE
#define _DEF_FFL(name,ff,fd) _DEF_ABC(name,float,float,ct_logical,ff) \
                             _DEF_ABC(name,double,double,ct_logical,fd)
#else
#define _DEF_FFL(name,ff,fd) _DEF_ABC(name,float,float,ct_logical,ff) 
#endif
#define _DEF_ZZL(name,fb,fi,fu)       _DEF_IIL(name,fb)    _DEF_IIL(name,fi)       _DEF_UUL(name,fu)
#define _DEF_SSL(name,fi,ff,fd)       _DEF_IIL(name,fi)    _DEF_FFL(name,ff,fd)
#define _DEF_NNL(name,fi,fu,ff,fd)    _DEF_IIL(name,fi)    _DEF_UUL(name,fu)       _DEF_FFL(name,ff,fd)
#define _DEF_GGL(name,fb,fi,fu,ff,fd) _DEF_LLL(name,fb)    _DEF_NNL(name,fi,fu,ff,fd)

// PROMOTED TYPES
#define _DEF_LLP(name,f)     _DEF_ABC(name,ct_logical,ct_logical,uint32_t,f)
#define _DEF_IIP(name,f)     _DEF_ABC(name,int8_t,int8_t,int32_t,f) \
                             _DEF_ABC(name,int16_t,int16_t,int32_t,f) \
                             _DEF_ABC(name,int32_t,int32_t,int32_t,f) \
                             _DEF_ABC(name,int64_t,int64_t,int64_t,f) 
#define _DEF_UUP(name,f)     _DEF_ABC(name,uint8_t,uint8_t,uint32_t,f) \
                             _DEF_ABC(name,uint16_t,uint16_t,uint32_t,f) \
                             _DEF_ABC(name,uint32_t,uint32_t,uint32_t,f) \
                             _DEF_ABC(name,uint64_t,uint64_t,uint64_t,f) 
#define _DEF_FFP(name,ff,fd) _DEF_FFF(name,ff,fd)
#define _DEF_ZZP(name,fb,fi,fu)       _DEF_LLP(name,fb)    _DEF_IIP(name,fi)       _DEF_UUP(name,fu)
#define _DEF_SSP(name,fi,ff,fd)       _DEF_IIP(name,fi)    _DEF_FFP(name,ff,fd)
#define _DEF_NNP(name,fi,fu,ff,fd)    _DEF_IIP(name,fi)    _DEF_UUP(name,fu)       _DEF_FFP(name,ff,fd)
#define _DEF_GGP(name,fb,fi,fu,ff,fd) _DEF_LLP(name,fb)    _DEF_NNP(name,fi,fu,ff,fd)

// Final macros used in .cpp code
#define DEF_LLL(name,f)             _DECL_ABC(name) _DEF_LLL(name,f)
#define DEF_III(name,f)             _DECL_ABC(name) _DEF_III(name,f)
#define DEF_UUU(name,f)             _DECL_ABC(name) _DEF_UUU(name,f)
#define DEF_FFF(name,ff,fd)         _DECL_ABC(name) _DEF_FFF(name,ff,fd)
#define DEF_ZZZ(name,fb,fi,fu)      _DECL_ABC(name) _DEF_ZZZ(name,fb,fi,fu)
#define DEF_SSS(name,fi,ff,fd)      _DECL_ABC(name) _DEF_SSS(name,fi,ff,fd)
#define DEF_NNN(name,fi,fu,ff,fd)   _DECL_ABC(name) _DEF_NNN(name,fi,fu,ff,fd)
#define DEF_GGG(name,fb,fi,fu,ff,fd)_DECL_ABC(name) _DEF_GGG(name,fb,fi,fu,ff,fd)

#define DEF_IIL(name,f)             _DECL_ABC(name) _DEF_IIL(name,f)
#define DEF_UUL(name,f)             _DECL_ABC(name) _DEF_UUL(name,f)
#define DEF_FFL(name,ff,fd)         _DECL_ABC(name) _DEF_FFL(name,ff,fd)
#define DEF_ZZL(name,fb,fi,fu)      _DECL_ABC(name) _DEF_ZZL(name,fb,fi,fu)
#define DEF_SSL(name,fi,ff,fd)      _DECL_ABC(name) _DEF_SSL(name,fi,ff,fd)
#define DEF_NNL(name,fi,fu,ff,fd)   _DECL_ABC(name) _DEF_NNL(name,fi,fu,ff,fd)
#define DEF_GGL(name,fb,fi,fu,ff,fd)_DECL_ABC(name) _DEF_GGL(name,fb,fi,fu,ff,fd)

#define DEF_IIP(name,f)             _DECL_ABC(name) _DEF_IIP(name,f)
#define DEF_UUP(name,f)             _DECL_ABC(name) _DEF_UUP(name,f)
#define DEF_FFP(name,ff,fd)         _DECL_ABC(name) _DEF_FFP(name,ff,fd)
#define DEF_ZZP(name,fb,fi,fu)      _DECL_ABC(name) _DEF_ZZP(name,fb,fi,fu)
#define DEF_SSP(name,fi,ff,fd)      _DECL_ABC(name) _DEF_SSP(name,fi,ff,fd)
#define DEF_NNP(name,fi,fu,ff,fd)   _DECL_ABC(name) _DEF_NNP(name,fi,fu,ff,fd)
#define DEF_GGP(name,fb,fi,fu,ff,fd)_DECL_ABC(name) _DEF_GGP(name,fb,fi,fu,ff,fd)

/////////////////////////////////////////////////////////////////////////////

// kernel_elemwise3_x_yy
// x:
//   C = output to c (output)
//   A = output to a (inplace)
//   B = output to b (inplace)
// yy:
//   dd = elementwise, no broadcasting
//   rd = broadcast a as row vector
//   cd = broadcast a as col vector
//   sd = broadcast a as scalar
//   dr = broadcast b as row vector
//   dc = broadcast b as col vector
//   ds = broadcast b as scalar

#define KERNEL_ELEMWISE3_C_DD_PARAMS   const typename functor::A* a, const typename functor::B* b, typename functor::C* c
#define KERNEL_ELEMWISE3_C_DD_ARGS     a,i,b,i,c,i
#define KERNEL_ELEMWISE3_C_DD_PREAMBLE 
#define KERNEL_ELEMWISE3_A_DD_PARAMS   typename functor::A* a, const typename functor::B* b
#define KERNEL_ELEMWISE3_A_DD_ARGS     a,i,b,i,a,i
#define KERNEL_ELEMWISE3_A_DD_PREAMBLE 
#define KERNEL_ELEMWISE3_B_DD_PARAMS   const typename functor::A* a, typename functor::B* b
#define KERNEL_ELEMWISE3_B_DD_ARGS     a,i,b,i,b,i
#define KERNEL_ELEMWISE3_B_DD_PREAMBLE 

#define KERNEL_ELEMWISE3_C_RD_PARAMS   const typename functor::A* a, const typename functor::B* b, typename functor::C* c, usize_t m
#define KERNEL_ELEMWISE3_C_RD_ARGS     a,i%m,b,i,c,i
#define KERNEL_ELEMWISE3_C_RD_PREAMBLE 
#define KERNEL_ELEMWISE3_B_RD_PARAMS   const typename functor::A* a, typename functor::B* b, usize_t m
#define KERNEL_ELEMWISE3_B_RD_ARGS     a,i%m,b,i,b,i
#define KERNEL_ELEMWISE3_B_RD_PREAMBLE 

#define KERNEL_ELEMWISE3_C_CD_PARAMS   const typename functor::A* a, const typename functor::B* b, typename functor::C* c, usize_t m
#define KERNEL_ELEMWISE3_C_CD_ARGS     a,i/m,b,i,c,i
#define KERNEL_ELEMWISE3_C_CD_PREAMBLE 
#define KERNEL_ELEMWISE3_B_CD_PARAMS   const typename functor::A* a, typename functor::B* b, usize_t m
#define KERNEL_ELEMWISE3_B_CD_ARGS     a,i/m,b,i,b,i
#define KERNEL_ELEMWISE3_B_CD_PREAMBLE 

#define KERNEL_ELEMWISE3_C_SD_PARAMS   const typename functor::A _a, const typename functor::B* b, typename functor::C* c
#define KERNEL_ELEMWISE3_C_SD_ARGS     a,0,b,i,c,i
#define KERNEL_ELEMWISE3_C_SD_PREAMBLE const A a[1] = { _a };
#define KERNEL_ELEMWISE3_B_SD_PARAMS   const typename functor::A _a, typename functor::B* b
#define KERNEL_ELEMWISE3_B_SD_ARGS     a,0,b,i,b,i
#define KERNEL_ELEMWISE3_B_SD_PREAMBLE const A a[1] = { _a };

#define KERNEL_ELEMWISE3_C_DR_PARAMS   const typename functor::A* a, const typename functor::B* b, typename functor::C* c, usize_t m
#define KERNEL_ELEMWISE3_C_DR_ARGS     a,i,b,i%m,c,i
#define KERNEL_ELEMWISE3_C_DR_PREAMBLE 
#define KERNEL_ELEMWISE3_A_DR_PARAMS   typename functor::A* a, const typename functor::B* b, usize_t m
#define KERNEL_ELEMWISE3_A_DR_ARGS     a,i,b,i%m,a,i
#define KERNEL_ELEMWISE3_A_DR_PREAMBLE 

#define KERNEL_ELEMWISE3_C_DC_PARAMS   const typename functor::A* a, const typename functor::B* b, typename functor::C* c, usize_t m
#define KERNEL_ELEMWISE3_C_DC_ARGS     a,i,b,i/m,c,i
#define KERNEL_ELEMWISE3_C_DC_PREAMBLE 
#define KERNEL_ELEMWISE3_A_DC_PARAMS   typename functor::A* a, const typename functor::B* b, usize_t m
#define KERNEL_ELEMWISE3_A_DC_ARGS     a,i,b,i/m,a,i
#define KERNEL_ELEMWISE3_A_DC_PREAMBLE 

#define KERNEL_ELEMWISE3_C_DS_PARAMS   const typename functor::A* a, const typename functor::B _b, typename functor::C* c
#define KERNEL_ELEMWISE3_C_DS_ARGS     a,i,b,0,c,i
#define KERNEL_ELEMWISE3_C_DS_PREAMBLE const B b[1] = { _b };
#define KERNEL_ELEMWISE3_A_DS_PARAMS   typename functor::A* a, const typename functor::B _b
#define KERNEL_ELEMWISE3_A_DS_ARGS     a,i,b,0,a,i
#define KERNEL_ELEMWISE3_A_DS_PREAMBLE const B b[1] = { _b };

#define KERNEL_ELEMWISE3_C_SS_PARAMS   const typename functor::A _a, const typename functor::B _b, typename functor::C* c
#define KERNEL_ELEMWISE3_C_SS_ARGS     a,i,b,0,c,i
#define KERNEL_ELEMWISE3_C_SS_PREAMBLE const A a[1] = { _a }; const B b[1] = { _b };

#define DEF_KERNEL_ELEMWISE3(inplace,broadcast) \
	template <typename functor>                            \
	__global__ void kernel_elemwise_b_##inplace##_##broadcast(KERNEL_ELEMWISE3_##inplace##_##broadcast##_PARAMS, usize_t size)  \
	{                                                                                                        \
		typedef typename functor::A A;                                                                       \
		typedef typename functor::B B;                                                                       \
		typedef typename functor::C C;                                                                       \
		DECL_KERNEL_VARS                                                                                     \
		KERNEL_ELEMWISE3_##inplace##_##broadcast##_PREAMBLE                                                   \
		for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)                                         \
			functor::apply(KERNEL_ELEMWISE3_##inplace##_##broadcast##_ARGS);                                  \
	}

DEF_KERNEL_ELEMWISE3(C,DD)  // output to c, no broadcasting
DEF_KERNEL_ELEMWISE3(A,DD)  // output to a, no broadcasting
DEF_KERNEL_ELEMWISE3(B,DD)  // output to b, no broadcasting
DEF_KERNEL_ELEMWISE3(C,RD)  // output to c, broadcast row vec a
DEF_KERNEL_ELEMWISE3(B,RD)  // output to b, broadcast row vec a
DEF_KERNEL_ELEMWISE3(C,CD)  // output to c, broadcast col vec a
DEF_KERNEL_ELEMWISE3(B,CD)  // output to b, broadcast col vec a
DEF_KERNEL_ELEMWISE3(C,SD)  // output to c, broadcast scalar a
DEF_KERNEL_ELEMWISE3(B,SD)  // output to b, broadcast scalar a
DEF_KERNEL_ELEMWISE3(C,DR)  // output to c, broadcast row vec b
DEF_KERNEL_ELEMWISE3(A,DR)  // output to a, broadcast row vec b
DEF_KERNEL_ELEMWISE3(C,DC)  // output to c, broadcast col vec b
DEF_KERNEL_ELEMWISE3(A,DC)  // output to a, broadcast col vec b
DEF_KERNEL_ELEMWISE3(C,DS)  // output to c, broadcast scalar b
DEF_KERNEL_ELEMWISE3(A,DS)  // output to a, broadcast scalar b
DEF_KERNEL_ELEMWISE3(C,SS)  // output to c, broadcast scalar result of f(a,b)


#define LAUNCH_KERNEL_ELEMWISE3(inplace,broadcast) \
	kernel_elemwise_b_##inplace##_##broadcast<functor><<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>> 
	
		
#define ELEMWISE_LAUNCHER_BXX_ERROR \
	SM_ERROR(format("NotImplementedError: Unsupported combination of argument value types in launch: %s:%s %s:%s -> %s:%s.", \
						vtype2str(a.vtype),dtype2str(a.dtype), \
						vtype2str(b.vtype),dtype2str(b.dtype), \
						vtype2str(c.vtype),dtype2str(c.dtype)).c_str()); 

template <typename functor, bool allow_inplace>
struct cuda_elemwise3_launcher {
	static void launch(const argument& a, const argument& b, const argument& c, usize_t size, const launchcfg& cfg)
	{
		typedef typename functor::A A;
		typedef typename functor::B B;
		typedef typename functor::C C;
		if (a.vtype == vt_darray && b.vtype == vt_darray && c.vtype == vt_darray) {
			// Launch either an elementwise or a row/col broadcasted version of the functor
			if (a.shape == b.shape) {
				LAUNCH_KERNEL_ELEMWISE3(C,DD)(a.get<const A*>(),b.get<const B*>(),c.get<C*>(),size);                 // direct elementwise operation
			} else if (a.shape.y == 1 && a.shape.x == b.shape.x) {
				LAUNCH_KERNEL_ELEMWISE3(C,RD)(a.get<const A*>(),b.get<const B*>(),c.get<C*>(),c.shape.x,size);          // broadcast row vector on left
			} else if (a.shape.x == 1 && a.shape.y == b.shape.y) {
				LAUNCH_KERNEL_ELEMWISE3(C,CD)(a.get<const A*>(),b.get<const B*>(),c.get<C*>(),c.shape.x,size);         // broadcast col vector on left
			} else if (b.shape.y == 1 && a.shape.x == b.shape.x) {
				LAUNCH_KERNEL_ELEMWISE3(C,DR)(a.get<const A*>(),b.get<const B*>(),c.get<C*>(),c.shape.x,size);         // broadcast row vector on right
			} else if (b.shape.x == 1 && a.shape.y == b.shape.y) {
				LAUNCH_KERNEL_ELEMWISE3(C,DC)(a.get<const A*>(),b.get<const B*>(),c.get<C*>(),c.shape.x,size);         // broadcast col vector on right
			} else 
				SM_ERROR("NotImplementedError: incompatible broadcasting dimensions at kernel launch."); 
				
		} else if (a.vtype == vt_carray && b.vtype == vt_darray && c.vtype == vt_darray) { 
			LAUNCH_KERNEL_ELEMWISE3(C,SD)(a.get<A>(),b.get<const B*>(),c.get<C*>(),size);                     // broadcast scalar on left
		} else if (a.vtype == vt_darray && b.vtype == vt_carray && c.vtype == vt_darray) {
			LAUNCH_KERNEL_ELEMWISE3(C,DS)(a.get<const A*>(),b.get<B>(),c.get<C*>(),size);                     // broadcast scalar on right
		} else if (a.vtype == vt_carray && b.vtype == vt_carray && c.vtype == vt_darray) {
			LAUNCH_KERNEL_ELEMWISE3(C,SS)(a.get<A>(),b.get<B>(),c.get<C*>(),size);                      // broadcast scalar-scalar result to output
		} else {
			ELEMWISE_LAUNCHER_BXX_ERROR;
		}
	}
};

template <typename functor>
struct cuda_elemwise3_launcher<functor,true> {
	static void launch(const argument& a, const argument& b, const argument& c, usize_t size, const launchcfg& cfg)
	{
		typedef typename functor::A A;
		typedef typename functor::B B;
		typedef typename functor::C C;
		if (a.get<void*>() == c.get<void*>()) {
			// Launch inplace version of functor, writing output to a and omitting c completely
			if (a.vtype == vt_darray && b.vtype == vt_darray && c.vtype == vt_darray) {
				if (a.shape == b.shape) {
					LAUNCH_KERNEL_ELEMWISE3(A,DD)(a.get<A*>(),b.get<const B*>(),size);                 // direct elementwise operation
				} else if (b.shape.y == 1 && a.shape.x == b.shape.x) {
					LAUNCH_KERNEL_ELEMWISE3(A,DR)(a.get<A*>(),b.get<const B*>(),a.shape.x,size);    // broadcast row vector on right
				} else if (b.shape.x == 1 && a.shape.y == b.shape.y) {
					LAUNCH_KERNEL_ELEMWISE3(A,DC)(a.get<A*>(),b.get<const B*>(),a.shape.x,size);    // broadcast col vector on right
				} else 
					SM_ERROR("NotImplementedError: incompatible broadcasting dimensions at kernel launch.\n"); 
			} else if (a.vtype == vt_darray && b.vtype == vt_carray && c.vtype == vt_darray) {
				LAUNCH_KERNEL_ELEMWISE3(A,DS)(a.get<A*>(),b.get<const B>(),size);                      // broadcast scalar on right
			} else {
				ELEMWISE_LAUNCHER_BXX_ERROR;
			}
		} else if (b.get<void*>() == c.get<void*>()) {
			// Launch inplace version of functor, writing output to b and omitting c completely
			if (a.vtype == vt_darray && b.vtype == vt_darray && c.vtype == vt_darray) {
				if (a.shape == b.shape) {
					LAUNCH_KERNEL_ELEMWISE3(B,DD)(a.get<const A*>(),b.get<B*>(),size);                 // direct elementwise operation
				} else if (a.shape.y == 1 && a.shape.x == b.shape.x) {
					LAUNCH_KERNEL_ELEMWISE3(B,RD)(a.get<const A*>(),b.get<B*>(),b.shape.x,size);    // broadcast row vector on left
				} else if (a.shape.x == 1 && a.shape.y == b.shape.y) {
					LAUNCH_KERNEL_ELEMWISE3(B,CD)(a.get<const A*>(),b.get<B*>(),b.shape.x,size);    // broadcast col vector on left
				} else 
					SM_ERROR("NotImplementedError: incompatible broadcasting dimensions at kernel launch.\n"); 
			} else if (a.vtype == vt_carray && b.vtype == vt_darray && c.vtype == vt_darray) { 
				LAUNCH_KERNEL_ELEMWISE3(B,SD)(a.get<const A>(),b.get<B*>(),size);                      // broadcast scalar on left
			} else {
				ELEMWISE_LAUNCHER_BXX_ERROR;
			}
		} else {
			// Launch general version of functor
			cuda_elemwise3_launcher<functor,false>::launch(a,b,c,size,cfg);
		}
	}
};

template <template <typename, typename, typename> class functor, bool try_inplace>
struct execute_elemwise3_typed {
	template <typename A, typename B, typename C>
	struct general {
		static void execute(opcode_t opcode, const argument& a, const argument& b, const argument& c)
		{
			usize_t size = (usize_t)c.size();
			if (size > 0) {
				// Launch one of many versions of functor, each specialized to the given argument types
				cuda_elemwise3_launcher<functor<A,B,C>,try_inplace>::launch(a,b,c,size,make_elemwise_launchcfg(size));
			}
		}
	};

	template <typename T> struct match   { static void execute(opcode_t opcode, const argument& a, const argument& b, const argument& c) { general<T,T,T   >::execute(opcode,a,b,c); } };
	template <typename T> struct logical { static void execute(opcode_t opcode, const argument& a, const argument& b, const argument& c) { general<T,T,dtype2ctype(b8)>::execute(opcode,a,b,c); } };
};

SM_NAMESPACE_END

#endif // __SM_CUDA_ELEMWISE3_H__
