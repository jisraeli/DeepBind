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
#ifndef __SM_TYPELIST_H__
#define __SM_TYPELIST_H__

#include <base/config.h>
#include <base/util.h>

SM_NAMESPACE_BEGIN

struct null_type { };

// typelist

template <typename T, typename _next = null_type> 
struct typelist {
	typedef T type;
	typedef _next next;
};

                          struct make_typelist0 { typedef null_type type; };
template <typename T0>    struct make_typelist1 { typedef typelist<T0,null_type> type; };
#define DECL_MAKE_TYPELISTN(n,m) \
	template <SM_REPEAT_N(typename T,0,n,)> struct make_typelist##n { typedef typelist<T0,typename make_typelist##m<SM_REPEAT_N(T,1,m,)>::type> type; };
DECL_MAKE_TYPELISTN(2,1)
DECL_MAKE_TYPELISTN(3,2)
DECL_MAKE_TYPELISTN(4,3)
DECL_MAKE_TYPELISTN(5,4)
DECL_MAKE_TYPELISTN(6,5)
DECL_MAKE_TYPELISTN(7,6)
DECL_MAKE_TYPELISTN(8,7)
DECL_MAKE_TYPELISTN(9,8)
DECL_MAKE_TYPELISTN(10,9)
DECL_MAKE_TYPELISTN(11,10)
DECL_MAKE_TYPELISTN(12,11)
DECL_MAKE_TYPELISTN(13,12)
DECL_MAKE_TYPELISTN(14,13)
DECL_MAKE_TYPELISTN(15,14)
DECL_MAKE_TYPELISTN(16,15)

template <SM_REPEAT_NB(typename T,0,16,=null_type)> struct make_typelist { typedef typename make_typelist16<SM_REPEAT_N(T,0,16,)>::type type; };
template <>                                         struct make_typelist<SM_REPEAT(null_type,16)> { typedef make_typelist0::type type; };
#define DECL_MAKE_TYPELIST(n,m) \
	template <SM_REPEAT_N(typename T,0,n,)> struct make_typelist<SM_REPEAT_N(T,0,n,),SM_REPEAT(null_type,m)> { typedef typename make_typelist##n<SM_REPEAT_N(T,0,n,)>::type type; };
DECL_MAKE_TYPELIST(1,15)
DECL_MAKE_TYPELIST(2,14)
DECL_MAKE_TYPELIST(3,13)
DECL_MAKE_TYPELIST(4,12)
DECL_MAKE_TYPELIST(5,11)
DECL_MAKE_TYPELIST(6,10)
DECL_MAKE_TYPELIST(7,9)
DECL_MAKE_TYPELIST(8,8)
DECL_MAKE_TYPELIST(9,7)
DECL_MAKE_TYPELIST(10,6)
DECL_MAKE_TYPELIST(11,5)
DECL_MAKE_TYPELIST(12,4)
DECL_MAKE_TYPELIST(13,3)
DECL_MAKE_TYPELIST(14,2)
DECL_MAKE_TYPELIST(15,1)

template <typename L> struct typelist_len            { enum { value = typelist_len<typename L::next>::value + 1 }; };
template <>           struct typelist_len<null_type> { enum { value = 0                                         }; };

template <typename L, size_t i> struct typelist_get                 { typedef typename typelist_get<typename L::next,i-1>::type type; };
template <typename L>           struct typelist_get<L,0>            { typedef typename L::type type; };
template <size_t i>             struct typelist_get<null_type,i> {  };

// concatenation

template <typename L1, typename L2>
struct typelist_cat {
	typedef typelist<typename L1::type,typename typelist_cat<typename L1::next,L2>::type> type;
};

template <typename L2>
struct typelist_cat<null_type,L2> {
	typedef typelist<typename L2::type,typename typelist_cat<null_type,typename L2::next>::type> type;
};

template <>
struct typelist_cat<null_type,null_type> {
	typedef null_type type;
};

// repetition

template <typename L1, int N>
struct typelist_rep {
	typedef typelist<typename L1::type,typename typelist_rep<L1,N-1>::type> type;
};

template <typename L1>
struct typelist_rep<L1,0> {
	typedef null_type type;
};

// wrap -- take a list L1=(T0,T1,...) and turn it into a list of lists ((T0),(T1),...)

template <typename L>
struct typelist_wrap {
	typedef typelist<typename make_typelist1<typename L::type>::type,
	                 typename typelist_wrap<typename L::next>::type> type;
};

template <>
struct typelist_wrap<null_type> {
	typedef null_type type;
};

// zip

template <typename L1, typename L2, typename L3=null_type>
struct typelist_zip {
	typedef typelist<typename make_typelist3<typename L1::type,
	                                         typename L2::type,
	                                         typename L3::type>::type,
	                                         typename typelist_zip<
	                                         typename L1::next,
	                                         typename L2::next,
	                                         typename L3::next>::type> type;
};

template <typename L1, typename L2>
struct typelist_zip<L1,L2,null_type> {
	typedef typelist<typename make_typelist2<typename L1::type,
	                                         typename L2::type>::type,
	                                         typename typelist_zip<
	                                         typename L1::next,
	                                         typename L2::next,
	                                         null_type>::type> type;
};

template <typename L1>
struct typelist_zip<L1,null_type,null_type> {
	typedef null_type type;
};

template <typename L2>
struct typelist_zip<null_type,L2,null_type> {
	typedef null_type type;
};

template <>
struct typelist_zip<null_type,null_type,null_type> {
	typedef null_type type;
};

// cartesian product

template <typename L1, typename L2, typename L2sub=L2>
struct typelist_prod {
	typedef typelist<typename make_typelist<typename L1::type,typename L2sub::type>::type,
		             typename typelist_prod<L1,L2,typename L2sub::next>::type> type;
};

template <>
struct typelist_prod<null_type,null_type,null_type> {
	typedef null_type type;
};

template <typename L1, typename L2>
struct typelist_prod<L1,L2,null_type> {
	typedef typename typelist_prod<typename L1::next,L2,L2>::type type;
};

template <typename L2>
struct typelist_prod<null_type,L2,L2> {
	typedef null_type type;
};

//////////////////////////////////////////////////////////////////////////////////////
// Also use typelist to hold list of compile-time integers via make_intlist<0,1,2,3...,15>::type

template <int N> struct typelist_int { enum { value = N }; };


                          struct make_intlist0 { typedef null_type type; };
template <int N0>    struct make_intlist1 { typedef typelist<typelist_int<N0>,null_type> type; };
#define DECL_MAKE_INTLISTN(n,m) \
	template <SM_REPEAT_N(int N,0,n,)> struct make_intlist##n { typedef typelist<typelist_int<N0>,typename make_intlist##m<SM_REPEAT_N(N,1,m,)>::type> type; };
DECL_MAKE_INTLISTN(2,1)
DECL_MAKE_INTLISTN(3,2)
DECL_MAKE_INTLISTN(4,3)
DECL_MAKE_INTLISTN(5,4)
DECL_MAKE_INTLISTN(6,5)
DECL_MAKE_INTLISTN(7,6)
DECL_MAKE_INTLISTN(8,7)
DECL_MAKE_INTLISTN(9,8)
DECL_MAKE_INTLISTN(10,9)
DECL_MAKE_INTLISTN(11,10)
DECL_MAKE_INTLISTN(12,11)
DECL_MAKE_INTLISTN(13,12)
DECL_MAKE_INTLISTN(14,13)
DECL_MAKE_INTLISTN(15,14)
DECL_MAKE_INTLISTN(16,15)

SM_NAMESPACE_END

#endif // __SM_TYPELIST_H__
