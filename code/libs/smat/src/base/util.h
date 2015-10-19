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
#ifndef __SM_UTIL_H__
#define __SM_UTIL_H__

#include <base/config.h>

#include <string>
#include <vector>
#include <cstdarg>
#ifdef SM_CPP11
#include <type_traits>
#endif

namespace std {
	template <class K> struct hash;
}

SM_NAMESPACE_BEGIN

#ifdef SM_CPP11
using std::common_type;
using std::remove_reference;
#else
// emulate std::common_type<T0,T1>, based on Blitz++ auto promotion

template <typename T> struct remove_reference { typedef T type; };

template <typename T>
struct _promote_rank {
    static const int rank = 0;
    static const bool known = false;
};

#define SM_DECLARE_PRECISION(T,_rank)                 \
    template<> struct _promote_rank<T> {              \
        static const int rank = _rank;                \
        static const bool known = true;               \
    };

SM_DECLARE_PRECISION(int,100)
SM_DECLARE_PRECISION(unsigned int,200)
SM_DECLARE_PRECISION(long,300)
SM_DECLARE_PRECISION(unsigned long,400)
SM_DECLARE_PRECISION(float,500)
SM_DECLARE_PRECISION(double,600)

template <typename T> struct _promote { typedef T type; };
template <> struct _promote<bool>     { typedef int type; };
template <> struct _promote<          char>     { typedef int type; };
template <> struct _promote<unsigned  char>     { typedef int type; };
template <> struct _promote<         short>     { typedef int type; };
template <> struct _promote<unsigned short>     { typedef int type; };

template<typename T0, typename T1, bool useT0>
struct _promote2 {
    typedef T0 type;
};

template<typename T0, typename T1>
struct _promote2<T0,T1,false> {
    typedef T1 type;
};

template <typename T0, typename T1>
struct common_type {
    // Handle promotion of small integers to int/unsigned int
    typedef typename _promote<T0>::type T0p;
    typedef typename _promote<T1>::type T1p;

    // True if T0 is higher ranked
    static const bool T0_better = _promote_rank<T0>::rank > _promote_rank<T1>::rank;
    static const bool T0_larger = sizeof(T0) >= sizeof(T1);

    // True if we know ranks for both T0 and T1
    static const bool know_both =  _promote_rank<T0>::known &&  _promote_rank<T1>::known;
    static const bool know_T0   =  _promote_rank<T0>::known && !_promote_rank<T1>::known;
    static const bool know_T1   = !_promote_rank<T0>::known &&  _promote_rank<T1>::known;

    // If we have both ranks, then use them.
    // If we have only one rank, then use the unknown type.
    // If we have neither rank, then promote to the larger type.
    static const bool useT0 = know_both ? T0_better : (know_T0 ? false : (know_T1 ? true : T0_larger));

    typedef typename _promote2<T0,T1,useT0>::type type;
};

#endif

const double PI = 3.14159265358979323846;

// These only work for integer types currently.
template <typename T0, typename T1> SM_INLINE typename remove_reference<typename common_type<T0,T1>::type>::type divup(T0 x, T1 denom)   { return (x+denom-1)/denom;    }
template <typename T0, typename T1> SM_INLINE typename remove_reference<typename common_type<T0,T1>::type>::type rndup(T0 x, T1 align)   { return divup(x,align)*align; }
template <typename T0, typename T1> SM_INLINE typename remove_reference<typename common_type<T0,T1>::type>::type rnddown(T0 x, T1 align) { return (x/align)*align;      }

BASE_EXPORT std::string format(const char* fmt, ...);
BASE_EXPORT void        print(const char* fmt, ...);
BASE_EXPORT void        check_interrupt(); // throws an exception if the user tried to interrupt computation (Ctrl+C for example)

BASE_EXPORT void set_print_fn(void (*)(const char* fmt, va_list va));
BASE_EXPORT void set_check_interrupt_fn(void (*)());

BASE_EXPORT unsigned gcd(unsigned a, unsigned b); // greatest common denominator, assuming a,b > 0
BASE_EXPORT unsigned lcm(unsigned a, unsigned b); // least common multiple, assuming a,b > 0

// based on boost's hash_combine
template <class T>
SM_INLINE void hash_combine(size_t& hashval, const T& key)
{
	std::hash<T> h;
	hashval ^= h(key) + 0x9e3779b9 + (hashval << 6) + (hashval >> 2);
}

template <typename A>
size_t hash_all(A a) {
	return std::hash<A>()(a);
}

template <typename C, typename P>
void erase_if(C& container, const P& pred)
{
	for (typename C::iterator & i = container.begin(); i != container.end();)
		if (pred(*i)) ++i;
		else i = container.erase(i);
}

#define SM_MASK_TYPE(type) \
	SM_INLINE type operator|(type a, type b) { return (type)((unsigned)a |  (unsigned)b); } \
	SM_INLINE type operator&(type a, type b) { return (type)((unsigned)a &  (unsigned)b); } \
	SM_INLINE type operator+(type a, type b) { return (type)((unsigned)a |  (unsigned)b); } \
	SM_INLINE type operator-(type a, type b) { return (type)((unsigned)a & ~(unsigned)b); } 

template <typename T0, typename T1> SM_INLINE typename remove_reference<typename common_type<T0,T1>::type>::type max(T0 a, T1 b) { return b < a ? a : b; }
template <typename T0, typename T1> SM_INLINE typename remove_reference<typename common_type<T0,T1>::type>::type min(T0 a, T1 b) { return b < a ? b : a; }

BASE_EXPORT std::vector<std::string> split(const std::string& s, const char* delims = 0);

SM_NAMESPACE_END

// Useful macros
#define SM_EXPAND(x) x
#define SM_PP_NARGS(...) SM_EXPAND(SM_NARGS_IMPL(__VA_ARGS__,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0))
#define SM_NARGS_IMPL(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,N,...) N

// Example:
//    f(SM_REPEAT(nullptr,4));             // this
//    f(nullptr,nullptr,nullptr,nullptr);  // expands to this
//
#define SM_REPEAT(x,count) SM_REP_##count(x)
#define SM_REP_0(x)   
#define SM_REP_1(x)                  x
#define SM_REP_2(x)    SM_REP_1(x),  x
#define SM_REP_3(x)    SM_REP_2(x),  x
#define SM_REP_4(x)    SM_REP_3(x),  x
#define SM_REP_5(x)    SM_REP_4(x),  x
#define SM_REP_6(x)    SM_REP_5(x),  x
#define SM_REP_7(x)    SM_REP_6(x),  x
#define SM_REP_8(x)    SM_REP_7(x),  x
#define SM_REP_9(x)    SM_REP_8(x),  x
#define SM_REP_10(x)   SM_REP_9(x),  x
#define SM_REP_11(x)   SM_REP_10(x), x
#define SM_REP_12(x)   SM_REP_11(x), x
#define SM_REP_13(x)   SM_REP_12(x), x
#define SM_REP_14(x)   SM_REP_13(x), x
#define SM_REP_15(x)   SM_REP_14(x), x
#define SM_REP_16(x)   SM_REP_15(x), x
#define SM_REP_17(x)   SM_REP_16(x), x
#define SM_REP_18(x)   SM_REP_17(x), x
#define SM_REP_19(x)   SM_REP_18(x), x
#define SM_REP_20(x)   SM_REP_19(x), x

// Example:
//    void f(SM_REPEAT_N(int z,0,5,f));                     // this
//    void f(int z0f, int z1f, int z2f, int z3f, int z4f);  // expands to this
// Note that x##i##y must be a valid token, so if y is "=5" 
// this will not necessarily compile. Use SM_REPEAT_NB instead.
//
#define SM_REPEAT_N(x,start,count,y) SM_REP##start##_##count(x,y)
#define SM_REP0_0(x,y)   
#define SM_REP0_1(x,y)                     x##0##y
#define SM_REP0_2(x,y)    SM_REP0_1(x,y),  x##1##y
#define SM_REP0_3(x,y)    SM_REP0_2(x,y),  x##2##y
#define SM_REP0_4(x,y)    SM_REP0_3(x,y),  x##3##y
#define SM_REP0_5(x,y)    SM_REP0_4(x,y),  x##4##y
#define SM_REP0_6(x,y)    SM_REP0_5(x,y),  x##5##y
#define SM_REP0_7(x,y)    SM_REP0_6(x,y),  x##6##y
#define SM_REP0_8(x,y)    SM_REP0_7(x,y),  x##7##y
#define SM_REP0_9(x,y)    SM_REP0_8(x,y),  x##8##y
#define SM_REP0_10(x,y)   SM_REP0_9(x,y),  x##9##y
#define SM_REP0_11(x,y)   SM_REP0_10(x,y), x##10##y
#define SM_REP0_12(x,y)   SM_REP0_11(x,y), x##11##y
#define SM_REP0_13(x,y)   SM_REP0_12(x,y), x##12##y
#define SM_REP0_14(x,y)   SM_REP0_13(x,y), x##13##y
#define SM_REP0_15(x,y)   SM_REP0_14(x,y), x##14##y
#define SM_REP0_16(x,y)   SM_REP0_15(x,y), x##15##y
#define SM_REP0_17(x,y)   SM_REP0_16(x,y), x##16##y
#define SM_REP0_18(x,y)   SM_REP0_17(x,y), x##17##y
#define SM_REP0_19(x,y)   SM_REP0_18(x,y), x##18##y
#define SM_REP0_20(x,y)   SM_REP0_19(x,y), x##19##y
#define SM_REP1_0(x,y)   
#define SM_REP1_1(x,y)                     x##1##y
#define SM_REP1_2(x,y)    SM_REP1_1(x,y),  x##2##y
#define SM_REP1_3(x,y)    SM_REP1_2(x,y),  x##3##y
#define SM_REP1_4(x,y)    SM_REP1_3(x,y),  x##4##y
#define SM_REP1_5(x,y)    SM_REP1_4(x,y),  x##5##y
#define SM_REP1_6(x,y)    SM_REP1_5(x,y),  x##6##y
#define SM_REP1_7(x,y)    SM_REP1_6(x,y),  x##7##y
#define SM_REP1_8(x,y)    SM_REP1_7(x,y),  x##8##y
#define SM_REP1_9(x,y)    SM_REP1_8(x,y),  x##9##y
#define SM_REP1_10(x,y)   SM_REP1_9(x,y),  x##10##y
#define SM_REP1_11(x,y)   SM_REP1_10(x,y), x##11##y
#define SM_REP1_12(x,y)   SM_REP1_11(x,y), x##12##y
#define SM_REP1_13(x,y)   SM_REP1_12(x,y), x##13##y
#define SM_REP1_14(x,y)   SM_REP1_13(x,y), x##14##y
#define SM_REP1_15(x,y)   SM_REP1_14(x,y), x##15##y
#define SM_REP1_16(x,y)   SM_REP1_15(x,y), x##16##y
#define SM_REP1_17(x,y)   SM_REP1_16(x,y), x##17##y
#define SM_REP1_18(x,y)   SM_REP1_17(x,y), x##18##y
#define SM_REP1_19(x,y)   SM_REP1_18(x,y), x##19##y
#define SM_REP1_20(x,y)   SM_REP1_19(x,y), x##20##y

// Example:
//    void f(SM_REPEAT_NB(int z,0,5,=0));                        // this
//    void f(int z0=0, int z1=0, int z2=0, int z3=0, int z4=0);  // expands to this
//
#define SM_REPEAT_NB(x,start,count,y) SM_BREP##start##_##count(x,y)
#define SM_BREP0_0(x,y)   
#define SM_BREP0_1(x,y)                      x##0 y
#define SM_BREP0_2(x,y)    SM_BREP0_1(x,y),  x##1 y
#define SM_BREP0_3(x,y)    SM_BREP0_2(x,y),  x##2 y
#define SM_BREP0_4(x,y)    SM_BREP0_3(x,y),  x##3 y
#define SM_BREP0_5(x,y)    SM_BREP0_4(x,y),  x##4 y
#define SM_BREP0_6(x,y)    SM_BREP0_5(x,y),  x##5 y
#define SM_BREP0_7(x,y)    SM_BREP0_6(x,y),  x##6 y
#define SM_BREP0_8(x,y)    SM_BREP0_7(x,y),  x##7 y
#define SM_BREP0_9(x,y)    SM_BREP0_8(x,y),  x##8 y
#define SM_BREP0_10(x,y)   SM_BREP0_9(x,y),  x##9 y
#define SM_BREP0_11(x,y)   SM_BREP0_10(x,y), x##10 y
#define SM_BREP0_12(x,y)   SM_BREP0_11(x,y), x##11 y
#define SM_BREP0_13(x,y)   SM_BREP0_12(x,y), x##12 y
#define SM_BREP0_14(x,y)   SM_BREP0_13(x,y), x##13 y
#define SM_BREP0_15(x,y)   SM_BREP0_14(x,y), x##14 y
#define SM_BREP0_16(x,y)   SM_BREP0_15(x,y), x##15 y
#define SM_BREP0_17(x,y)   SM_BREP0_16(x,y), x##16 y
#define SM_BREP0_18(x,y)   SM_BREP0_17(x,y), x##17 y
#define SM_BREP0_19(x,y)   SM_BREP0_18(x,y), x##18 y
#define SM_BREP0_20(x,y)   SM_BREP0_19(x,y), x##19 y
#define SM_BREP1_0(x,y)
#define SM_BREP1_1(x,y)                      x##1 y
#define SM_BREP1_2(x,y)    SM_BREP1_1(x,y),  x##2 y
#define SM_BREP1_3(x,y)    SM_BREP1_2(x,y),  x##3 y
#define SM_BREP1_4(x,y)    SM_BREP1_3(x,y),  x##4 y
#define SM_BREP1_5(x,y)    SM_BREP1_4(x,y),  x##5 y
#define SM_BREP1_6(x,y)    SM_BREP1_5(x,y),  x##6 y
#define SM_BREP1_7(x,y)    SM_BREP1_6(x,y),  x##7 y
#define SM_BREP1_8(x,y)    SM_BREP1_7(x,y),  x##8 y
#define SM_BREP1_9(x,y)    SM_BREP1_8(x,y),  x##9 y
#define SM_BREP1_10(x,y)   SM_BREP1_9(x,y),  x##10 y
#define SM_BREP1_11(x,y)   SM_BREP1_10(x,y), x##11 y
#define SM_BREP1_12(x,y)   SM_BREP1_11(x,y), x##12 y
#define SM_BREP1_13(x,y)   SM_BREP1_12(x,y), x##13 y
#define SM_BREP1_14(x,y)   SM_BREP1_13(x,y), x##14 y
#define SM_BREP1_15(x,y)   SM_BREP1_14(x,y), x##15 y
#define SM_BREP1_16(x,y)   SM_BREP1_15(x,y), x##16##y
#define SM_BREP1_17(x,y)   SM_BREP1_16(x,y), x##17##y
#define SM_BREP1_18(x,y)   SM_BREP1_17(x,y), x##18##y
#define SM_BREP1_19(x,y)   SM_BREP1_18(x,y), x##19##y
#define SM_BREP1_20(x,y)   SM_BREP1_19(x,y), x##20##y

#define SM_DECR(n) SM_DECR_##n
#define SM_DECR_0 -1
#define SM_DECR_1  0
#define SM_DECR_2  1
#define SM_DECR_3  2
#define SM_DECR_4  3
#define SM_DECR_5  4
#define SM_DECR_6  5
#define SM_DECR_7  6
#define SM_DECR_8  7
#define SM_DECR_9  8
#define SM_DECR_10 9
#define SM_DECR_11 10
#define SM_DECR_12 11
#define SM_DECR_13 12
#define SM_DECR_14 13
#define SM_DECR_15 14
#define SM_DECR_16 15
#define SM_DECR_17 16
#define SM_DECR_18 17
#define SM_DECR_19 18
#define SM_DECR_20 19

#define SM_INCR(n) SM_INCR_##n
#define SM_INCR_0  1
#define SM_INCR_1  2
#define SM_INCR_2  3
#define SM_INCR_3  4
#define SM_INCR_4  5
#define SM_INCR_5  6
#define SM_INCR_6  7
#define SM_INCR_7  8
#define SM_INCR_8  9
#define SM_INCR_9  10
#define SM_INCR_10 11
#define SM_INCR_11 12
#define SM_INCR_12 13
#define SM_INCR_13 14
#define SM_INCR_14 15
#define SM_INCR_15 16
#define SM_INCR_16 17
#define SM_INCR_17 18
#define SM_INCR_18 19
#define SM_INCR_19 20
#define SM_INCR_20 21


#endif // __SM_UTIL_H__
