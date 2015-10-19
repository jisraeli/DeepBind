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
#ifndef __SM_RANDOM_H__
#define __SM_RANDOM_H__

#include <base/config.h>
#include <base/assert.h>
#include <random>

SM_NAMESPACE_BEGIN

BASE_EXPORT void    set_rand_seed(size_t seed);
BASE_EXPORT size_t  bump_rand_seed();
BASE_EXPORT double  rand_dbl();
BASE_EXPORT float   rand_flt();
BASE_EXPORT int     rand_int();
BASE_EXPORT bool    rand_bool();
BASE_EXPORT double  randn_dbl();
BASE_EXPORT float   randn_flt();

// The global procedures are not thread-safe

template <typename T> T rand() { }
template <> SM_INLINE double   rand<double>()   { return rand_dbl(); }
template <> SM_INLINE float    rand<float>()    { return rand_flt(); }
template <> SM_INLINE int      rand<int>()      { return rand_int(); }
template <> SM_INLINE unsigned rand<unsigned>() { return (unsigned)rand_int(); }
template <> SM_INLINE char     rand<char>()     { return (char)rand_int(); }
template <> SM_INLINE unsigned char rand<unsigned char>() { return (unsigned char)rand_int(); }
template <> SM_INLINE bool     rand<bool>()     { return rand_bool(); }

template <typename T> T randn() { }
template <> SM_INLINE double   randn<double>()   { return randn_dbl(); }
template <> SM_INLINE float    randn<float>()    { return randn_flt(); }

// A thread-local distrib object is a thread-safe way to generate random numbers

class distrib {
public:
	void seed(size_t seed) { _gen.seed((unsigned long)seed); }
protected:
	std::mt19937 _gen;
};

template <typename T> struct uniform_distrib: public distrib { public: T operator()() { SM_UNIMPLEMENTED(); } };
template <typename T> struct normal_distrib:  public distrib { public: T operator()() { SM_UNIMPLEMENTED(); } };

#define SM_DEF_DISTRIB(name,dtype,disttype) \
	template <> class name<dtype>: public distrib { \
	public: dtype operator()() { return (dtype)_dist(_gen); }  \
	name() { size_t s = bump_rand_seed(); if (s) seed(s); }    \
	private: std::disttype<dtype> _dist;                       \
	};

SM_DEF_DISTRIB(uniform_distrib,double,uniform_real_distribution)
SM_DEF_DISTRIB(uniform_distrib,float,uniform_real_distribution)
SM_DEF_DISTRIB(uniform_distrib,int,uniform_int_distribution)
SM_DEF_DISTRIB(uniform_distrib,unsigned,uniform_int_distribution)
SM_DEF_DISTRIB(uniform_distrib,char,uniform_int_distribution)
SM_DEF_DISTRIB(uniform_distrib,unsigned char,uniform_int_distribution)
SM_DEF_DISTRIB(normal_distrib,double,normal_distribution)
SM_DEF_DISTRIB(normal_distrib,float,normal_distribution)

SM_NAMESPACE_END

#endif // __SM_RANDOM_H__
