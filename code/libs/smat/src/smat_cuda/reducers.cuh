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
#ifndef __SM_SMAT_CUDA_REDUCERS_H__
#define __SM_SMAT_CUDA_REDUCERS_H__

#include <smat_cuda/launch_util.h>
#include <limits>
#include <cfloat>

SM_NAMESPACE_BEGIN

// Define dtype_limits<T>::min() and dtype_limits<T>::max() for 
// each type T that a kernel may be specialzed.
template <typename T> struct dtype_limits {};
#define DEF_DTYPE_LIMIT(dtype,_min,_max) \
	template <> struct dtype_limits<dtype>   { SM_DEVICE_INLINE static dtype min() { return _min; } SM_DEVICE_INLINE static dtype max() { return _max; } };
DEF_DTYPE_LIMIT(bool    ,    false,  true)
DEF_DTYPE_LIMIT(int8_t  , CHAR_MIN,  CHAR_MAX)
DEF_DTYPE_LIMIT(uint8_t ,        0, UCHAR_MAX)
DEF_DTYPE_LIMIT(int16_t , SHRT_MIN,  SHRT_MAX)
DEF_DTYPE_LIMIT(uint16_t,        0, USHRT_MAX)
DEF_DTYPE_LIMIT(int32_t ,  INT_MIN,   INT_MAX)
DEF_DTYPE_LIMIT(uint32_t,        0,  UINT_MAX)
DEF_DTYPE_LIMIT(int64_t ,LLONG_MIN, LLONG_MAX)
DEF_DTYPE_LIMIT(uint64_t,        0,ULLONG_MAX)
DEF_DTYPE_LIMIT(float   , -FLT_MAX,   FLT_MAX)
DEF_DTYPE_LIMIT(double  , -FLT_MAX,   DBL_MAX)

#define DEFINE_REDUCER(name,init,elem,part,fin) \
	template <typename _value_type, typename _result_type> \
	struct name {                              \
		typedef _value_type  value_type;                  \
		typedef _result_type result_type;                 \
		SM_DEVICE_INLINE name()             { result = init; } \
		SM_DEVICE_INLINE void element(value_type x)   { elem; }          \
		SM_DEVICE_INLINE void partial(result_type p)  { part; }          \
		SM_DEVICE_INLINE void finalize(usize_t size)  { fin; }           \
		result_type result;                                              \
	};

template <typename T> SM_DEVICE_INLINE T      reducer_element_max(T x, T y) { return y > x ? y : x; }
template <typename T> SM_DEVICE_INLINE T      reducer_element_min(T x, T y) { return y < x ? y : x; }
template <>           SM_DEVICE_INLINE float  reducer_element_max(float  x, float  y) { return ::fmaxf(x,y); }
template <>           SM_DEVICE_INLINE float  reducer_element_min(float  x, float  y) { return ::fminf(x,y); }
template <>           SM_DEVICE_INLINE double reducer_element_max(double x, double y) { return ::fmax(x,y); }
template <>           SM_DEVICE_INLINE double reducer_element_min(double x, double y) { return ::fmin(x,y); }

DEFINE_REDUCER(reducer_max,
			   dtype_limits<value_type>::min(),
			   result = reducer_element_max(result,x),
			   result = reducer_element_max(result,p),
			   )

DEFINE_REDUCER(reducer_min,
			   dtype_limits<value_type>::max(),
			   result = reducer_element_min(result,x),
			   result = reducer_element_min(result,p),
			   )

DEFINE_REDUCER(reducer_sum,
			   0,
			   result += x,
			   result += p,
			   )

DEFINE_REDUCER(reducer_mean,
			   0,
			   result += x,
			   result += p,
			   if (size) result /= size)

DEFINE_REDUCER(reducer_nnz,
			   0,
			   if (x) ++result,
			   result += p,
			   )


// For kernel_reduce, we want to call it with the normal reduce kernels, but then also for 
// subsequent "collect" passes in that algorithm we need to override the default behaviour 
// of element so that it becomes a partial operation.
template <typename reducer>
struct reducer_partial_results: public reducer { 
	SM_DEVICE_INLINE void element(typename reducer::result_type x) { reducer::partial(x); }
};

#if SM_WANT_UINT
	typedef uindex_t reduce_index_t;
#elif SM_WANT_INT
	typedef index_t reduce_index_t;
#else
	typedef float reduce_index_t;
#endif

SM_NAMESPACE_END

#endif // __SM_SMAT_CUDA_REDUCERS_H__
