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
#ifndef __SMAT_DTYPES_H__
#define __SMAT_DTYPES_H__

#include <smat/config.h>
#include <cstddef>
#ifdef SM_CPP11
#include <cstdint>
#endif

SM_NAMESPACE_BEGIN

#ifndef SM_CPP11
typedef          char      int8_t;
typedef unsigned char      uint8_t;
typedef          short     int16_t;
typedef unsigned short     uint16_t;
typedef          int       int32_t;
typedef unsigned int       uint32_t;
typedef          long long int64_t;
typedef unsigned long long uint64_t;
#endif

typedef int32_t   index_t; // If you change this, update slice_end below.
typedef int32_t   isize_t; // Also, be sure to update c_index_t,c_isize_t, and c_slice_end in smat_dll.py
typedef uint32_t  uindex_t;
typedef uint32_t  usize_t; // The reason isize and usize are provided is because, working with "int"
                           // by default avoids wasting time on incompatible type warnings and several
                           // kinds of bugs; usize can be used, for example, when we know we need 
                           // the full 4GB of addressing that 32-bits it provides.

///////////////////////////// dtype_t ///////////////////////////////////

enum dtype_t {
	default_dtype=-1, // use the dtype specified by set_default_dtype, whatever that is
	b8,    // bool
	i8,    // int8_t   (8 bits)
	u8,    // uint8_t  (8 bits)
	i16,   // int16_t  (16 bits)
	u16,   // uint16_t (16 bits)
	i32,   // int32_t  (32 bits)
	u32,   // uint32_t (32 bits)
	i64,   // int64_t  (64 bits)
	u64,   // uint64_t (64 bits)
	f32,   // float    (32 bits)
	f64    // double   (64 bits)
};
enum { num_dtypes = 11 };


// Macros to disable support for certain dtypes at compile time (e.g. to speed up compilation, or reduce DLL size)
#ifndef SM_DTYPE_B8
#define SM_DTYPE_B8 1
#endif
#ifndef SM_DTYPE_I8
#define SM_DTYPE_I8 1
#endif
#ifndef SM_DTYPE_U8
#define SM_DTYPE_U8 1
#endif
#ifndef SM_DTYPE_I16
#define SM_DTYPE_I16 1
#endif
#ifndef SM_DTYPE_U16
#define SM_DTYPE_U16 1
#endif
#ifndef SM_DTYPE_I32
#define SM_DTYPE_I32 1
#endif
#ifndef SM_DTYPE_U32
#define SM_DTYPE_U32 1
#endif
#ifndef SM_DTYPE_I64
#define SM_DTYPE_I64 1
#endif
#ifndef SM_DTYPE_U64
#define SM_DTYPE_U64 1
#endif
#ifndef SM_DTYPE_F32
#define SM_DTYPE_F32 1
#endif
#ifndef SM_DTYPE_F64
#define SM_DTYPE_F64 1
#endif

SM_EXPORT void        set_default_dtype(dtype_t dt);  // can be float or integral
SM_EXPORT void        set_default_dtypef(dtype_t dt); // floats only
SM_EXPORT dtype_t     get_default_dtype();            // default type (can be float or integral)
SM_EXPORT dtype_t     get_default_dtypef();           // default type for floats only
SM_EXPORT dtype_t     actual_dtype(dtype_t dtype);    // returns (dtype==default_dtype) ? get_default_dtype() : dtype
SM_EXPORT dtype_t     arithmetic_result_dtype(dtype_t dta, dtype_t dtb);
SM_EXPORT int         dtype_size(dtype_t dt);
SM_EXPORT const char* dtype2str(dtype_t dt);

template <typename T> struct _ctype2dtype {};  // ctype2dtype<T>::type is the dtype corresponding to ctype T
template <> struct _ctype2dtype<bool>      { static const dtype_t type = b8;  };
template <> struct _ctype2dtype<int8_t>    { static const dtype_t type = i8;  };
template <> struct _ctype2dtype<uint8_t>   { static const dtype_t type = u8;  };
template <> struct _ctype2dtype<int16_t>   { static const dtype_t type = i16; };
template <> struct _ctype2dtype<uint16_t>  { static const dtype_t type = u16; };
template <> struct _ctype2dtype<int32_t>   { static const dtype_t type = i32; };
template <> struct _ctype2dtype<uint32_t>  { static const dtype_t type = u32; };
template <> struct _ctype2dtype<int64_t>   { static const dtype_t type = i64; };
template <> struct _ctype2dtype<uint64_t>  { static const dtype_t type = u64; };
template <> struct _ctype2dtype<float>     { static const dtype_t type = f32; };
template <> struct _ctype2dtype<double>    { static const dtype_t type = f64; };
#define ctype2dtype(ctype) _ctype2dtype<ctype>::type

template <dtype_t dt> struct _dtype2ctype {};  // dtype2ctype<dt>::type is the ctype corresponding to dtype "dt"
template <> struct _dtype2ctype<b8>   { typedef bool     type;  };
template <> struct _dtype2ctype<i8>   { typedef int8_t   type;  };
template <> struct _dtype2ctype<u8>   { typedef uint8_t  type;  };
template <> struct _dtype2ctype<i16>  { typedef int16_t  type;  };
template <> struct _dtype2ctype<u16>  { typedef uint16_t type;  };
template <> struct _dtype2ctype<i32>  { typedef int32_t  type;  };
template <> struct _dtype2ctype<u32>  { typedef uint32_t type;  };
template <> struct _dtype2ctype<i64>  { typedef int64_t  type;  };
template <> struct _dtype2ctype<u64>  { typedef uint64_t type;  };
template <> struct _dtype2ctype<f32>  { typedef float    type;  };
template <> struct _dtype2ctype<f64>  { typedef double   type;  };
#define dtype2ctype(dtype) _dtype2ctype<dtype>::type

// For each type, define a natural corresponding floating point type
// that captures enough information about the original type.
// The casting rules below match that of Numpy.
template <typename T> struct _ctype2ftype { };
template <> struct _ctype2ftype<bool>      { typedef float  type; };
template <> struct _ctype2ftype<int8_t>    { typedef float  type; };
template <> struct _ctype2ftype<uint8_t>   { typedef float  type; };
template <> struct _ctype2ftype<int16_t>   { typedef float  type; };
template <> struct _ctype2ftype<uint16_t>  { typedef float  type; };
template <> struct _ctype2ftype<int32_t>   { typedef float  type; };
template <> struct _ctype2ftype<uint32_t>  { typedef float  type; };
#if SM_WANT_DOUBLE
template <> struct _ctype2ftype<int64_t>   { typedef double type; };
template <> struct _ctype2ftype<uint64_t>  { typedef double type; };
#else
template <> struct _ctype2ftype<int64_t>   { typedef float type; };
template <> struct _ctype2ftype<uint64_t>  { typedef float type; };
#endif
template <> struct _ctype2ftype<float>     { typedef float  type; };
template <> struct _ctype2ftype<double>    { typedef double type; };
#define ctype2ftype(ctype) _ctype2ftype<ctype>::type

// For each type, define a natural corresponding "promoted" type.
// The casting rules below match that of Numpy's sum() operation.
template <typename T> struct _ctype2ptype { };
#if SM_WANT_UINT
template <> struct _ctype2ptype<bool>      { typedef uint32_t type; };
#elif SM_WANT_INT
template <> struct _ctype2ptype<bool>      { typedef int32_t type; };
#else
template <> struct _ctype2ptype<bool>      { typedef float type; };
#endif
template <> struct _ctype2ptype<int8_t>    { typedef int32_t  type; };
template <> struct _ctype2ptype<uint8_t>   { typedef uint32_t type; };
template <> struct _ctype2ptype<int16_t>   { typedef int32_t  type; };
template <> struct _ctype2ptype<uint16_t>  { typedef uint32_t type; };
template <> struct _ctype2ptype<int32_t>   { typedef int32_t  type; };
template <> struct _ctype2ptype<uint32_t>  { typedef uint32_t type; };
template <> struct _ctype2ptype<int64_t>   { typedef int64_t  type; };
template <> struct _ctype2ptype<uint64_t>  { typedef uint64_t type; };
template <> struct _ctype2ptype<float>     { typedef float    type; };
template <> struct _ctype2ptype<double>    { typedef double   type; };
#define ctype2ptype(ctype) _ctype2ptype<ctype>::type

#if SM_WANT_BOOL
const dtype_t dt_logical = b8;
typedef bool ct_logical;
#elif SM_WANT_UINT
const dtype_t dt_logical = u8;
typedef uint8_t ct_logical;
#elif SM_WANT_INT
const dtype_t dt_logical = i8;
typedef int8_t ct_logical;
#else
const dtype_t dt_logical = f32;
typedef float ct_logical;
#endif

SM_NAMESPACE_END

#endif // __SMAT_DTYPES_H__
