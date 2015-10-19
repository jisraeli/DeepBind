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
#ifndef __SMAT_CONFIG_H__
#define __SMAT_CONFIG_H__

#include <base/config.h>
#include <exception>
#include <stdexcept>
#include <string>

const int cuda_uuid = 0x01;

#ifdef SMAT_EXPORTS
#define SM_EXPORT SM_DLLEXPORT
#define SM_EXTERN_TEMPLATE
#else
#define SM_EXPORT SM_DLLIMPORT
#define SM_EXTERN_TEMPLATE extern
#endif

#define SM_WANT_BOOL 1
#define SM_WANT_INT 1
#define SM_WANT_UINT 1
#define SM_WANT_DOUBLE 1

#ifndef SM_WANT_BOOL
#define SM_WANT_BOOL 1
#endif
#if SM_WANT_BOOL
#define SM_BOOL_TYPES  bool
#else
#define SM_BOOL_TYPES
#endif

#ifndef SM_WANT_INT
#define SM_WANT_INT 1
#endif
#if SM_WANT_INT
#define SM_INT_TYPES   int8_t, int16_t, int32_t, int64_t
#else
#define SM_INT_TYPES
#endif

#ifndef SM_WANT_UINT
#define SM_WANT_UINT 1
#endif
#if SM_WANT_UINT
#define SM_UINT_TYPES  uint8_t,uint16_t,uint32_t,uint64_t
#else
#define SM_UINT_TYPES 
#endif

#ifndef SM_WANT_DOUBLE
#define SM_WANT_DOUBLE 1
#endif
#if SM_WANT_DOUBLE
#define SM_FLOAT_TYPES float,double
#else
#define SM_FLOAT_TYPES float
#endif


#define SM_API_TRY    try {
#define SM_API_CATCH  } catch (const std::exception& e) {\
                         g_smat_last_error = e.what(); \
                         return; \
                      }
#define SM_API_CATCH_AND_RETURN(default_rval)\
                      } catch (const std::exception& e) {\
                         g_smat_last_error = e.what();\
                         return default_rval;\
                      }
extern "C" SM_EXPORT std::string g_smat_last_error;

#endif // __SMAT_CONFIG_H__
