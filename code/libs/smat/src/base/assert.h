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
#ifndef __SM_ASSERT_H__
#define __SM_ASSERT_H__

#include <base/config.h>

#if defined(_WIN32)
#ifdef _WIN64
#define DL64
#include <intrin.h>
#ifndef SM_DEBUGBREAK
#define SM_DEBUGBREAK if (_SM::g_want_debug_break) __debugbreak()
#endif
#else
#define DL32
#ifndef SM_DEBUGBREAK
#define SM_DEBUGBREAK if (_SM::g_want_debug_break) __asm { int 3 }
#endif
#endif
#endif

#if defined(__GNUC__)
#if defined(__x86_64__) || defined(__ppc64__)
#define DL64
#else
#define DL32
#endif
#ifndef SM_DEBUGBREAK
#define SM_DEBUGBREAK  // not supported
#endif
#endif

SM_NAMESPACE_BEGIN
#if defined(_WIN32)
BASE_EXPORT extern      bool g_want_debug_break;
#endif
BASE_EXPORT             void assert_failed_print(const char* fmt, ...);
BASE_EXPORT SM_NORETURN void assert_failed(const char* fmt, ...);
SM_NAMESPACE_END

#define SM_ASSERT_FAILED(fmt,...) _SM::assert_failed_print(fmt,__VA_ARGS__); SM_DEBUGBREAK; _SM::assert_failed(fmt,__VA_ARGS__);

#define SM_ERROR(msg)          { SM_ASSERT_FAILED("%s\n\tin %s:%d",(const char*)msg,__FILE__,__LINE__); }
#define SM_ASSERT(expr)        { if (expr) { } else { SM_ASSERT_FAILED("AssertionError: ASSERT(%s) failed in %s:%d\n",#expr,__FILE__,__LINE__); } }
#define SM_ASSERTMSG(expr,msg) { if (expr) { } else { SM_ASSERT_FAILED("%s\n\nASSERT(%s) failed in %s:%d\n",(const char*)(msg),#expr,__FILE__,__LINE__); } }
#define SM_UNREACHABLE()       { SM_DEBUGBREAK; SM_ASSERT_FAILED("AssertionError: unreachable code in %s:%d\n",__FILE__,__LINE__); }
#define SM_UNIMPLEMENTED()     { SM_DEBUGBREAK; SM_ASSERT_FAILED("NotImplementedError: raised in %s:%d\n",__FILE__,__LINE__); }

#if defined(_DEBUG)
#ifndef SM_ENABLE_DBASSERT
#define SM_ENABLE_DBASSERT
#endif
#endif

#ifdef SM_ENABLE_DBASSERT
#define SM_DBSTATEMENT(expr) expr
#else
#define SM_DBSTATEMENT(expr)
#endif

#ifdef SM_ENABLE_DBASSERT
#define SM_DBASSERT(expr)      SM_ASSERT(expr)
#else
#define SM_DBASSERT(expr)      { }
#endif



#endif // __SM_ASSERT_H__
