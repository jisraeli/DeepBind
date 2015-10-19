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
#ifndef __BASE_CONFIG_H__
#define __BASE_CONFIG_H__

#if defined _WIN32 || defined __CYGWIN__
#define SM_INLINE    __forceinline
#define SM_NOINLINE  __declspec(noinline)
#define SM_INTERFACE __declspec(novtable)
#define SM_NORETURN  __declspec(noreturn)
#define SM_DLLEXPORT __declspec(dllexport)
#define SM_DLLIMPORT __declspec(dllimport)
#define SM_THREADLOCAL __declspec(thread)
#elif defined __GNUC__
#define SM_INLINE    inline  //__attribute__ ((always_inline))
#define SM_NOINLINE  __attribute__ ((noinline))
#define SM_INTERFACE
#define SM_NORETURN  __attribute__ ((noreturn))
#define SM_DLLEXPORT __attribute__ ((visibility ("default")))
#define SM_DLLIMPORT 
#define SM_THREADLOCAL __thread
#else
#define SM_INLINE   inline
#define SM_NOINLINE
#define SM_INTERFACE
#define SM_NORETURN
#define SM_DLLEXPORT
#define SM_DLLIMPORT
#endif

#if defined (_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

#if (defined(_MSC_VER) && _MSC_VER >= 1700) || defined(__GXX_EXPERIMENTAL_CXX0X__)
#define SM_CPP11
#endif


#ifdef BASE_EXPORTS
#define BASE_EXPORT SM_DLLEXPORT
#define BASE_EXTERN_TEMPLATE
#else
#define BASE_EXPORT SM_DLLIMPORT
#define BASE_EXTERN_TEMPLATE extern
#endif

#define SM_NOCOPY(C) protected: C(const C&); C& operator=(const C&);
#define SM_COPYABLE(C) public: C(const C&); C& operator=(const C&);
#define SM_COPY_CTOR(C) C::C(const C& src)
#define SM_COPY_OPER(C) C& C::operator=(const C& src)
#ifdef SM_CPP11
#define SM_MOVEABLE(C) public: C(C&&); C& operator=(C&&);
#define SM_MOVE_CTOR(C) C::C(C&& src)
#define SM_MOVE_OPER(C) C& C::operator=(C&& src)
#else
#define SM_MOVEABLE(C)
#define SM_MOVE_CTOR(C)
#define SM_MOVE_OPER(C)
#endif

#ifndef SM_USE_NAMESPACE
#define SM_USE_NAMESPACE 1
#endif

#if SM_USE_NAMESPACE
#define _SM sm
#define SM_NAMESPACE_BEGIN namespace _SM {
#define SM_NAMESPACE_END   } // namespace _SM
#define USING_NAMESPACE_SM using namespace _SM
#else
#define _SM
#define SM_NAMESPACE_BEGIN
#define SM_NAMESPACE_END
#define USING_NAMESPACE_SM
#endif

#endif // __BASE_CONFIG_H__
