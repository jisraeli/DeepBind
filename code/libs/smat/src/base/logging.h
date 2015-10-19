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
#ifndef __SM_LOGGING_H__
#define __SM_LOGGING_H__

#include <base/config.h>
#include <cstddef>

#ifndef SM_ENABLE_LOGGING
#define SM_ENABLE_LOGGING
#endif

#ifdef SM_ENABLE_LOGGING
#define SM_LOG(id,...) {if (_SM::get_log_policy(id) != lp_ignore) _SM::log_entry(id,__VA_ARGS__); }
#else
#define SM_LOG(id,...) 
#endif

SM_NAMESPACE_BEGIN

enum logging_policy_t {
	lp_ignore = 0,
	lp_record = 1 << 0,
	lp_write  = 1 << 1,
	lp_print  = 1 << 2
};

SM_INLINE logging_policy_t operator|(logging_policy_t a, logging_policy_t b) { return (logging_policy_t)((unsigned)a | (unsigned)(b)); }
SM_INLINE logging_policy_t operator&(logging_policy_t a, logging_policy_t b) { return (logging_policy_t)((unsigned)a & (unsigned)(b)); }

BASE_EXPORT void log_entry(const char* id, const char* fmt, ...);
BASE_EXPORT void set_log_policy(const char* id, logging_policy_t p);
BASE_EXPORT void set_log_capacity(size_t capacity);
BASE_EXPORT logging_policy_t get_log_policy(const char* id);

SM_NAMESPACE_END

#endif // __SM_LOGGING_H__
