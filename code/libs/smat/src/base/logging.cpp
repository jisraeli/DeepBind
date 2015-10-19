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
#include <base/logging.h>
#include <base/util.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <list>

SM_NAMESPACE_BEGIN

using namespace std;

struct log_entry_t {
	log_entry_t() { id[0] = '\0'; msg[0] = '\0'; }
	void assign(const char* id, const char* fmt, va_list& va)
	{
		strncpy(this->id,id,16); this->id[15] = '\0';
		vsnprintf(msg,256,fmt,va); this->msg[192] = '\0';
	}
	void print() const
	{
		_SM::print("%s: %s\n",id,msg);
	}
	char id[16];
	char msg[256];
};

static vector<log_entry_t> g_log_entries(4096);
static size_t              g_log_pos = 0;
static unordered_map<string,logging_policy_t> g_policies;

BASE_EXPORT void log_entry(const char* id, const char* fmt, ...)
{
	logging_policy_t policy = get_log_policy(id);
	if (policy == lp_ignore)
		return;  // if not logging this kind of event, return immediately without formatting the message

	va_list va;
	va_start(va,fmt);

	if (policy & lp_record) {
		g_log_entries[g_log_pos].assign(id,fmt,va);
		if (policy & lp_print)
			g_log_entries[g_log_pos].print();
		g_log_pos++;
		if (g_log_pos >= g_log_entries.size())
			g_log_pos = 0;
	} else if (policy & lp_print) {
		log_entry_t entry;
		entry.assign(id,fmt,va);
		entry.print();
	}
}

BASE_EXPORT void set_log_policy(const char* id, logging_policy_t p)
{
	g_policies[id] = p;
}

BASE_EXPORT void set_log_capacity(size_t capacity)
{
	g_log_entries.clear();
	g_log_entries.resize(capacity);
	g_log_pos = 0;
}

BASE_EXPORT logging_policy_t get_log_policy(const char* id)
{
	auto it = g_policies.find(id);
	if (it == g_policies.end())
		return lp_ignore;
	return it->second;
}

SM_NAMESPACE_END
