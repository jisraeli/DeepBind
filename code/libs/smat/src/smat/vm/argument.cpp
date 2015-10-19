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
#include <smat/vm/argument.h>

SM_NAMESPACE_BEGIN

static const char* g_vtype_str[] = {
	"none",
	"harray",
	"darray",
	"carray",
	"iarray",
	"user",
};
const char* vtype2str(vtype_t vt) { return g_vtype_str[vt]; }

///////////////////////////////////////////////////////////////

typedef void (*user_deleter_t)(void*);

user_deleter_t& get_user_deleter(argument& op) { return *reinterpret_cast<user_deleter_t*>(&op.shape); }

argument::argument()
: vtype(vt_none)
, dtype(default_dtype)
, shape(0,0,0)
, strides(0,0,0)
{
	*(void**)&value[0] = 0;
}

argument::~argument()
{
	if (vtype == vt_user) {
		// If the argument holds a reference to a user type (i.e. not an array),
		// then check if the user also provided a deleter
		void* user_ptr = get<void*>();
		user_deleter_t user_deleter = get_user_deleter(*this);
		if (user_ptr && user_deleter)
			user_deleter(user_ptr);
	}
}

argument::argument(const argument& src)
: vtype(src.vtype)
, dtype(src.dtype)
, shape(src.shape)
, strides(src.strides)
{
	SM_ASSERT(src.vtype != vt_user)
	*(void**)&value[0] = *(void**)&src.value[0];
}

argument& argument::operator=(const argument& src)
{
	SM_ASSERT(src.vtype != vt_user)
	vtype = src.vtype;
	dtype = src.dtype;
	shape = src.shape;
	strides = src.strides;
	*(void**)&value[0] = *(void**)&src.value[0];
	return *this;
}

argument::argument(argument&& src)
: vtype(src.vtype)
, dtype(src.dtype)
, shape(src.shape)
, strides(src.strides)
{
	*(void**)&value[0] = *(void**)&src.value[0];
	*(void**)&src.value[0] = 0;
}

argument& argument::operator=(argument&& src)
{
	vtype = src.vtype;
	dtype = src.dtype;
	shape = src.shape;
	strides = src.strides;
	*(void**)&value[0] = *(void**)&src.value[0];
	*(void**)&src.value[0] = 0;
	return *this;
}

argument user_arg(void* user, void (*deleter)(void*))
{
	argument arg;
	arg.vtype = vt_user;
	arg.set(user);
	get_user_deleter(arg) = deleter;
	return std::move(arg);
}

SM_NAMESPACE_END
