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
#include <smat/vm/machine.h>
#include <smat/vm/context.h>
#include <base/assert.h>
#include <base/util.h>
#include <base/range.h>
#include <base/random.h>
#include <base/optionset.h>
#include <base/logging.h>
#include <base/sized_array.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <stdexcept>

SM_NAMESPACE_BEGIN

using namespace std;

////////////////////////////////////////////////////////////////////

machine::machine()
{
}

machine::~machine()
{
}

void machine::set_options(const optionset& opt)
{
	// Set which machine/memory events get logged
	auto log_ids = opt.get_strings("log");
	for (auto id : log_ids)
		set_log_policy(id.c_str(),thread_ctx()._verbose ? lp_print : lp_ignore);
}

void machine::validate(instruction& instr, const instruction_info& info)
{
	// First perform some preliminary validation based on whatever generic 
	// constraints were specified when the instruction type was registered.
	// Note that these may modify "instr" when possible (for example, if the
	// original instruction has a constant with mismatched dtype, but the 
	// constant's dtype can be coerced.
	validate_dtypes(instr,info);
	validate_strides(instr,info);

	// Then check if the instruction has a specific validation procedure
	// registered for the current backend. If so, call it as well.
	auto& impl = get_instruction_impl(thread_ctx().backend_info().uuid,instr.opcode);
	if (impl.validate)
		if (!call_validate_fn(impl.validate,instr,info.narg))
			SM_LOG("warn","Instruction %s failed validation.",info.mnemonic);
}

template <typename T>
void cast_arg_value(argument& arg, dtype_t dt)
{
	switch (dt) {
	case b8:  arg.set(arg.get<T>() != 0); break;
	case i8:  arg.set((int8_t  )arg.get<T>()); break;
	case u8:  arg.set((uint8_t )arg.get<T>()); break;
	case i16: arg.set((int16_t )arg.get<T>()); break;
	case u16: arg.set((uint16_t)arg.get<T>()); break;
	case i32: arg.set((int32_t )arg.get<T>()); break;
	case u32: arg.set((uint32_t)arg.get<T>()); break;
	case i64: arg.set((int64_t )arg.get<T>()); break;
	case u64: arg.set((uint64_t)arg.get<T>()); break;
	case f32: arg.set((float   )arg.get<T>()); break;
	case f64: arg.set((double  )arg.get<T>()); break;
	}
	arg.dtype = dt;
}

static void coerce_arg(argument& arg, dtype_t dt)
{
	switch (arg.dtype) {
	case b8:  cast_arg_value<bool    >(arg,dt); break;
	case i8:  cast_arg_value<int8_t  >(arg,dt); break;
	case u8:  cast_arg_value<uint8_t >(arg,dt); break;
	case i16: cast_arg_value<int16_t >(arg,dt); break;
	case u16: cast_arg_value<uint16_t>(arg,dt); break;
	case i32: cast_arg_value<int32_t >(arg,dt); break;
	case u32: cast_arg_value<uint32_t>(arg,dt); break;
	case i64: cast_arg_value<int64_t >(arg,dt); break;
	case u64: cast_arg_value<uint64_t>(arg,dt); break;
	case f32: cast_arg_value<float   >(arg,dt); break;
	case f64: cast_arg_value<double  >(arg,dt); break;
	}
}

static aprops_t s_dtype2aprop(dtype_t dt)
{
	switch (dt) {
	case b8:  return aprop_bool;
	case i8:  return aprop_int;
	case u8:  return aprop_uint;
	case i16: return aprop_int;
	case u16: return aprop_uint;
	case i32: return aprop_int;
	case u32: return aprop_uint;
	case i64: return aprop_int;
	case u64: return aprop_uint;
	case f32: return aprop_float;
	case f64: return aprop_float;
	}
	SM_UNREACHABLE();
}

// validate_dtypes
//   We want to catch unsupported/invalid argument types as the instructions
//   are emitted, rather than when they are executed, so that the error is
//   propagated to the user immediately and they can see which line of
//   their code is causing the problem.
//
void machine::validate_dtypes(instruction& instr, const instruction_info& info)
{
	// First check that argument dtypes are all supported by this instruction
	for (int i : range(info.narg)) {
		auto& arg = instr.arg[i];
		if (info.aprops[i] & aprop_user)
			SM_ASSERTMSG(arg.vtype == vt_user,format("TypeError: Operation '%s' requires argument #%d of user type.\n",info.mnemonic,i+1).c_str())
		else if (arg.vtype != vt_none)
			SM_ASSERTMSG(info.aprops[i] & s_dtype2aprop(arg.dtype),format("TypeError: Operation '%s' does not support argument #%d of type %s.\n",info.mnemonic,i+1,dtype2str(arg.dtype)).c_str())
	}

	// If this instruction also requires matching argument dtypes, check it now.
	// If mismatched dtypes can be easily coerced (e.g. constants converted
	// from int to float) then do so. Otherwise report an error.
	//
	auto match = [&](aprops_t match_types) {
		// First separate the constant operands (cop) and variable operands (vop)
		sized_array<int,instruction::max_arg> vop,cop;
		for (int i : range(info.narg)) {
			if (info.aprops[i] & match_types) {
				if (instr.arg[i].vtype == vt_carray)
					cop.push_back(i);
				else if (instr.arg[i].vtype != vt_user)
					vop.push_back(i);
			}
		}

		// If any of the constant argument dtypes do not match the variable 
		// argument dtypes, try to convert them.
		dtype_t dt0 = instr.arg[vop.empty() ? info.narg-1 : vop[0]].dtype;
		for (int i = 0; i < cop.size(); ++i)
			if (instr.arg[cop[i]].dtype != dt0)
				coerce_arg(instr.arg[cop[i]],dt0);

		// If any of the variable argument dtypes do not match, we have a problem
		if (vop.size() > 1) {
			for (int i = 1; i < vop.size(); ++i)
				if (instr.arg[vop[i]].dtype != dt0)
					SM_ERROR(format("TypeError: Operation '%s' requires arguments with matching data types, but received %s and %s.\n",info.mnemonic,dtype2str(dt0),dtype2str(instr.arg[vop[i]].dtype)).c_str());
		}
	};

	if (info.iprops & iprop_match_all) match(aprop_in | aprop_out);
	if (info.iprops & iprop_match_in)  match(aprop_in);
	if (info.iprops & iprop_match_out) match(aprop_out);
	
	if (instr.opcode == oc_copy) {
		// The copy instruction is a special case, since it 
		// supports dtype coercion only for const source arg.
		if (instr.arg[0].vtype == vt_carray || instr.arg[0].vtype == vt_iarray)
			match(aprop_in | aprop_out);
	}
}

void machine::validate_strides(instruction& instr, const instruction_info& info)
{
	for (int i : range(info.narg)) {
		auto& arg = instr.arg[i];
		if ((info.aprops[i] & aprop_strides) || (arg.vtype != vt_darray && arg.vtype != vt_harray))
			continue;
		// The instruction does not support arbitrary strides on this argument.
		// Check that the argument has full stride.
		if (arg.shape.size() > 0) {
			SM_ASSERTMSG(arg.strides.x == 1,"NotImplementedError: Column slices must be contiguous.");
			SM_ASSERT(arg.strides.y >= arg.shape.x*arg.strides.x);
			if (arg.shape.y > 1 && arg.strides.y != arg.shape.x*arg.strides.x) 
				SM_ERROR("NotImplementedError: Instruction does not yet support column slicing on multi-row matrices.");
		}
	}
}

void machine::execute(const instruction& instr, const instruction_info& info, const instruction_impl& impl)
{
	if (thread_ctx()._verbose >= 1)
		SM_LOG("exec","%s",instr2str(instr).c_str());
	SM_ASSERTMSG(impl.execute,format("NotImplementedError: Instruction '%s' not implemented for backend '%s'.",info.mnemonic,thread_ctx().backend_info().name).c_str());
	call_execute_fn(impl.execute,instr,info.narg);
}

///////////////////////////////////////////////////////////////////////////

SM_NAMESPACE_END
