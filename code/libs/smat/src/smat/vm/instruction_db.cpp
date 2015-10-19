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
#include <smat/vm/instruction_db.h>
#include <base/assert.h>
#include <base/util.h>
#include <base/range.h>
#include <vector>
#include <map>
#include <algorithm>
#include <cstring>

SM_NAMESPACE_BEGIN

using namespace std;

string instr2str(const instruction& i)
{

	// Helper function for operands
	auto arg2str = [&](const argument& arg) -> string {
		string str;
		if (arg.vtype == vt_harray) {
			str = format("h:%010llx",arg.get<void*>());
		} else if (arg.vtype == vt_darray) {
			str = format("d:%010llx",arg.get<size_t>());
		} else if (arg.vtype == vt_carray) {
			switch (arg.dtype) {
			case b8:  str = format("%12s",arg.get<bool>() ? "true" : "false"); break;
			case i8:  str = format("%12d",(int)arg.get<char>()); break;
			case u8:  str = format("%12u",(unsigned)arg.get<unsigned char>()); break;
			case i16: str = format("%12d",(int)arg.get<short>()); break;
			case u16: str = format("%12u",(unsigned)arg.get<unsigned short>()); break;
			case i32: str = format("%12d",arg.get<int>()); break;
			case u32: str = format("%12u",arg.get<unsigned>()); break;
			case i64: str = format("%12lld",arg.get<long long>()); break;
			case u64: str = format("%12llu",arg.get<unsigned long long>()); break;
			case f32: str = format("%12g",(double)arg.get<float>()); break;
			case f64: str = format("%12g",arg.get<double>()); break;
			}
		} else if (arg.vtype == vt_user) {
			str = format("u:%010llx",arg.get<size_t>());
		} else if (arg.vtype == vt_iarray) {
			str = format("%12s","identity");
		}
		return str;
	};

	// Helper function for comment
	auto arginfo2str = [&](const argument& arg) -> string {
		string str;
		if (arg.vtype == vt_carray)
			str = format("%s",dtype2str(arg.dtype));
		else if (arg.vtype == vt_user)
			str = "user";
		else
			str = format("%s:%s",shape2str(arg.shape).c_str(),dtype2str(arg.dtype));
		return str;
	};

	string str;

	// Opcode name
	auto& info = get_instruction_info(i.opcode);
	str += format("%s",info.mnemonic);
	str.resize(10,' ');

	// Argument list
	for (int j : range(info.narg)) {
		if (j > 0)
			str += ", ";
		str += arg2str(i.arg[j]);
	}

	// Now the comment, going at end of line
	string comment;
	for (int j : range(info.narg)) {
		if (j > 0)
			comment += ", ";
		if (i.arg[j].vtype == vt_none)
			comment += "none";
		else
			comment += arginfo2str(i.arg[j]);
	}

	str += "\t; ";
	str += comment;
	str += '\n';
	return str;
}

string instr2str(const instruction_list& instr)
{ 
	return instr2str(instr.begin(),instr.end());
}

string instr2str(instruction_list::const_iterator first, instruction_list::const_iterator last)
{
	string str;
	for (auto i = first; i != last; ++i)
		str += instr2str(*i);
	return str;

	/*
	vector<int> reflist;
	for (auto& r : m_reg)
		if (r.second.refcount > 0)
			reflist.push_back(r.first);
	if (!reflist.empty()) {
		str += "\n// still being referenced: ";
		sort(reflist.begin(),reflist.end());
		for (auto& i : reflist) {
			str += reg2str(i);
			str += ',';
		}
		str.pop_back();
		str += '\n';
	}
	*/
	return str;
}

typedef vector<instruction_info>         instruction_info_registry; // [opcode]
typedef vector<vector<instruction_impl>> instruction_impl_registry; // [backend_uuid][opcode]

instruction_info_registry g_instruction_info;
instruction_impl_registry g_instruction_impl;
extern dllhandle_t g_extension_loading; // defined in extension.cpp

static bool s_builtin_added = false;

void register_builtin_instructions()
{
	s_builtin_added = true; 
	const iprops_t CO = iprop_commutes,
				   EL = iprop_elemwise,
				   IP = iprop_inplace,
				   MI = iprop_match_in,
				   MO = iprop_match_out,
				   MA = iprop_match_all,
				   RX = iprop_reduce_x,
				   RY = iprop_reduce_y,
				   RZ = iprop_reduce_z,
				   none = iprop_none;
	const aprops_t INP  = aprop_in,
				   OUT  = aprop_out,
				   B    = SM_WANT_BOOL ? aprop_bool : 
				          SM_WANT_UINT ? aprop_uint : 
						  SM_WANT_INT  ? aprop_int : aprop_float,
				   I    = SM_WANT_INT  ? aprop_int : SM_WANT_UINT ? aprop_uint : aprop_float,
				   U    = SM_WANT_UINT ? aprop_uint : SM_WANT_INT ? aprop_int : aprop_float,
				   F    = aprop_float,
				   N    = aprop_num,
				   A    = aprop_any,
				   ST   = aprop_strides;

	auto old_extension_loading = g_extension_loading;
	g_extension_loading = 0; // make sure the builtin instructions are not associated with an extension, even if they were added during registration of an extension.
	try {
		#define ADD_BUILTIN(opname,iprops,...) SM_ASSERT(oc_##opname == add_instruction(#opname,iprops,__VA_ARGS__))
		SM_ASSERT(oc_noop == add_instruction("noop",none));
		ADD_BUILTIN(alloc, EL,                                      OUT|A);
		ADD_BUILTIN(free,  EL,            INP|A);
		ADD_BUILTIN(copy,  EL,            INP|A|ST,                 OUT|A|ST);
		ADD_BUILTIN(rand,  EL,                                      OUT|A);
		ADD_BUILTIN(randn, EL,                                      OUT|F);
		ADD_BUILTIN(bernl, EL,            INP|F,                    OUT|A);
		ADD_BUILTIN(add,   EL|MA|IP|CO,   INP|A,       INP|A,       OUT|A);
		ADD_BUILTIN(sub,   EL|MA|IP,      INP|A,       INP|A,       OUT|A);
		ADD_BUILTIN(mul,   EL|MA|IP|CO,   INP|A,       INP|A,       OUT|A);
		ADD_BUILTIN(div,   EL|MA|IP,      INP|A,       INP|A,       OUT|A);
		ADD_BUILTIN(mod,   EL|MA|IP,      INP|A,       INP|A,       OUT|A);
		ADD_BUILTIN(pow,   EL|MA|IP,      INP|A,       INP|A,       OUT|A);
		ADD_BUILTIN(dot,      MA,         INP|F,       INP|F,       OUT|F);
		ADD_BUILTIN(dottn,    MA,         INP|F,       INP|F,       OUT|F);
		ADD_BUILTIN(dotnt,    MA,         INP|F,       INP|F,       OUT|F);
		ADD_BUILTIN(dottt,    MA,         INP|F,       INP|F,       OUT|F);
		ADD_BUILTIN(neg,   EL|MA|IP,      INP|I+F,                  OUT|I+F);
		ADD_BUILTIN(abs,   EL|MA|IP,      INP|A,                    OUT|A);
		ADD_BUILTIN(sign,  EL|MA|IP,      INP|A,                    OUT|A);
		ADD_BUILTIN(signb, EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(sin,   EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(cos,   EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(tan,   EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(asin,  EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(acos,  EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(atan,  EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(sinh,  EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(cosh,  EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(tanh,  EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(asinh, EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(acosh, EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(atanh, EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(exp,   EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(exp2,  EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(log,   EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(log2,  EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(sigm,  EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(sqrt,  EL|MA|IP,      INP|A,                    OUT|A);
		ADD_BUILTIN(sqr,   EL|MA|IP,      INP|A,                    OUT|A);
		ADD_BUILTIN(rnd,   EL|MA|IP,      INP|A,                    OUT|A);
		ADD_BUILTIN(flr,   EL|MA|IP,      INP|A,                    OUT|A);
		ADD_BUILTIN(ceil,  EL|MA|IP,      INP|A,                    OUT|A);
		ADD_BUILTIN(sat,   EL|MA|IP,      INP|F,                    OUT|F);
		ADD_BUILTIN(isinf, EL|   IP,      INP|A,                    OUT|B);
		ADD_BUILTIN(isnan, EL|   IP,      INP|A,                    OUT|B);
		ADD_BUILTIN(lnot,  EL|   IP,      INP|A,                    OUT|B);
		ADD_BUILTIN(lor,   EL|MI|IP|CO,   INP|A,       INP|A,       OUT|B);
		ADD_BUILTIN(land,  EL|MI|IP|CO,   INP|A,       INP|A,       OUT|B);
		ADD_BUILTIN(not,   EL|MA|IP,      INP|A-F,                  OUT|A-F);
		ADD_BUILTIN(or,    EL|MA|IP|CO,   INP|A-F,     INP|A-F,     OUT|A-F);
		ADD_BUILTIN(and,   EL|MA|IP|CO,   INP|A-F,     INP|A-F,     OUT|A-F);
		ADD_BUILTIN(xor,   EL|MA|IP|CO,   INP|A-F,     INP|A-F,     OUT|A-F);
		ADD_BUILTIN(eq,    EL|MI|IP|CO,   INP|A,       INP|A,       OUT|B);
		ADD_BUILTIN(ne,    EL|MI|IP|CO,   INP|A,       INP|A,       OUT|B);
		ADD_BUILTIN(lt,    EL|MI|IP,      INP|A,       INP|A,       OUT|B);
		ADD_BUILTIN(le,    EL|MI|IP,      INP|A,       INP|A,       OUT|B);
		ADD_BUILTIN(maxe,  EL|MA|IP|CO,   INP|A,       INP|A,       OUT|A);
		ADD_BUILTIN(mine,  EL|MA|IP|CO,   INP|A,       INP|A,       OUT|A);
		ADD_BUILTIN(max,   RX+RY+RZ,      INP|A,                    OUT|A);
		ADD_BUILTIN(max_x, RX,            INP|A,                    OUT|A);
		ADD_BUILTIN(max_y, RY,            INP|A,                    OUT|A);
		ADD_BUILTIN(min,   RX+RY+RZ,      INP|A,                    OUT|A);
		ADD_BUILTIN(min_x, RX,            INP|A,                    OUT|A);
		ADD_BUILTIN(min_y, RY,            INP|A,                    OUT|A);
		ADD_BUILTIN(sum,   RX+RY+RZ,      INP|A,                    OUT|A);
		ADD_BUILTIN(sum_x, RX,            INP|A,                    OUT|A);
		ADD_BUILTIN(sum_y, RY,            INP|A,                    OUT|A);
		ADD_BUILTIN(mean,  RX+RY+RZ,      INP|A,                    OUT|A);
		ADD_BUILTIN(mean_x,RX,            INP|A,                    OUT|A);
		ADD_BUILTIN(mean_y,RY,            INP|A,                    OUT|A);
		ADD_BUILTIN(nnz,   RX+RY+RZ,      INP|A,                    OUT|A);
		ADD_BUILTIN(nnz_x, RX,            INP|A,                    OUT|A);
		ADD_BUILTIN(nnz_y, RY,            INP|A,                    OUT|A);
		ADD_BUILTIN(diff_x,MA|IP,         INP|A,                    OUT|A);
		ADD_BUILTIN(diff_y,MA|IP,         INP|A,                    OUT|A);
		ADD_BUILTIN(rep,   MA,            INP|A,                    OUT|A);
		ADD_BUILTIN(tile,  MA,            INP|A,                    OUT|A);
		ADD_BUILTIN(trace, MA,            INP|A,                    OUT|A);
		ADD_BUILTIN(trans, MA,            INP|A,                    OUT|A);
		ADD_BUILTIN(arang, MA,            INP|N,                    OUT|N);
		ADD_BUILTIN(mask,  EL,            INP|OUT|F,   INP|B);
		ADD_BUILTIN(dropout_fp_tr, none,  INP|F,       INP|F,   OUT|F,  OUT|B);
		ADD_BUILTIN(dropout_bp_tr, none,  INP|F,       INP|B,   OUT|F);
		#undef ADD_BUILTIN
	} catch (...) {
		g_extension_loading = old_extension_loading;
		throw;
	}
	g_extension_loading = old_extension_loading;
}

static opcode_t add_instruction(const char* mnemonic, int narg, iprops_t iprops, aprops_t aprops0, aprops_t aprops1, aprops_t aprops2, aprops_t aprops3, aprops_t aprops4)
{
	// First make sure the mnemonic is not already used by another instruction
	if (!s_builtin_added)
		register_builtin_instructions(); // Ensure that the built-in instructions get the first slots.
	
	auto entry = find_if(g_instruction_info.begin(),g_instruction_info.end(),
		[&mnemonic](const instruction_info& info) { return strcmp(info.mnemonic,mnemonic) == 0; });
	if (entry != g_instruction_info.end())
		SM_ERROR(format("AssertionError: Cannot insert instruction '%s' as that mnemonic already exists.",mnemonic).c_str());

	// Set up a new entry and add it to the registry
	instruction_info info;
	strncpy(info.mnemonic,mnemonic,32);
	info.mnemonic[32] = '\0';
	info.narg = narg;
	info.module = g_extension_loading;
	info.iprops = iprops;
	info.aprops[0] = aprops0;
	info.aprops[1] = aprops1;
	info.aprops[2] = aprops2;
	info.aprops[3] = aprops3;
	info.aprops[4] = aprops4;
	g_instruction_info.push_back(info);

	return (opcode_t)g_instruction_info.size()-1;
}

opcode_t add_instruction(const char* mnemonic, iprops_t iprops)
{
	return add_instruction(mnemonic,0,iprops,aprop_unused,aprop_unused,aprop_unused,aprop_unused,aprop_unused);
}

opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0)
{
	return add_instruction(mnemonic,1,iprops,aprops0,aprop_unused,aprop_unused,aprop_unused,aprop_unused);
}

opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0, aprops_t aprops1)
{
	return add_instruction(mnemonic,2,iprops,aprops0,aprops1,aprop_unused,aprop_unused,aprop_unused);
}

opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0, aprops_t aprops1, aprops_t aprops2)
{
	return add_instruction(mnemonic,3,iprops,aprops0,aprops1,aprops2,aprop_unused,aprop_unused);
}

opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0, aprops_t aprops1, aprops_t aprops2, aprops_t aprops3)
{
	return add_instruction(mnemonic,4,iprops,aprops0,aprops1,aprops2,aprops3,aprop_unused);
}

opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0, aprops_t aprops1, aprops_t aprops2, aprops_t aprops3, aprops_t aprops4)
{
	return add_instruction(mnemonic,5,iprops,aprops0,aprops1,aprops2,aprops3,aprops4);
}

void remove_instruction(opcode_t opcode)
{
	SM_ASSERT((size_t)opcode < g_instruction_info.size());
	SM_ASSERT((size_t)opcode >= (size_t)num_builtin_opcodes);
	memset(&g_instruction_info[opcode],0,sizeof(instruction_info));
}

void remove_instructions(dllhandle_t module)
{
	for (auto& i : g_instruction_info)
		if (i.module == module)
			memset(&i,0,sizeof(instruction_info));
}

const instruction_info& get_instruction_info(opcode_t opcode)
{
	if (g_instruction_info.empty())
		register_builtin_instructions();
	SM_ASSERT((size_t)opcode < g_instruction_info.size());
	return g_instruction_info[opcode];
}

//////////////////////////////////////////////////////////////////////////

void call_execute_fn(void* callback, const instruction& instr, int narg)
{
	if (narg < 0)
		narg = get_instruction_info(instr.opcode).narg;
	switch (narg) {
	case 1: ((execute_fn1)callback)(instr.opcode,instr.arg[0]); break;
	case 2: ((execute_fn2)callback)(instr.opcode,instr.arg[0],instr.arg[1]); break;
	case 3: ((execute_fn3)callback)(instr.opcode,instr.arg[0],instr.arg[1],instr.arg[2]); break;
	case 4: ((execute_fn4)callback)(instr.opcode,instr.arg[0],instr.arg[1],instr.arg[2],instr.arg[3]); break;
	case 5: ((execute_fn5)callback)(instr.opcode,instr.arg[0],instr.arg[1],instr.arg[2],instr.arg[3],instr.arg[4]); break;
	default: SM_UNREACHABLE();
	}
}

bool call_validate_fn(void* callback, const instruction& instr, int narg)
{
	if (narg < 0)
		narg = get_instruction_info(instr.opcode).narg;
	switch (narg) {
	case 1: return ((validate_fn1)callback)(instr.opcode,instr.arg[0]);
	case 2: return ((validate_fn2)callback)(instr.opcode,instr.arg[0],instr.arg[1]);
	case 3: return ((validate_fn3)callback)(instr.opcode,instr.arg[0],instr.arg[1],instr.arg[2]);
	case 4: return ((validate_fn4)callback)(instr.opcode,instr.arg[0],instr.arg[1],instr.arg[2],instr.arg[3]);
	case 5: return ((validate_fn5)callback)(instr.opcode,instr.arg[0],instr.arg[1],instr.arg[2],instr.arg[3],instr.arg[4]);
	default: SM_UNREACHABLE();
	}
}

void add_instruction_impl(int backend_uuid, opcode_t opcode, int narg, void* execute, void* validate)
{
	if (!s_builtin_added)
		register_builtin_instructions(); // Ensure that the built-in instructions get the first slots.

	// First make sure the instruction exists and has the right number of operands for the callbacks.
	if ((size_t)opcode >= g_instruction_info.size())
		SM_ERROR("AssertionError: Cannot add implementation for unrecognized instruction opcode.");
	auto& info = get_instruction_info(opcode);
	if (info.narg != narg)
		SM_ERROR("AssertionError: Incorrect number of operands in implementation callback.");

	// Next pull out the table for the machine for which these callbacks implement the instruction.
	if (backend_uuid >= (int)g_instruction_impl.size())
		g_instruction_impl.resize(backend_uuid+1);
	auto& db = g_instruction_impl[backend_uuid];

	// Make sure there's a slot for the specified instruction, by filling up the rest with empties.
	while (db.size() <= (size_t)opcode) {
		instruction_impl item = { 0,0,0 };
		db.push_back(item);
	}

	// Store the callbacks into that machine's "database"
	auto& impl = db[opcode];
	SM_ASSERTMSG(!impl.execute,"AssertionError: Cannot re-add an instruction implementation without removing the old one first.");
	impl.module = g_extension_loading;
	impl.execute = execute;
	impl.validate = validate;
}

void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn0 execute, execute_fn0 validate)
{
	add_instruction_impl(backend_uuid,opcode,0,(void*)execute,(void*)validate);
}

void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn1 execute, execute_fn1 validate)
{
	add_instruction_impl(backend_uuid,opcode,1,(void*)execute,(void*)validate);
}

void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn2 execute, execute_fn2 validate)
{
	add_instruction_impl(backend_uuid,opcode,2,(void*)execute,(void*)validate);
}

void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn3 execute, execute_fn3 validate)
{
	add_instruction_impl(backend_uuid,opcode,3,(void*)execute,(void*)validate);
}

void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn4 execute, execute_fn4 validate)
{
	add_instruction_impl(backend_uuid,opcode,4,(void*)execute,(void*)validate);
}

void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn5 execute, execute_fn5 validate)
{
	add_instruction_impl(backend_uuid,opcode,5,(void*)execute,(void*)validate);
}

void remove_instruction_impls(dllhandle_t module)
{
	for (auto& db : g_instruction_impl) {
		for (auto& impl : db) {
			if (impl.module == module) {
				impl.module = 0;
				impl.execute = 0;
				impl.validate = 0;
			}
		}
	}
}

static instruction_impl s_instruction_impl_none = { 0, 0, 0 };

const instruction_impl& get_instruction_impl(int backend_uuid, opcode_t opcode)
{
	if (backend_uuid >= (int)g_instruction_impl.size())
		return s_instruction_impl_none;
	const auto& db = g_instruction_impl[backend_uuid];
	if (opcode >= (int)db.size())
		return s_instruction_impl_none;
	return db[opcode];
}

SM_NAMESPACE_END
