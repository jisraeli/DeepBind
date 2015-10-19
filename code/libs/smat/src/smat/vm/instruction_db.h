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
#ifndef __SM_INSTRUCTION_DB_H__
#define __SM_INSTRUCTION_DB_H__

#include <smat/vm/instruction.h>
#include <smat/dtypes.h>
#include <base/os.h>
#include <base/util.h>

SM_NAMESPACE_BEGIN

class context;

////////////////////////////////////////////////////////////////////////

class instruction_list: public std::list<instruction> { };

std::string instr2str(const instruction& instr);
std::string instr2str(const instruction_list& instr);
std::string instr2str(instruction_list::const_iterator first, instruction_list::const_iterator last);

// List of built-in instructions
enum builtin_opcode_t {
	oc_noop,  // allocate block
	oc_alloc,  // allocate block
	oc_free,   // free block
	oc_copy,   // elemwise copy     (with broadcasting)
	oc_rand,   // fill with uniformly distributed random values in range [0,1]
	oc_randn,  // fill with normally  distributed random values with mean 0 and variance 1
	oc_bernl,  // fill with bernoulli distributed random values with given expected value
	oc_add,    // elemwise add      (with broadcasting)
	oc_sub,    // elemwise subtract (with broadcasting)
	oc_mul,    // elemwise multiply (with broadcasting)
	oc_div,    // elemwise divide   (with broadcasting)
	oc_mod,    // elemwise modulo   (with broadcasting)
	oc_pow,    // elemwise power    (with broadcasting)
	oc_dot,    // matrix product of A and B
	oc_dottn,  // matrix product of A^T and B
	oc_dotnt,  // matrix product of A and B^T
	oc_dottt,  // matrix product of A^T and B^T
	oc_neg,    // elemwise negate
	oc_abs,    // elemwise absolute value
	oc_sign,   // elemwise sign {-1,0,+1}
	oc_signb,  // elemwise signbit {-1,+1}
	oc_sin,    // elemwise sine
	oc_cos,    // elemwise cosine
	oc_tan,    // elemwise tangent
	oc_asin,   // elemwise inverse sine
	oc_acos,   // elemwise inverse cosine
	oc_atan,   // elemwise inverse tangent
	oc_sinh,   // elemwise hyperbolic sine
	oc_cosh,   // elemwise hyperbolic cosine
	oc_tanh,   // elemwise hyperbolic tangent
	oc_asinh,  // elemwise inverse hyperbolic sine
	oc_acosh,  // elemwise inverse hyperbolic cosine
	oc_atanh,  // elemwise inverse hyperbolic tangent
	oc_exp,    // elemwise base-e exponential
	oc_exp2,   // elemwise base-2 exponential
	oc_log,    // elemwise base-e logarithm
	oc_log2,   // elemwise base-2  logarithm
	oc_sigm,   // elemwise logistic sigmoid 1/(1+exp(-x))
	oc_sqrt,   // elemwise square root
	oc_sqr,    // elemwise square
	oc_rnd,    // elemwise round
	oc_flr,    // elemwise floor
	oc_ceil,   // elemwise ceil
	oc_sat,    // elemwise saturate to range [0.0,1.0]
	oc_isinf,  // elemwise isinf test
	oc_isnan,  // elemwise isnan test
	oc_lnot,   // elemwise logical not
	oc_lor,    // elemwise logical or  (with broadcasting)
	oc_land,   // elemwise logical and (with broadcasting)
	oc_not,    // elemwise bitwise not
	oc_or,     // elemwise bitwise or  (with broadcasting)
	oc_and,    // elemwise bitwise and (with broadcasting)
	oc_xor,    // elemwise bitwise xor (with broadcasting)
	oc_eq,     // elemwise == test     (with broadcasting)
	oc_ne,     // elemwise != test     (with broadcasting)
	oc_lt,     // elemwise <  test     (with broadcasting)
	oc_le,     // elemwise <= test     (with broadcasting)
	oc_maxe,   // elemwise maximum     (with broadcasting)
	oc_mine,   // elemwise minimum     (with broadcasting)
	oc_max,    // maximum of all values
	oc_max_x,  // maximum along each row (result is col vec)
	oc_max_y,  // maximum value each col (result is row vec)
	oc_min,    // minimum value
	oc_min_x,  // minimum along each row (result is col vec)
	oc_min_y,  // minimum value each col (result is row vec)
	oc_sum,    // sum of all values
	oc_sum_x,  // sum along each row (result in col vec)
	oc_sum_y,  // sum along each col (result is row vec)
	oc_mean,   // mean of all values
	oc_mean_x, // mean along each row (result in col vec)
	oc_mean_y, // mean along each col (result is row vec)
	oc_nnz,    // count number of non-zero values
	oc_nnz_x,  // count number of non-zero values along each row (results in col vec)
	oc_nnz_y,  // count number of non-zero values along each col (results in row vec)
	oc_diff_x, // backward difference along each row
	oc_diff_y, // backward difference along each col
	oc_rep,    // repeat
	oc_tile,   // tile
	oc_trace,  // trace
	oc_trans,  // transpose
	oc_arang,  // like numpy arange (currently only generates consecutive integers)
	oc_mask,   // apply a mask
	oc_dropout_fp_tr, // dropout forwardprop training mode
	oc_dropout_bp_tr, // dropout backprop training mode
	num_builtin_opcodes
};  // If you add a new builtin instruction, be sure to update
    //   1) g_instr_info[] in instruction.cpp
    //   2) machine::exec(...) in machine.cpp

// aprops_t
//    Properties of a particular argument of an instruction.
//    These are checked by the built-in validation function 
//    for convenience, before calling custom validation (if any).
//
enum aprops_t {
	aprop_unused  = 0,
	aprop_in      = 1 << 0,  // can values be read from the argument data?
	aprop_out     = 1 << 1,  // can values be written to the argument's data?
	aprop_bool    = 1 << 2,  // can operands be logical?
	aprop_int     = 1 << 3,  // can operands be signed integral?
	aprop_uint    = 1 << 4,  // can operands be unsigned integral?
	aprop_float   = 1 << 5,  // can operands be float/double?
	aprop_user    = 1 << 6,  // must argument be user struct?
	aprop_strides = 1 << 7,  // can each argument have its own unique stride?
	aprop_num     =              aprop_int | aprop_uint | aprop_float,  // numeric type
	aprop_any     = aprop_bool | aprop_int | aprop_uint | aprop_float   // any type
};
SM_MASK_TYPE(aprops_t)

// iprops_t
//    Properties of a particular instruction.
//    The built-in optimizer will try to apply optimizations
//    based on what the instruction permits.
// 
enum iprops_t {
	iprop_none      = 0,
	iprop_commutes  = 1 << 0,  // is "op A,B>C" equivalent to  "op B,A>C" ?
	iprop_elemwise  = 1 << 1,  // is an elementwise operation?
	iprop_inplace   = 1 << 2,  // is it safe to transform "op A,B>C" to "op A,B>A" when A is not needed after? (likewise if B is not needed after)
	iprop_match_in  = 1 << 3,  // should we enforce dtypes to match among input operands?
	iprop_match_out = 1 << 4,  // should we enforce dtypes to match among output operands? 
	iprop_match_all = 1 << 5,  // should we enforce dtypes to match among all operands? (different from "match_in & match_out")
	iprop_reduce_x  = 1 << 6,  // does the instruction reduce along x dimension?
	iprop_reduce_y  = 1 << 7,  // does the instruction reduce along y dimension?
	iprop_reduce_z  = 1 << 8,  // does the instruction reduce along z dimension?
	iprop_reduce_xy = iprop_reduce_x+iprop_reduce_y,
	iprop_reduce_xz = iprop_reduce_x+iprop_reduce_z,
	iprop_reduce_yz = iprop_reduce_y+iprop_reduce_z,
	iprop_reduce    = iprop_reduce_x+iprop_reduce_y+iprop_reduce_z
};
SM_MASK_TYPE(iprops_t)

struct instruction_info {
	char        mnemonic[33]; // "add", or "sub"; up to 32 chars + null terminator
	int         narg;         // number of operands used
	dllhandle_t module;       // the module (DLL) from which this instruction was added
	iprops_t    iprops;       // instruction properties
	aprops_t    aprops[instruction::max_arg]; // argument properties.
};

// add_instruction:
//    Register a new instruction type into the database.
//    Typically called by the register_ext function of an
//    an extension DLL.
//
SM_EXPORT opcode_t add_instruction(const char* mnemonic, iprops_t iprops);
SM_EXPORT opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0);
SM_EXPORT opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0, aprops_t aprops1);
SM_EXPORT opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0, aprops_t aprops1, aprops_t aprops2);
SM_EXPORT opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0, aprops_t aprops1, aprops_t aprops2, aprops_t aprops3);
SM_EXPORT opcode_t add_instruction(const char* mnemonic, iprops_t iprops, aprops_t aprops0, aprops_t aprops1, aprops_t aprops2, aprops_t aprops3, aprops_t aprops4);

// remove_instruction:
//    Unregister an existing instruction.
//
SM_EXPORT void remove_instruction(opcode_t opcode);
SM_EXPORT void remove_instructions(dllhandle_t module);

// get_instruction:
//    Returns the info associated with an instruction opcode.
//
SM_EXPORT const instruction_info& get_instruction_info(opcode_t opcode);


typedef void (*execute_fn0)(opcode_t);
typedef void (*execute_fn1)(opcode_t, const argument&);
typedef void (*execute_fn2)(opcode_t, const argument&, const argument&);
typedef void (*execute_fn3)(opcode_t, const argument&, const argument&, const argument&);
typedef void (*execute_fn4)(opcode_t, const argument&, const argument&, const argument&, const argument&);
typedef void (*execute_fn5)(opcode_t, const argument&, const argument&, const argument&, const argument&, const argument&);

typedef bool (*validate_fn0)(opcode_t);
typedef bool (*validate_fn1)(opcode_t, const argument&);
typedef bool (*validate_fn2)(opcode_t, const argument&, const argument&);
typedef bool (*validate_fn3)(opcode_t, const argument&, const argument&, const argument&);
typedef bool (*validate_fn4)(opcode_t, const argument&, const argument&, const argument&, const argument&);
typedef bool (*validate_fn5)(opcode_t, const argument&, const argument&, const argument&, const argument&, const argument&);

SM_EXPORT void call_execute_fn(void* callback, const instruction& instr, int narg=-1); // if narg is omitted, it will be looked up by get_instruction_info
SM_EXPORT bool call_validate_fn(void* callback, const instruction& instr, int narg=-1); // if narg is omitted, it will be looked up by get_instruction_info

struct instruction_impl {
	dllhandle_t module;
	void*       execute;
	void*       validate;
};

// add_instruction_impl:
//    Register a machine-specific implementation of an instruction.
//    For example, the register_ext function of the "cuda" backend
//    can provide callbacks for cuda-specific implementation.
//
SM_EXPORT void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn0 execute, execute_fn0 validate=0);
SM_EXPORT void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn1 execute, execute_fn1 validate=0);
SM_EXPORT void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn2 execute, execute_fn2 validate=0);
SM_EXPORT void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn3 execute, execute_fn3 validate=0);
SM_EXPORT void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn4 execute, execute_fn4 validate=0);
SM_EXPORT void add_instruction_impl(int backend_uuid, opcode_t opcode, execute_fn5 execute, execute_fn5 validate=0);

// remove_instruction_impl:
//    Unregister all instructions implemented by the given module (DLL).
//
SM_EXPORT void remove_instruction_impls(dllhandle_t module);

// get_instruction_impl:
//    Return the implementation callbacks associated with the instruction and machine backend.
//
SM_EXPORT const instruction_impl& get_instruction_impl(int backend_uuid, opcode_t opcode);

////////////////////////////////////////////////////////////////////////

SM_NAMESPACE_END

#endif // __SM_INSTRUCTION_DB_H__
