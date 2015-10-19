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
#include <smat_cuda/config.h>
#include <smat/vm/instruction_db.h>
#include <cstring>

SM_NAMESPACE_BEGIN

#define DECL_EXEC1(name) void execute_##name(opcode_t, const argument&);
#define DECL_EXEC2(name) void execute_##name(opcode_t, const argument&, const argument&);
#define DECL_EXEC3(name) void execute_##name(opcode_t, const argument&, const argument&, const argument&);
#define DECL_EXEC4(name) void execute_##name(opcode_t, const argument&, const argument&, const argument&, const argument&);
#define DECL_EXEC1V(name) DECL_EXEC1(name); void execute_##name(opcode_t, const argument&);
#define DECL_EXEC2V(name) DECL_EXEC2(name); void validate_##name(opcode_t, const argument&, const argument&);
#define DECL_EXEC3V(name) DECL_EXEC3(name); void validate_##name(opcode_t, const argument&, const argument&, const argument&);
#define DECL_EXEC4V(name) DECL_EXEC4(name); void validate_##name(opcode_t, const argument&, const argument&, const argument&, const argument&);

DECL_EXEC1(alloc_free)
DECL_EXEC2(copy)
DECL_EXEC2(elemwise2)
DECL_EXEC3(elemwise3)
DECL_EXEC1(rand)
DECL_EXEC2(bernoulli)
DECL_EXEC3(dot)
DECL_EXEC2(arange)
DECL_EXEC2(apply_mask)
DECL_EXEC2(diff)
DECL_EXEC2(repeat)
DECL_EXEC2(tile)
DECL_EXEC2(transpose)
DECL_EXEC2(reduce)
DECL_EXEC2(reduce_x)
DECL_EXEC2(reduce_y)
DECL_EXEC4V(dropout_fp_tr)
DECL_EXEC3V(dropout_bp_tr)


#pragma warning(disable : 4190 4297)  // disable warning about C linkage of shape_t, and about throwing exceptions from C functions
extern "C" {

void register_dropout();

// register_ext:
//    Adds execution/validation callbacks for CUDA-based implementations
//    of all the standard smat machine instructions.
//    
SM_DLLEXPORT void register_ext()
{
	#define REGISTER(opcode,exec) add_instruction_impl(cuda_uuid,oc_##opcode,execute_##exec);
	REGISTER(alloc, alloc_free);
	REGISTER(free,  alloc_free);
	REGISTER(copy,  copy);
	REGISTER(rand,  rand);
	REGISTER(randn, rand);
	REGISTER(bernl, bernoulli);
	REGISTER(add,   elemwise3);
	REGISTER(sub,   elemwise3);
	REGISTER(mul,   elemwise3);
	REGISTER(div,   elemwise3);
	REGISTER(mod,   elemwise3);
	REGISTER(pow,   elemwise3);
	REGISTER(dot,   dot);
	REGISTER(dottn, dot);
	REGISTER(dotnt, dot);
	REGISTER(dottt, dot);
	REGISTER(neg,   elemwise2);
	REGISTER(abs,   elemwise2);
	REGISTER(sign,  elemwise2);
	REGISTER(signb, elemwise2);
	REGISTER(sin,   elemwise2);
	REGISTER(cos,   elemwise2);
	REGISTER(tan,   elemwise2);
	REGISTER(asin,  elemwise2);
	REGISTER(acos,  elemwise2);
	REGISTER(atan,  elemwise2);
	REGISTER(sinh,  elemwise2);
	REGISTER(cosh,  elemwise2);
	REGISTER(tanh,  elemwise2);
	REGISTER(asinh, elemwise2);
	REGISTER(acosh, elemwise2);
	REGISTER(atanh, elemwise2);
	REGISTER(exp,   elemwise2);
	REGISTER(exp2,  elemwise2);
	REGISTER(log,   elemwise2);
	REGISTER(log2,  elemwise2);
	REGISTER(sigm,  elemwise2);
	REGISTER(sqrt,  elemwise2);
	REGISTER(sqr,   elemwise2);
	REGISTER(rnd,   elemwise2);
	REGISTER(flr,   elemwise2);
	REGISTER(ceil,  elemwise2);
	REGISTER(sat,   elemwise2);
	REGISTER(isinf, elemwise2);
	REGISTER(isnan, elemwise2);
	REGISTER(lnot,  elemwise2);
	REGISTER(lor,   elemwise3);
	REGISTER(land,  elemwise3);
	REGISTER(not,   elemwise2);
	REGISTER(or,    elemwise3);
	REGISTER(and,   elemwise3);
	REGISTER(xor,   elemwise3);
	REGISTER(eq,    elemwise3);
	REGISTER(ne,    elemwise3);
	REGISTER(lt,    elemwise3);
	REGISTER(le,    elemwise3);
	REGISTER(maxe,  elemwise3);
	REGISTER(mine,  elemwise3);
	REGISTER(max,   reduce);
	REGISTER(max_x, reduce_x);
	REGISTER(max_y, reduce_y);
	REGISTER(min,   reduce);
	REGISTER(min_x, reduce_x);
	REGISTER(min_y, reduce_y);
	REGISTER(sum,   reduce);
	REGISTER(sum_x, reduce_x);
	REGISTER(sum_y, reduce_y);
	REGISTER(mean,  reduce);
	REGISTER(mean_x,reduce_x);
	REGISTER(mean_y,reduce_y);
	REGISTER(nnz,   reduce);
	REGISTER(nnz_x, reduce_x);
	REGISTER(nnz_y, reduce_y);
	//REGISTER(diff_x,diff); // not yet implemented
	REGISTER(diff_y,diff);
	REGISTER(rep, repeat);
	REGISTER(tile,tile);
	//REGISTER(trace, 0);
	REGISTER(trans, transpose);
	REGISTER(arang, arange);
	REGISTER(mask,  apply_mask);

	add_instruction_impl(cuda_uuid,oc_dropout_fp_tr,execute_dropout_fp_tr,validate_dropout_fp_tr);
	add_instruction_impl(cuda_uuid,oc_dropout_bp_tr,execute_dropout_bp_tr,validate_dropout_bp_tr);
}

}

SM_NAMESPACE_END
