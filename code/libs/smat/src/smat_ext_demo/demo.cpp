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
#include <smat_cuda/launch_util.h>
#include <smat/vm/instruction_db.h>
#include <smat/vm/context.h>
#include <smat/smat.h>
#include <cstring>
using namespace sm;

struct clamp_args_t {
	double lo,hi;
};

void launch_lerp(dim3 gdim, dim3 bdim, unsigned smem, cudaStream_t stream,
                 usize_t size, dtype_t dtype,
                 const void* a, 
                 const void* b,
                       void* c,
                 double alpha);

void launch_clamp(dim3 gdim, dim3 bdim, unsigned smem, cudaStream_t stream,
                  usize_t size, dtype_t dtype,
                  void* a,
                  double lo, double hi);

// execution function:
//    Called when the virtual machine gets around to executing 
//    the lerp instruction, long after it was validated and inserted
//    into the execution stream. 
//    A CUDA execution function configures and calls a CUDA kernel.
//
void execute_lerp(opcode_t opcode, const argument& a,
                                   const argument& b,
                                   const argument& c,
                                   const argument& alpha)
{
	usize_t size = a.size();
	launchcfg cfg = make_elemwise_launchcfg(size);
	launch_lerp(cfg.gdim,cfg.bdim,cfg.smem,cfg.stream,
	                size,a.dtype,
	                a.get<const void*>(),
	                b.get<const void*>(),
	                c.get<      void*>(),
	                alpha.get<double>());

}

void execute_clamp(opcode_t opcode, const argument& a,
                                    const argument& b)
{
	const clamp_args_t* args = b.get<const clamp_args_t*>();
	usize_t size = a.size();
	launchcfg cfg = make_elemwise_launchcfg(size);
	launch_clamp(cfg.gdim,cfg.bdim,cfg.smem,cfg.stream,
	             size,a.dtype,
	             a.get<void*>(),
	             args->lo,args->hi);
}

// opcode_xxx
//    When the an instruction is registered, the machine returns an integer 
//    identifier (type opcode_t) for the instruction, so store it here.
//
opcode_t oc_lerp  = -1;
opcode_t oc_clamp = -1;

#pragma warning(disable : 4190 4297)  // disable warning about C linkage of shape_t, and about throwing exceptions from C functions

extern "C" {

// py_lerp: 
//    A C function that emits the 'lerp' instruction,
//    exported from the DLL, callable from Python via ctypes.
//
SM_DLLEXPORT smat* api_lerp(const smat* A, const smat* B, double alpha)
{ 
	SM_API_TRY
	smat* C = new smat(A->shape(),A->dtype());
	thread_ctx().emit(oc_lerp,A->as_arg(),   // emit instruction "lerp A,B,C,alpha" to the virtual machine
	                          B->as_arg(),
	                          C->as_arg(),
	                          carray(alpha));
	return C;
	SM_API_CATCH_AND_RETURN(0)
}

SM_DLLEXPORT smat* api_clamp(smat* A, const clamp_args_t* args)
{ 
	auto deleter = [](void* ptr) { delete (clamp_args_t*)ptr; };
	SM_API_TRY
	clamp_args_t* args_copy = new clamp_args_t(*args); // make a copy so that the calling code can do whatever it wants with its own args
	thread_ctx().emit(oc_clamp,A->as_arg(),            // emit instruction "clamp A,args" to the virtual machine
	                           user_arg(args_copy,deleter));
	return A;
	SM_API_CATCH_AND_RETURN(0)
}

// registration function:
//    After smat loads an extension DLL, the first thing it does is
//    look for an exported C function called "register_ext" and
//    calls it exactly once. The registration function notifies
//    smat of all instructions that this extension provides.
//    
SM_DLLEXPORT void register_ext()
{
	// We can define completely new instructions.
	// This demo provides a "lerp" and a "clamp" instruction.
	oc_lerp  = add_instruction("lerp",
			iprop_elemwise|             // flag for the optimizer that says "lerp is elementwise"
			iprop_match_all,            // flag for the machine that says "all operands should have matching dtypes"
			aprop_in |aprop_float,      // first  argument is floating-point input  (a)
			aprop_in |aprop_float,      // second argument is floating-point input  (b)
			aprop_out|aprop_float,      // third  argument is floating-point output (c)
			aprop_in |aprop_float       // fourth argument is floating-point input  (alpha)
		);

	oc_clamp = add_instruction("clamp",
			iprop_elemwise,
			aprop_in | aprop_out | aprop_float,      // first  argument (a) is floating-point array, both input and output
			aprop_in | aprop_user                    // second argument (hi,lo) is user structure, a pointer to an example_args_t struct
		);

	// Once the new instructions are in the database, we can provide
	// implementations for any particular backend (CUDA,MKL etc).
	// The way this works is to associate a callback with the (backend_uuid,opcode) pair.
	add_instruction_impl(cuda_uuid,oc_lerp ,execute_lerp ,0);   // register callbacks for executing/validating the "lerp" instruction
	add_instruction_impl(cuda_uuid,oc_clamp,execute_clamp,0);   // register callbacks for executing/validating the "clamp" instruction
}

}
