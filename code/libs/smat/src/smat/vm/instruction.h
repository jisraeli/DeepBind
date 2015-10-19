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
#ifndef __SM_INSTRUCTION_H__
#define __SM_INSTRUCTION_H__

#include <smat/vm/argument.h>
#include <string>
#include <list>

SM_NAMESPACE_BEGIN

typedef int opcode_t;

// instruction
//   Instruction specification.
//
struct SM_EXPORT instruction { SM_MOVEABLE(instruction) SM_NOCOPY(instruction)
public:
#ifdef SM_CPP11
	typedef argument&& operand_ref;
#else
	typedef argument& operand_ref;
#endif
	enum { max_arg = 6 };
	instruction(opcode_t opcode);
	instruction(opcode_t opcode, operand_ref arg0);
	instruction(opcode_t opcode, operand_ref arg0, operand_ref arg1);
	instruction(opcode_t opcode, operand_ref arg0, operand_ref arg1, operand_ref arg2);
	instruction(opcode_t opcode, operand_ref arg0, operand_ref arg1, operand_ref arg2, operand_ref arg3);
	instruction(opcode_t opcode, operand_ref arg0, operand_ref arg1, operand_ref arg2, operand_ref arg3, operand_ref arg4);
	~instruction();

	opcode_t opcode;       // opcode for instruction
	argument arg[max_arg]; // operands for instruction
};

SM_NAMESPACE_END

#endif // __SM_INSTRUCTION_H__
