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
#include <smat_cuda/elemwise2.cuh>
#include <smat/vm/instruction_db.h>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>

SM_NAMESPACE_BEGIN

template <typename T> SM_DEVICE_INLINE T signfd(T x) { return (T)(x > 0) - (T)(x < 0); }
template <typename T> SM_DEVICE_INLINE T signi(T x)  { return (T)(x > 0) - (T)(x < 0); }
template <typename T> SM_DEVICE_INLINE T sqrti(T x)  { return (T)(::sqrtf((typename ctype2ftype(T))x)+0.5f); }
SM_DEVICE_INLINE double clampd(double x)  { return ::max(0.0,::min(1.0,x)); }
SM_DEVICE_INLINE float  lsigf(float  x)   { return 1/(1+::__expf(-x)); }
SM_DEVICE_INLINE double lsigd(double x)   { return 1/(1+::exp(-x)); }

// Define some macros to make, for example, "sqrt(x)" be shorthand for "out(i) = sqrt(arg(i))" 
#define EVAL_AS_X(f) A x = a[i]; x = x; b[j] = (f);
#define EVAL2(types,name,f1,f2)           types(name, EVAL_AS_X(f1), EVAL_AS_X(f2))
#define EVAL3(types,name,f1,f2,f3)        types(name, EVAL_AS_X(f1), EVAL_AS_X(f2), EVAL_AS_X(f3))
#define EVAL4(types,name,f1,f2,f3,f4)     types(name, EVAL_AS_X(f1), EVAL_AS_X(f2), EVAL_AS_X(f3), EVAL_AS_X(f4))
#define EVAL5(types,name,f1,f2,f3,f4,f5)  types(name, EVAL_AS_X(f1), EVAL_AS_X(f2), EVAL_AS_X(f3), EVAL_AS_X(f4), EVAL_AS_X(f5))

EVAL2(DEF_FF,sin,      ::__sinf(x),      ::sin(x))
EVAL2(DEF_FF,cos,      ::__cosf(x),      ::cos(x))
EVAL2(DEF_FF,tan,      ::__tanf(x),      ::tan(x))
EVAL2(DEF_FF,asin,     ::asinf(x),       ::asin(x))
EVAL2(DEF_FF,acos,     ::acosf(x),       ::acos(x))
EVAL2(DEF_FF,atan,     ::atanf(x),       ::atan(x))
EVAL2(DEF_FF,sinh,     ::sinhf(x),       ::sinh(x))
EVAL2(DEF_FF,cosh,     ::coshf(x),       ::cosh(x))
EVAL2(DEF_FF,tanh,     ::tanhf(x),       ::tanh(x))
EVAL2(DEF_FF,asinh,    ::asinhf(x),      ::asinh(x))
EVAL2(DEF_FF,acosh,    ::acoshf(x),      ::acosh(x))
EVAL2(DEF_FF,atanh,    ::atanhf(x),      ::atanh(x))
EVAL2(DEF_FF,exp,      ::__expf(x),      ::exp(x))
EVAL2(DEF_FF,exp2,      ::exp2f(x),     ::exp2(x))
EVAL2(DEF_FF,log,      ::__logf(x),      ::log(x))
EVAL2(DEF_FF,log2,      ::log2f(x),     ::log2(x))
EVAL2(DEF_FF,sigm,   _SM::lsigf(x), _SM::lsigd(x))
EVAL2(DEF_FF,sat ,::__saturatef(x),_SM::clampd(x))
EVAL3(DEF_SS,neg, -x, -x, -x)
EVAL3(DEF_ZZ,not, !x, ~x, ~x)
EVAL5(DEF_GG,abs,  x,    x<0?-x:x,         x,       ::fabsf(x),        ::fabs(x))
EVAL5(DEF_GG,sign, x,    signi(x),      x!=0,        signfd(x),        signfd(x))
EVAL5(DEF_GG,signb,false,   x < 0,         0,       signbit(x),        signbit(x))
EVAL5(DEF_GG,sqrt, x,    sqrti(x),  sqrti(x),       ::sqrtf(x),        ::sqrt(x))
EVAL5(DEF_GG,sqr,  x,         x*x,       x*x,              x*x,              x*x)
EVAL5(DEF_GG,rnd,  x,           x,         x,      ::roundf(x),       ::round(x))
EVAL5(DEF_GG,flr,  x,           x,         x,      ::floorf(x),       ::floor(x))
EVAL5(DEF_GG,ceil, x,           x,         x,       ::ceilf(x),        ::ceil(x))
EVAL5(DEF_GL,isinf,false,   false,     false,(bool)__isinff(x), (bool)__isinf(x))
EVAL5(DEF_GL,isnan,false,   false,     false,(bool)__isnanf(x), (bool)__isnan(x))
EVAL5(DEF_GL,lnot,    !x,    x==0,      x==0,        x == 0.0f,        x == 0.0)

void execute_elemwise2(opcode_t opcode, const argument& a, const argument& b)
{
	#define LAUNCH_CASE(types,matched,f,try_inplace)  \
		if (opcode == oc_##f) { \
			DECL_SPECIALIZATION_TABLE(types,execute_fn2,execute_elemwise2_typed<k_##f,try_inplace>::matched); \
			specialization_table(a.dtype)(opcode,a,b);  \
			return; \
		}
	LAUNCH_CASE(T_F,match,sin,true)
	LAUNCH_CASE(T_F,match,cos,true)
	LAUNCH_CASE(T_F,match,tan,true)
	LAUNCH_CASE(T_F,match,asin,true)
	LAUNCH_CASE(T_F,match,acos,true)
	LAUNCH_CASE(T_F,match,atan,true)
	LAUNCH_CASE(T_F,match,sinh,true)
	LAUNCH_CASE(T_F,match,cosh,true)
	LAUNCH_CASE(T_F,match,tanh,true)
	LAUNCH_CASE(T_F,match,asinh,true)
	LAUNCH_CASE(T_F,match,acosh,true)
	LAUNCH_CASE(T_F,match,atanh,true)
	LAUNCH_CASE(T_F,match,exp,true)
	LAUNCH_CASE(T_F,match,exp2,true)
	LAUNCH_CASE(T_F,match,log,true)
	LAUNCH_CASE(T_F,match,log2,true)
	LAUNCH_CASE(T_F,match,sigm,true)
	LAUNCH_CASE(T_F,match,sat,true)
	LAUNCH_CASE(T_S,match,neg,true)
	LAUNCH_CASE(T_G,match,abs,true)
	LAUNCH_CASE(T_G,match,sign,true)
#if SM_WANT_BOOL || SM_WANT_INT || SM_WANT_UINT 
	LAUNCH_CASE(T_Z,match,not,true)
#endif
	LAUNCH_CASE(T_F,match,signb,true)
	LAUNCH_CASE(T_G,match,sqrt,true)
	LAUNCH_CASE(T_G,match,sqr,true)
	LAUNCH_CASE(T_G,match,rnd,true)
	LAUNCH_CASE(T_G,match,flr,true)
	LAUNCH_CASE(T_G,match,ceil,true)
	LAUNCH_CASE(T_G,logical,isinf,false)
	LAUNCH_CASE(T_G,logical,isnan,false)
	LAUNCH_CASE(T_G,logical,lnot,false)
	SM_UNIMPLEMENTED()
}

SM_NAMESPACE_END
