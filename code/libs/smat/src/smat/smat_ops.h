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
#ifndef __SM_SMAT_OPS_H__
#define __SM_SMAT_OPS_H__

#include <smat/config.h>
#undef isinf // on linux it seems isinf is defined as a macro, uuuugh
#undef isnan
#undef arange
#undef nnz
#undef floor
#undef ceil
#undef sqrt
#undef arange
#undef swap
#undef round
#undef signbit
#undef neg
#undef abs


SM_NAMESPACE_BEGIN

#define SMAT_OPS_GLOBALS_FRIEND(mat) SMAT_OPS_GLOBALS_IMPL(mat,SM_EXPORT friend,,,,)
#define SMAT_OPS_GLOBALS(mat)        SMAT_OPS_GLOBALS_IMPL(mat,SM_EXPORT,=1,=default_dtype,=noaxis,=xaxis)

#define SMAT_OPS_GLOBALS_IMPL(mat,prefix,default_m,default_dt,default_axis_none,default_axis_x) \
	prefix mat operator+(const mat& A, const mat& B); \
	prefix mat operator-(const mat& A, const mat& B); \
	prefix mat operator*(const mat& A, const mat& B); \
	prefix mat operator/(const mat& A, const mat& B); \
	prefix mat operator%(const mat& A, const mat& B); \
	prefix mat pow(const mat& A, const mat& B); \
	prefix mat operator==(const mat& A, const mat& B); \
	prefix mat operator!=(const mat& A, const mat& B); \
	prefix mat operator<(const mat& A, const mat& B); \
	prefix mat operator<=(const mat& A, const mat& B); \
	prefix mat operator>(const mat& A, const mat& B); \
	prefix mat operator>=(const mat& A, const mat& B); \
	prefix mat operator||(const mat& A, const mat& B); \
	prefix mat operator&&(const mat& A, const mat& B); \
	prefix mat operator|(const mat& A, const mat& B); \
	prefix mat operator^(const mat& A, const mat& B); \
	prefix mat operator&(const mat& A, const mat& B); \
	prefix mat operator~(const mat& A); \
	prefix mat operator!(const mat& A); \
	prefix mat operator-(const mat& A); \
	prefix mat  dot(const mat& A, const mat& B); \
	prefix void dot(const smat& A, const smat& B, smat& out); \
	prefix mat  dot_tn(const mat& A, const mat& B); \
	prefix void dot_tn(const mat& A, const mat& B, smat& out); \
	prefix mat  dot_nt(const mat& A, const mat& B); \
	prefix void dot_nt(const mat& A, const mat& B, smat& out); \
	prefix mat  dot_tt(const mat& A, const mat& B); \
	prefix void dot_tt(const mat& A, const mat& B, smat& out); \
	prefix mat neg(const mat& A); \
	prefix mat abs(const mat& A); \
	prefix mat sign(const mat& A); \
	prefix mat signbit(const mat& A); \
	prefix mat sin(const mat& A); \
	prefix mat cos(const mat& A); \
	prefix mat tan(const mat& A); \
	prefix mat arcsin(const mat& A); \
	prefix mat arccos(const mat& A); \
	prefix mat arctan(const mat& A); \
	prefix mat sinh(const mat& A); \
	prefix mat cosh(const mat& A); \
	prefix mat tanh(const mat& A); \
	prefix mat arcsinh(const mat& A); \
	prefix mat arccosh(const mat& A); \
	prefix mat arctanh(const mat& A); \
	prefix mat exp(const mat& A); \
	prefix mat exp2(const mat& A); \
	prefix mat log(const mat& A); \
	prefix mat log2(const mat& A); \
	prefix mat sigm(const mat& A); \
	prefix mat logistic(const mat& A); \
	prefix mat sqrt(const mat& A); \
	prefix mat square(const mat& A); \
	prefix mat round(const mat& A); \
	prefix mat floor(const mat& A); \
	prefix mat ceil(const mat& A); \
	prefix mat clip(const mat& A, double lo, double hi); \
	prefix mat isinf(const mat& A); \
	prefix mat isnan(const mat& A); \
	prefix mat isclose(const mat& A, const mat& B, double rtol, double atol); \
	prefix mat allclose(const mat& A, const mat& B, double rtol, double atol); \
	prefix mat maximum(const mat& A, const mat& B); \
	prefix mat minimum(const mat& A, const mat& B); \
	prefix mat max(const mat& A, axis_t axis default_axis_none); \
	prefix mat min(const mat& A, axis_t axis default_axis_none); \
	prefix mat sum(const mat& A, axis_t axis default_axis_none); \
	prefix mat mean(const mat& A, axis_t axis default_axis_none); \
	prefix mat trace(const mat& A); \
	prefix mat nnz(const mat& A, axis_t axis default_axis_none); \
	prefix mat any(const mat& A, axis_t axis default_axis_none); \
	prefix mat all(const mat& A, axis_t axis default_axis_none); \
	prefix mat diff(const mat& A, axis_t axis default_axis_x); \
	prefix mat repeat(const mat& A, shape_t n); \
	prefix mat tile(const mat& A, shape_t n); \
	prefix mat trans(const mat& A); \
	prefix void swap(mat& A, mat& B); \
	prefix mat empty(shape_t shape,                  dtype_t dtype default_dt);  \
	prefix mat empty_like(const mat& A,              dtype_t dtype default_dt);  \
	prefix mat zeros(shape_t shape,                  dtype_t dtype default_dt);  \
	prefix mat zeros_like(const mat& A,              dtype_t dtype default_dt);  \
	prefix mat ones(shape_t shape,                   dtype_t dtype default_dt);  \
	prefix mat ones_like(const mat& A,               dtype_t dtype default_dt);  \
	prefix mat eye(isize_t n,                        dtype_t dtype default_dt);  \
	prefix mat arange(index_t stop,                  dtype_t dtype default_dt);  \
	prefix mat arange(index_t start, index_t stop,   dtype_t dtype default_dt);  \
	prefix mat rand(shape_t shape,                   dtype_t dtype default_dt);  \
	prefix mat randn(shape_t shape,                  dtype_t dtype default_dt);  \
	prefix mat bernoulli(shape_t shape, float p,     dtype_t dtype default_dt);  \
	/* now functions that do now have corresponding opcodes, but are executed as a combination of other instrutions */ \
	prefix mat softmax(const mat& A, axis_t axis default_axis_x); \
	prefix void apply_mask(mat& A, const mat& mask);

#define SMAT_OPS_MEMBERS(mat) \
	mat& operator+=(const mat& A); \
	mat& operator-=(const mat& A); \
	mat& operator*=(const mat& A); \
	mat& operator/=(const mat& A); \
	mat& operator%=(const mat& A); \
	mat& operator|=(const mat& A); \
	mat& operator^=(const mat& A); \
	mat& operator&=(const mat& A); \
	mat& ipow(const mat& A); 

SM_NAMESPACE_END

#endif // __SM_SMAT_OPS_H__
