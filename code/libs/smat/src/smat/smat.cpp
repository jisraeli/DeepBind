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
#include <smat/smat.h>
#include <smat/vm/heap.h>
#include <smat/vm/machine.h>
#include <smat/vm/context.h>
//#include <smat/vm/instruction_db.h>
#include <base/assert.h>
#include <base/util.h>
#include <base/logging.h>

#define LOG_SMAT_EVENT(e)        SM_LOG("smat","[0x%012llx,%s,%s] \t%s",this,shape2str(_.shape).c_str(),dtype2str(_.dtype),e)
#define EMIT thread_ctx().emit
#define AXIS_OPCODE(op,axis) ((axis == noaxis) ? oc_##op : (axis == xaxis) ? oc_##op##_x : (axis == yaxis) ? oc_##op##_y : (opcode_t)-1)

SM_NAMESPACE_BEGIN

using namespace std;

////////////////////////////////////////////////////////////////////////////////

enum broadcast_type_t {
	bt_invalid,
	bt_x,
	bt_y,
	bt_xy,
	bt_none
};

broadcast_type_t broadcast_type(const smat& dst, const smat& src)
{
	shape_t a = dst.shape(), b = src.shape();
	if (a == b) return bt_none;
	if (b.x == 1   && b.y == 1)   return bt_xy;
	if (b.x == 1   && b.y == a.y) return bt_x;
	if (b.x == a.x && b.y == 1)   return bt_y;
	return bt_invalid;
}

#define SM_BROADCAST_ERROR(dst,src) SM_ERROR(format("BroadcastError: Matrix of dimensions %s cannot be broadcast into %s.",shape2str((src).shape()).c_str(),shape2str((dst).shape()).c_str()).c_str())
#define SM_ASSERT_DIM(condition) SM_ASSERTMSG(condition,   "BroadcastError: Matrix dimensions mismatched.")
#define SM_ASSERT_DIM_EQ(A,B)    SM_ASSERTMSG((A).shape() == (B).shape(), format("BroadcastError: Matrix dimensions %s and %s mismatched.",shape2str((A).shape()).c_str(),shape2str((B).shape()).c_str()).c_str())
#define SM_ASSERT_NOT_EMPTY(A)   SM_ASSERTMSG((A).size() > 0,"IndexError: Matrix must have positive size in each dimension.")
#define SM_ASSERT_BROADCASTABLE(dst,src) { if (broadcast_type(dst,src) == bt_invalid) SM_BROADCAST_ERROR(dst,src); }

////////////////////////////////////////////////////////////////////

smat::smat() { LOG_SMAT_EVENT("ctor()"); }

#define SMAT_CTOR_SCALAR(T) smat::smat(T scalar): _(carray(scalar)) { LOG_SMAT_EVENT("ctor(scalar)"); coerce_to_supported_dtype(); }

SMAT_CTOR_SCALAR(bool)
SMAT_CTOR_SCALAR(int8_t)
SMAT_CTOR_SCALAR(uint8_t)
SMAT_CTOR_SCALAR(int16_t)
SMAT_CTOR_SCALAR(uint16_t)
SMAT_CTOR_SCALAR(int32_t)
SMAT_CTOR_SCALAR(uint32_t)
SMAT_CTOR_SCALAR(int64_t)
SMAT_CTOR_SCALAR(uint64_t)
SMAT_CTOR_SCALAR(float)
SMAT_CTOR_SCALAR(double)

smat::smat(isize_t n, dtype_t dt, identity_tag): _(iarray(dt == default_dtype ? get_default_dtype() : dt,shape_t(n,n))) { LOG_SMAT_EVENT("ctor(identity)"); coerce_to_supported_dtype(); }

smat::smat(shape_t shape, dtype_t dt)
: _(darray(0,dt == default_dtype ? get_default_dtype() : dt,shape))
{
	LOG_SMAT_EVENT("ctor(shape,dt)");
	SM_ASSERT(shape.x >= 0);
	SM_ASSERT(shape.y >= 0);
	SM_ASSERT(shape.z >= 0);
	SM_ASSERTMSG((size_t)shape.size() == (size_t)shape.x*(size_t)shape.y*(size_t)shape.z,"OverflowError: integer overflow, shape is too large.");
	coerce_to_supported_dtype();
	_alloc = make_shared<heap_alloc>(thread_ctx().alloc((size_t)shape.size()*dtype_size(dt)));
	_.set(_alloc->addr);
}

smat::smat(const smat& src)
: _(src._)
, _alloc(src._alloc)
{
	LOG_SMAT_EVENT("ctor(src)");
}

smat::smat(const smat& src, slice_t rows, slice_t cols)
: _(src._)
, _alloc(src._alloc)
{
	LOG_SMAT_EVENT("ctor(src,rows,cols)");

	// Wrap negative indices to the end of rows/columns.
	rows.bind(_.shape.y);
	cols.bind(_.shape.x);

	// Now adapt to our new size, and point to the right start of memory
	_.shape.y = rows.size();
	_.shape.x = cols.size();
	_.set(_.get<size_t>() + (rows.first*_.strides.y + cols.first*_.strides.x)*dtype_size(_.dtype));
}

smat::smat(smat&& src)
: _(move(src._))
, _alloc(move(src._alloc))
{
	LOG_SMAT_EVENT("ctor(&&src)");
}

smat::~smat()
{
	LOG_SMAT_EVENT("dtor");
	release_alloc();
}

void smat::release_alloc()
{
	if (_alloc.use_count() == 1) {    // if we're the last one pointing to this allocation, tell the context to free it up.
		_ = argument();
		thread_ctx().free(*_alloc);
	}
	_alloc.reset();
}

bool smat::is_fullstride() const {
	return _.strides == fullstride(_.shape);
}

void smat::reshape(shape_t s)
{
	auto size = _.size();
	if (size > 0)
		SM_ASSERTMSG((_.strides.x == 1 && _.strides.y == _.strides.x*_.shape.x) || _.shape.y == 1,"ValueError: Cannot reshape a column-slice matrix inplace unless it's a row vector; make a copy first.");
	SM_ASSERTMSG(s.z==1,"NotImplementedError: 3D arrays not yet supported");
	if (s.x == -1 && s.y > 0) {
		s.x = (isize_t)(size/s.y);
	} else if (s.y == -1 && s.x > 0) {
		s.y = (isize_t)(size/s.x);
	}
	if (s.size() != size)
		SM_ERROR(format("ValueError: Cannot reshape %s to %s; different number of elements.",shape2str(_.shape).c_str(),shape2str(s).c_str()).c_str());
	_.shape = s;                // commit to new shape
	_.strides = fullstride(s);  // update strides to match new shape (assumes full-stride)
}

void smat::resize(shape_t s)
{
	if (_.shape == s) {
		// do nothing
	} else if (_.size() == s.size()) {
		reshape(s); // just change shape
	} else {
		// Reallocate array
		release_alloc();
		_alloc = make_shared<heap_alloc>(thread_ctx().alloc((size_t)s.size()*dtype_size(_.dtype)));
		_.set(_alloc->addr);
		_.shape = s;                // commit to new shape
		_.strides = fullstride(s);  // update strides to match new shape (assumes full-stride)
	}
}

smat& smat::operator=(const smat& src)
{
	LOG_SMAT_EVENT("operator=(src)");
	if (this != &src) {
		release_alloc();
		_ = src._;
		_alloc = src._alloc;
	}
	return *this;
}

smat& smat::operator=(smat&& src)
{
	LOG_SMAT_EVENT("operator=(&&src)");
	release_alloc();
	_ = move(src._);
	_alloc = move(src._alloc);
	return *this;
}

const argument& smat::as_arg() const { return _; }

void smat::assign(const smat& A)
{
	SM_ASSERT_BROADCASTABLE(*this,A);
	EMIT(oc_copy,A._,_);
}

smat smat::copy() const
{
	if (_.vtype == vt_darray) {
		smat C(_.shape,_.dtype);
		EMIT(oc_copy,_,C._);
		return C;
	} else if (_.vtype == vt_harray) {
		SM_UNIMPLEMENTED();
	} else {
		SM_DBASSERT(!_alloc);
		return *this; // if constant array, there's no allocation to speak of, so can just return an immediate copy
	}
}

void smat::copy_from_void(void* src, coord_t strides)
{
	if (strides == coord_t())
		strides = _.strides;
	EMIT(oc_copy,harray(src,_.dtype,_.shape,strides),_);
}

void smat::copy_to_void(void* dst, coord_t strides) const
{
	if (strides == coord_t())
		strides = _.strides;
	EMIT(oc_copy,_,harray(dst,_.dtype,_.shape,strides));
}

smat& smat::inplace_op(opcode_t op, const smat& A)
{
	SM_ASSERT_BROADCASTABLE(*this,A);
	if (A.is_scalar()) {
		// Handle some special cases for efficiency
		double a = A.as_scalar<double>();
		if (a == 0 && (op == oc_add || op == oc_sub)) return *this;
		if (a == 1 && (op == oc_mul || op == oc_div || op == oc_pow)) return *this;
		if (a == 0 && (op == oc_mul || op == oc_pow)){ assign(smat(0)); return *this; }
		if (a == 2 && (op == oc_pow))                { EMIT(oc_sqr,_,_); return *this; }
	}
	EMIT(op,_,A._,_);
	return *this;
}

smat& smat::operator+=(const smat& A) { return inplace_op(oc_add,A); }
smat& smat::operator-=(const smat& A) {	return inplace_op(oc_sub,A); }
smat& smat::operator*=(const smat& A) {	return inplace_op(oc_mul,A); }
smat& smat::operator/=(const smat& A) {	return inplace_op(oc_div,A); }
smat& smat::operator%=(const smat& A) {	return inplace_op(oc_mod,A); }
smat& smat::operator|=(const smat& A) {	return inplace_op(oc_or,A);  }
smat& smat::operator^=(const smat& A) {	return inplace_op(oc_xor,A); }
smat& smat::operator&=(const smat& A) {	return inplace_op(oc_and,A); }
smat& smat::ipow(const smat& A)       { return inplace_op(oc_pow,A); }

smat smat::elemwise(opcode_t opcode, const smat& A, dtype_t dt_out)
{
	if (dt_out == default_dtype)
		dt_out = A._.dtype;
	smat C(A.shape(),dt_out);
	EMIT(opcode,A._,C._);
	return C;
}

smat smat::elemwise(opcode_t opcode, const smat& A, const smat& B, dtype_t dt_out)
{
	broadcast_type_t bt_AB = broadcast_type(A,B);
	broadcast_type_t bt_BA = broadcast_type(B,A);
	if (dt_out == default_dtype) {
		if      (A._.vtype == vt_carray) dt_out = B._.dtype;
		else if (B._.vtype == vt_carray) dt_out = A._.dtype;
		else dt_out = arithmetic_result_dtype(A._.dtype,B._.dtype);
	}
	shape_t shape_out;
	if      (bt_AB != bt_invalid) shape_out = A.shape();
	else if (bt_BA != bt_invalid) shape_out = B.shape();
	else SM_BROADCAST_ERROR(A,B);

	smat C(shape_out,dt_out);
	EMIT(opcode,A._,B._,C._);
	return C;
}

smat smat::reduce(opcode_t opcode, const smat& A)
{
	dtype_t dt = A._.dtype;

	// Mimick numpy by promoting dtype to int32/uint32 for sums over bool,char,short.
	//
	if (opcode == oc_sum || opcode == oc_sum_x || opcode == oc_sum_y)
		if (dt == i8 || dt == i16)
			dt = i32;
		else if (dt == b8 || dt == u8 || dt == u16)
			dt = SM_WANT_UINT ? u32 : f32;

	// When counting non-zeros, always accumulate to an integer
	//
	if (opcode == oc_nnz || opcode == oc_nnz_x || opcode == oc_nnz_y)
		dt = sizeof(uindex_t) == sizeof(int) ? 
#if SM_WANT_UINT
			u32 : u64;
#elif SM_WANT_INT
			i32 : i64;
#else
			f32 : f32;
#endif

	// Mimick numpy by promoting dtype to float32/float64 for mean over integral types.
	//
	if (opcode == oc_mean || opcode == oc_mean_x || opcode == oc_mean_y)
		if (dt != f32 && dt != f64) 
			dt = (dt == i64 || dt == u64) && SM_WANT_DOUBLE ? f64 : get_default_dtypef();

	// Reduce all elements, only along each row, or only along each column?
	auto info = get_instruction_info(opcode);
	if ((info.iprops & iprop_reduce) == iprop_reduce) {   // reduce to a scalar
		smat C(shape_t(1,1),dt);
		EMIT(opcode,A._,C._);
		return C;
	} else if (info.iprops & iprop_reduce_y) {  // reduce along each col, resulting in row vector
		smat C(shape_t(A._.shape.x,1),dt);
		EMIT(opcode,A._,C._);
		return C;
	} else if (info.iprops & iprop_reduce_x) {  // reduce along each row, resulting in col vector
		smat C(shape_t(1,A._.shape.y),dt);
		EMIT(opcode,A._,C._);
		return C;
	}
	SM_ERROR("AssertionError: misconfigered instruction properties on reduce operation.\n");
}

void smat::coerce_to_supported_dtype()
{
	if (_.vtype == vt_carray) {
		if (_.dtype == f64 && !thread_ctx().is_supported(_.dtype)) {
			_.dtype = f32;
			_.set((float)_.get<double>());
		}
	} else if (!thread_ctx().is_supported(_.dtype))
		SM_ERROR(format("TypeError: Type '%s' not supported by current backend machine.\n",dtype2str(_.dtype)).c_str());
}

smat empty_like(const smat& A, dtype_t dt) { return empty(A._.shape,dt == default_dtype ? A._.dtype : dt); }
smat zeros_like(const smat& A, dtype_t dt) { return zeros(A._.shape,dt == default_dtype ? A._.dtype : dt); }
smat ones_like(const smat& A, dtype_t dt)  { return ones(A._.shape ,dt == default_dtype ? A._.dtype : dt); }
smat empty(shape_t shape, dtype_t dt) { return smat(shape,dt); }
smat zeros(shape_t shape, dtype_t dt) { smat A(shape,dt); A.assign(0); return A; }
smat ones (shape_t shape, dtype_t dt) { smat A(shape,dt); A.assign(1); return A; }
smat eye  (isize_t n,     dtype_t dt) { smat A(shape_t(n,n),dt); A.assign(smat(n,dt,smat::identity_tag())); return A; }
smat rand(shape_t shape,  dtype_t dt) { smat A(shape,dt); EMIT(oc_rand,A._); return A; }
smat randn(shape_t shape, dtype_t dt) { smat A(shape,dt); EMIT(oc_randn,A._); return A; }
smat bernoulli(shape_t shape, float p, dtype_t dt) { smat A(shape,dt); EMIT(oc_bernl,carray(p),A._); return A; }
smat arange(index_t stop, dtype_t dt) { return arange(0,stop,dt); }
smat arange(index_t start, index_t stop,  dtype_t dt)
{
	smat A(shape_t(stop-start,1),dt);
	EMIT(oc_arang,carray(start),A._);
	return A;
}

////////////////////////////////////////////////////////////////////////////////

SM_EXPORT smat dot(const smat& A, const smat& B)
{
	if (A._.shape.x != B._.shape.y)
		SM_ERROR(format("BroadcastError: Matrix dimensions %s and %s cannot undergo matrix product.\n",shape2str(A._.shape).c_str(),shape2str(B._.shape).c_str()).c_str());
	smat C(shape_t(B._.shape.x,A._.shape.y),arithmetic_result_dtype(A._.dtype,B._.dtype));
	EMIT(oc_dot,A._,B._,C._);
	return C;
}

SM_EXPORT void dot(const smat& A, const smat& B, smat& out)
{
	if (A._.shape.x != B._.shape.y)
		SM_ERROR(format("BroadcastError: Matrix dimensions %s and %s cannot undergo matrix product.\n",shape2str(A._.shape).c_str(),shape2str(B._.shape).c_str()).c_str());
	if (out._.dtype != arithmetic_result_dtype(A._.dtype,B._.dtype))
		SM_ERROR("TypeError: Output array has incompatible dtype with input arrays.");
	out.resize(shape_t(B._.shape.x,A._.shape.y));
	EMIT(oc_dot,A._,B._,out._);
}

SM_EXPORT smat dot_tn(const smat& A, const smat& B)
{
	if (A._.shape.y != B._.shape.y)
		SM_ERROR(format("BroadcastError: Matrix dimensions %s and %s cannot undergo A^T * B matrix product.\n",shape2str(A._.shape).c_str(),shape2str(B._.shape).c_str()).c_str());
	smat C(shape_t(B._.shape.x,A._.shape.x),arithmetic_result_dtype(A._.dtype,B._.dtype));
	EMIT(oc_dottn,A._,B._,C._);
	return C;
}

SM_EXPORT void dot_tn(const smat& A, const smat& B, smat& out)
{
	if (A._.shape.y != B._.shape.y)
		SM_ERROR(format("BroadcastError: Matrix dimensions %s and %s cannot undergo A^T * B matrix product.\n",shape2str(A._.shape).c_str(),shape2str(B._.shape).c_str()).c_str());
	if (out._.dtype != arithmetic_result_dtype(A._.dtype,B._.dtype))
		SM_ERROR("TypeError: Output array has incompatible dtype with input arrays.");
	out.resize(shape_t(B._.shape.x,A._.shape.x));
	EMIT(oc_dottn,A._,B._,out._);
}

SM_EXPORT smat dot_nt(const smat& A, const smat& B)
{
	if (A._.shape.x != B._.shape.x)
		SM_ERROR(format("BroadcastError: Matrix dimensions %s and %s cannot undergo A * B^T matrix product.\n",shape2str(A._.shape).c_str(),shape2str(B._.shape).c_str()).c_str());
	smat C(shape_t(B._.shape.y,A._.shape.y),arithmetic_result_dtype(A._.dtype,B._.dtype));
	EMIT(oc_dotnt,A._,B._,C._);
	return C;
}

SM_EXPORT void dot_nt(const smat& A, const smat& B, smat& out)
{
	if (A._.shape.x != B._.shape.x)
		SM_ERROR(format("BroadcastError: Matrix dimensions %s and %s cannot undergo A * B^T matrix product.\n",shape2str(A._.shape).c_str(),shape2str(B._.shape).c_str()).c_str());
	if (out._.dtype != arithmetic_result_dtype(A._.dtype,B._.dtype))
		SM_ERROR("TypeError: Output array has incompatible dtype with input arrays.");
	out.resize(shape_t(B._.shape.y,A._.shape.y));
	EMIT(oc_dotnt,A._,B._,out._);
}

SM_EXPORT smat dot_tt(const smat& A, const smat& B)
{
	if (A._.shape.y != B._.shape.x)
		SM_ERROR(format("BroadcastError: Matrix dimensions %s and %s cannot undergo A^T * B^T matrix product.\n",shape2str(A._.shape).c_str(),shape2str(B._.shape).c_str()).c_str());
	smat C(shape_t(B._.shape.y,A._.shape.x),arithmetic_result_dtype(A._.dtype,B._.dtype));
	EMIT(oc_dottt,A._,B._,C._);
	return C;
}

SM_EXPORT void dot_tt(const smat& A, const smat& B, smat& out)
{
	if (A._.shape.y != B._.shape.x)
		SM_ERROR(format("BroadcastError: Matrix dimensions %s and %s cannot undergo A^T * B^T matrix product.\n",shape2str(A._.shape).c_str(),shape2str(B._.shape).c_str()).c_str());
	if (out._.dtype != arithmetic_result_dtype(A._.dtype,B._.dtype))
		SM_ERROR("TypeError: Output array has incompatible dtype with input arrays.");
	out.resize(shape_t(B._.shape.y,A._.shape.x));
	EMIT(oc_dottt,A._,B._,out._);
}


SM_EXPORT smat operator+(const smat& A, const smat& B)  { return smat::elemwise(oc_add,A,B); }
SM_EXPORT smat operator-(const smat& A, const smat& B)  { return smat::elemwise(oc_sub,A,B); }
SM_EXPORT smat operator*(const smat& A, const smat& B)  { return smat::elemwise(oc_mul,A,B); }
SM_EXPORT smat operator/(const smat& A, const smat& B)  { return smat::elemwise(oc_div,A,B); }
SM_EXPORT smat operator%(const smat& A, const smat& B)  { return smat::elemwise(oc_mod,A,B); }
SM_EXPORT smat operator==(const smat& A, const smat& B) { return smat::elemwise(oc_eq ,A,B,dt_logical); }
SM_EXPORT smat operator!=(const smat& A, const smat& B) { return smat::elemwise(oc_ne ,A,B,dt_logical); }
SM_EXPORT smat operator< (const smat& A, const smat& B) { return smat::elemwise(oc_lt ,A,B,dt_logical); }
SM_EXPORT smat operator<=(const smat& A, const smat& B) { return smat::elemwise(oc_le ,A,B,dt_logical); }
SM_EXPORT smat operator> (const smat& A, const smat& B) { return smat::elemwise(oc_lt ,B,A,dt_logical); }
SM_EXPORT smat operator>=(const smat& A, const smat& B) { return smat::elemwise(oc_le ,B,A,dt_logical); }
SM_EXPORT smat operator||(const smat& A, const smat& B) { return smat::elemwise(oc_lor ,A,B,dt_logical); }
SM_EXPORT smat operator&&(const smat& A, const smat& B) { return smat::elemwise(oc_land,A,B,dt_logical); }
SM_EXPORT smat operator| (const smat& A, const smat& B) { return smat::elemwise(oc_or ,A,B); }
SM_EXPORT smat operator^ (const smat& A, const smat& B) { return smat::elemwise(oc_xor,A,B); }
SM_EXPORT smat operator& (const smat& A, const smat& B) { return smat::elemwise(oc_and,A,B); }
SM_EXPORT smat operator~(const smat& A) { return smat::elemwise(oc_not  ,A); }
SM_EXPORT smat operator!(const smat& A) { return smat::elemwise(oc_lnot ,A,dt_logical); }
SM_EXPORT smat operator-(const smat& A) { return smat::elemwise(oc_neg  ,A); }
SM_EXPORT smat abs(const smat& A)       { return smat::elemwise(oc_abs  ,A); }
SM_EXPORT smat sign(const smat& A)      { return smat::elemwise(oc_sign ,A); }
SM_EXPORT smat signbit(const smat& A)   { return smat::elemwise(oc_signb,A); }
SM_EXPORT smat sin(const smat& A)       { return smat::elemwise(oc_sin  ,A); }
SM_EXPORT smat cos(const smat& A)       { return smat::elemwise(oc_cos  ,A); }
SM_EXPORT smat tan(const smat& A)       { return smat::elemwise(oc_tan  ,A); }
SM_EXPORT smat arcsin(const smat& A)    { return smat::elemwise(oc_asin ,A); }
SM_EXPORT smat arccos(const smat& A)    { return smat::elemwise(oc_acos ,A); }
SM_EXPORT smat arctan(const smat& A)    { return smat::elemwise(oc_atan ,A); }
SM_EXPORT smat sinh(const smat& A)      { return smat::elemwise(oc_sinh ,A); }
SM_EXPORT smat cosh(const smat& A)      { return smat::elemwise(oc_cosh ,A); }
SM_EXPORT smat tanh(const smat& A)      { return smat::elemwise(oc_tanh ,A); }
SM_EXPORT smat arcsinh(const smat& A)   { return smat::elemwise(oc_asinh,A); }
SM_EXPORT smat arccosh(const smat& A)   { return smat::elemwise(oc_acosh,A); }
SM_EXPORT smat arctanh(const smat& A)   { return smat::elemwise(oc_atanh,A); }
SM_EXPORT smat exp(const smat& A)       { return smat::elemwise(oc_exp  ,A); }
SM_EXPORT smat exp2(const smat& A)      { return smat::elemwise(oc_exp2 ,A); }
SM_EXPORT smat log(const smat& A)       { return smat::elemwise(oc_log  ,A); }
SM_EXPORT smat log2(const smat& A)      { return smat::elemwise(oc_log2 ,A); }
SM_EXPORT smat sigm(const smat& A)      { return smat::elemwise(oc_sigm ,A); }
SM_EXPORT smat logistic(const smat& A)  { return smat::elemwise(oc_sigm ,A); }
SM_EXPORT smat sqrt(const smat& A)      { return smat::elemwise(oc_sqrt ,A); }
SM_EXPORT smat square(const smat& A)    { return smat::elemwise(oc_sqr  ,A); }
SM_EXPORT smat round(const smat& A)     { return smat::elemwise(oc_rnd  ,A); }
SM_EXPORT smat floor(const smat& A)     { return smat::elemwise(oc_flr  ,A); }
SM_EXPORT smat ceil(const smat& A)      { return smat::elemwise(oc_ceil ,A); }
SM_EXPORT smat isinf(const smat& A)     { return smat::elemwise(oc_isinf,A,dt_logical); }
SM_EXPORT smat isnan(const smat& A)     { return smat::elemwise(oc_isnan,A,dt_logical); }
SM_EXPORT smat isclose(const smat& A, const smat& B, double rtol, double atol)  { return abs(A-B) <= (atol + rtol*abs(B)); }
SM_EXPORT smat allclose(const smat& A, const smat& B, double rtol, double atol) { return all(isclose(A,B,rtol,atol)); }
SM_EXPORT smat maximum(const smat& A, const smat& B) { return smat::elemwise(oc_maxe,A,B); }
SM_EXPORT smat minimum(const smat& A, const smat& B) { return smat::elemwise(oc_mine,A,B); }
SM_EXPORT smat max(const smat& A, axis_t axis)  { return smat::reduce(AXIS_OPCODE(max,axis),A);  }
SM_EXPORT smat min(const smat& A, axis_t axis)  { return smat::reduce(AXIS_OPCODE(min,axis),A);  }
SM_EXPORT smat sum(const smat& A, axis_t axis)  { return smat::reduce(AXIS_OPCODE(sum,axis),A);  }
SM_EXPORT smat mean(const smat& A, axis_t axis) { return smat::reduce(AXIS_OPCODE(mean,axis),A);  }
SM_EXPORT smat nnz(const smat& A, axis_t axis)  { return smat::reduce(AXIS_OPCODE(nnz,axis),A);  }
SM_EXPORT smat any(const smat& A, axis_t axis)  { return nnz(A,axis) != 0; }
SM_EXPORT smat all(const smat& A, axis_t axis)
{
	if (axis == noaxis) return nnz(A,noaxis) == A.size();
	if (axis == xaxis)  return nnz(A,xaxis)  == A._.shape.x;
	if (axis == yaxis)  return nnz(A,yaxis)  == A._.shape.y;
	SM_ERROR("Unsupported axis in all().")
}

SM_EXPORT smat pow(const smat& A, const smat& B)
{
	if (B.is_scalar()) {
		double b = B.as_scalar<double>();
		if (b == 0) return ones(A._.shape,A._.dtype);
		if (b == 1) return A.copy();
		if (b == 2) return square(A);
	}
	return smat::elemwise(oc_pow,A,B);
}

SM_EXPORT smat clip(const smat& A, double lo, double hi)
{
	if (lo != 0 || hi != 1)
		SM_ERROR("NotImplementedError: Clip operation currently only supports lo=0, hi=1.\n");
	return smat::elemwise(oc_sat,A);
}

SM_EXPORT smat trans(const smat& A)
{
	smat C(shape_t(A._.shape.y,A._.shape.x),A._.dtype);
	EMIT(oc_trans,A._,C._);
	return C;
}

SM_EXPORT smat diff(const smat& A, axis_t axis)
{
	if (axis == xaxis) {
		smat C(shape_t(A._.shape.x-1,A._.shape.y),A._.dtype);
		EMIT(oc_diff_x,A._,C._);
		return C;
	} else if (axis == yaxis) {
		smat C(shape_t(A._.shape.x,A._.shape.y-1),A._.dtype);
		EMIT(oc_diff_y,A._,C._);
		return C;
	}
	SM_ERROR("Unsupported axis.")
}

SM_EXPORT smat rep_op(opcode_t oc, const smat& A, shape_t n)
{
	auto  s = A.shape();
	auto dt = A.dtype();

	// Special case 1: a dimension is zero
	if (n.x == 0) return empty(shape_t(0,s.y,s.z),dt);
	if (n.y == 0) return empty(shape_t(s.x,0,s.z),dt);
	if (n.z == 0) return empty(shape_t(s.x,s.y,0),dt);

	// Special case 2: all dimensions are one
	if (n.size() == 1) return A.copy(); // emulate numpy copy/view semantics here

	// General case.
	smat C(A.shape()*n,dt);
	EMIT(oc,A.as_arg(),C.as_arg());
	return C;
}

SM_EXPORT smat repeat(const smat& A, shape_t n) { return rep_op(oc_rep,A,n); }
SM_EXPORT smat tile(const smat& A, shape_t n)   { return rep_op(oc_tile,A,n); }
SM_EXPORT smat trace(const smat& A)             { return smat::reduce(oc_trace,A); }

SM_EXPORT void swap(smat& A, smat& B)
{
	// Simply swap representations; no need to emit copy instructions
	std::swap(A._,B._);
	std::swap(A._alloc,B._alloc);
}

SM_EXPORT smat softmax(const smat& A, axis_t axis)
{
	smat B = exp(A - max(A,axis));  // Adjust for softmax stability
	return B/sum(B,axis);           // Exponentiate and normalize each row/col.
}

SM_EXPORT void apply_mask(smat& A, const smat& mask)
{
	EMIT(oc_mask,A._,mask._);
}

///////////////////////////////////////////////////////////////////////////

#pragma warning(push)
#pragma warning(disable : 4190 4297)  // disable warning about C linkage of shape_t, and about throwing exceptions from C functions

extern "C" {

SM_EXPORT string g_smat_last_error;

SM_EXPORT const char* api_get_last_error()   { return g_smat_last_error.empty() ? nullptr : g_smat_last_error.c_str(); }
SM_EXPORT void        api_clear_last_error() { g_smat_last_error.clear(); }
SM_EXPORT void        api_set_debug_break(bool enabled)
{ 
#ifdef _WIN32
	g_want_debug_break = enabled;
#endif
}


SM_EXPORT smat* api_empty_like(const smat* A, dtype_t dt) { SM_API_TRY return new smat(move(empty_like(*A,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_zeros_like(const smat* A, dtype_t dt) { SM_API_TRY return new smat(move(zeros_like(*A,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_ones_like (const smat* A, dtype_t dt) { SM_API_TRY return new smat(move(ones_like (*A,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_empty(shape_t& shape, dtype_t dt) { SM_API_TRY return new smat(move(empty(shape,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_zeros(shape_t& shape, dtype_t dt) { SM_API_TRY return new smat(move(zeros(shape,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_ones (shape_t& shape, dtype_t dt) { SM_API_TRY return new smat(move(ones (shape,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_eye  (isize_t n,      dtype_t dt) { SM_API_TRY return new smat(move(eye  (n,    dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_arange(index_t start, index_t stop, dtype_t dt) { SM_API_TRY return new smat(move(arange(start,stop,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_rand (shape_t& shape, dtype_t dt) { SM_API_TRY return new smat(move(rand (shape,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_randn(shape_t& shape, dtype_t dt) { SM_API_TRY return new smat(move(randn(shape,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_bernoulli(shape_t& shape, float p, dtype_t dt) { SM_API_TRY return new smat(move(bernoulli(shape,p,dt))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_b8 (bool     val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_i8 (int8_t   val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_u8 (uint8_t  val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_i16(int16_t  val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_u16(uint16_t val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_i32(int32_t  val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_u32(uint32_t val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_i64(int64_t  val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_u64(uint64_t val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_f32(float    val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_const_f64(double   val) { SM_API_TRY return new smat(val); SM_API_CATCH_AND_RETURN(0) }

SM_EXPORT void    api_delete(const smat* M)                 { SM_API_TRY delete M;                 SM_API_CATCH }
SM_EXPORT isize_t api_nrow(const smat* M)                   { SM_API_TRY return M->shape().y;      SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT isize_t api_ncol(const smat* M)                   { SM_API_TRY return M->shape().x;      SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT size_t  api_size(const smat* M)                   { SM_API_TRY return M->size();         SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT void    api_shape(const smat* M, shape_t& shape)  { SM_API_TRY shape = M->shape();       SM_API_CATCH }
SM_EXPORT smat*   api_reshape(const smat* M, shape_t& shape){ SM_API_TRY smat* A = new smat(*M); A->reshape(shape); return A; SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT dtype_t api_dtype(const smat* M)                  { SM_API_TRY return M->dtype();        SM_API_CATCH_AND_RETURN(f32) }

SM_EXPORT smat* api_slice(smat* M, slice_t& rows, slice_t& cols) { SM_API_TRY return new smat((*M)(rows,cols)); SM_API_CATCH_AND_RETURN(0) }

SM_EXPORT void api_copy_from(    smat* M, void* src, isize_t rstride, isize_t cstride) { SM_API_TRY isize_t size = dtype_size(M->dtype()); return M->copy_from(src,strides_t(cstride/size,rstride/size)); SM_API_CATCH }
SM_EXPORT void api_copy_to(const smat* M, void* dst, isize_t rstride, isize_t cstride) { SM_API_TRY isize_t size = dtype_size(M->dtype()); return M->copy_to(dst,strides_t(cstride/size,rstride/size));   SM_API_CATCH }
SM_EXPORT void api_assign(smat* M, const smat* A)           { SM_API_TRY return M->assign(*A);     SM_API_CATCH }

SM_EXPORT smat* api_add(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(*A + *B));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_sub(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(*A - *B));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_mul(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(*A * *B));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_div(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(*A / *B));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_mod(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(*A % *B));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_pow(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(pow(*A,*B))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT void  api_iadd(smat* A, const smat* B)       { SM_API_TRY *A += *B; SM_API_CATCH }
SM_EXPORT void  api_isub(smat* A, const smat* B)       { SM_API_TRY *A -= *B; SM_API_CATCH }
SM_EXPORT void  api_imul(smat* A, const smat* B)       { SM_API_TRY *A *= *B; SM_API_CATCH }
SM_EXPORT void  api_idiv(smat* A, const smat* B)       { SM_API_TRY *A /= *B; SM_API_CATCH }
SM_EXPORT void  api_imod(smat* A, const smat* B)       { SM_API_TRY *A %= *B; SM_API_CATCH }
SM_EXPORT void  api_ipow(smat* A, const smat* B)       { SM_API_TRY A->ipow(*B); SM_API_CATCH }
SM_EXPORT smat* api_dot(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(dot(*A,*B))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_dot_tn(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(dot_tn(*A,*B))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_dot_nt(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(dot_nt(*A,*B))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_dot_tt(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(dot_tt(*A,*B))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT void  api_dot_out(const smat* A, const smat* B, smat* out)     { SM_API_TRY dot(*A,*B,*out);    SM_API_CATCH }
SM_EXPORT void  api_dot_tn_out(const smat* A, const smat* B, smat* out)  { SM_API_TRY dot_tn(*A,*B,*out); SM_API_CATCH }
SM_EXPORT void  api_dot_nt_out(const smat* A, const smat* B, smat* out)  { SM_API_TRY dot_nt(*A,*B,*out); SM_API_CATCH }
SM_EXPORT void  api_dot_tt_out(const smat* A, const smat* B, smat* out)  { SM_API_TRY dot_tt(*A,*B,*out); SM_API_CATCH }
SM_EXPORT smat* api_eq(const smat* A, const smat* B)   { SM_API_TRY return new smat(move(*A == *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_ne(const smat* A, const smat* B)   { SM_API_TRY return new smat(move(*A != *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_lt(const smat* A, const smat* B)   { SM_API_TRY return new smat(move(*A <  *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_le(const smat* A, const smat* B)   { SM_API_TRY return new smat(move(*A <= *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_gt(const smat* A, const smat* B)   { SM_API_TRY return new smat(move(*A >  *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_ge(const smat* A, const smat* B)   { SM_API_TRY return new smat(move(*A >= *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_or(const smat* A, const smat* B)   { SM_API_TRY return new smat(move(*A | *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_xor(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(*A ^ *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_and(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(*A & *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_lor(const smat* A, const smat* B)  { SM_API_TRY return new smat(move(*A || *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_land(const smat* A, const smat* B) { SM_API_TRY return new smat(move(*A && *B)); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT void  api_ior(smat* A, const smat* B)        { SM_API_TRY *A |= *B; SM_API_CATCH }
SM_EXPORT void  api_ixor(smat* A, const smat* B)       { SM_API_TRY *A ^= *B; SM_API_CATCH }
SM_EXPORT void  api_iand(smat* A, const smat* B)       { SM_API_TRY *A &= *B; SM_API_CATCH }
SM_EXPORT smat* api_not(const smat* A)       { SM_API_TRY return new smat(move(~(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_lnot(const smat* A)      { SM_API_TRY return new smat(move(!(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_neg(const smat* A)       { SM_API_TRY return new smat(move(-(*A)));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_abs(const smat* A)       { SM_API_TRY return new smat(move(abs(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_sign(const smat* A)      { SM_API_TRY return new smat(move(sign(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_signbit(const smat* A)   { SM_API_TRY return new smat(move(signbit(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_sin(const smat* A)       { SM_API_TRY return new smat(move(sin(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_cos(const smat* A)       { SM_API_TRY return new smat(move(cos(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_tan(const smat* A)       { SM_API_TRY return new smat(move(tan(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_arcsin(const smat* A)    { SM_API_TRY return new smat(move(arcsin(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_arccos(const smat* A)    { SM_API_TRY return new smat(move(arccos(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_arctan(const smat* A)    { SM_API_TRY return new smat(move(arctan(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_sinh(const smat* A)      { SM_API_TRY return new smat(move(sinh(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_cosh(const smat* A)      { SM_API_TRY return new smat(move(cosh(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_tanh(const smat* A)      { SM_API_TRY return new smat(move(tanh(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_arcsinh(const smat* A)   { SM_API_TRY return new smat(move(arcsinh(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_arccosh(const smat* A)   { SM_API_TRY return new smat(move(arccosh(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_arctanh(const smat* A)   { SM_API_TRY return new smat(move(arctanh(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_exp(const smat* A)       { SM_API_TRY return new smat(move(exp(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_exp2(const smat* A)      { SM_API_TRY return new smat(move(exp2(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_log(const smat* A)       { SM_API_TRY return new smat(move(log(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_log2(const smat* A)      { SM_API_TRY return new smat(move(log2(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_logistic(const smat* A)  { SM_API_TRY return new smat(move(logistic(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_sqrt(const smat* A)      { SM_API_TRY return new smat(move(sqrt(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_square(const smat* A)    { SM_API_TRY return new smat(move(square(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_round(const smat* A)     { SM_API_TRY return new smat(move(round(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_floor(const smat* A)     { SM_API_TRY return new smat(move(floor(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_ceil(const smat* A)      { SM_API_TRY return new smat(move(ceil(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_clip(const smat* A, double lo, double hi)  { SM_API_TRY return new smat(move(clip(*A,lo,hi))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_isinf(const smat* A)     { SM_API_TRY return new smat(move(isinf(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_isnan(const smat* A)     { SM_API_TRY return new smat(move(isnan(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_isclose(const smat* A, const smat* B, double rtol, double atol)  { SM_API_TRY return new smat(move(isclose(*A,*B,rtol,atol))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_allclose(const smat* A, const smat* B, double rtol, double atol)  { SM_API_TRY return new smat(move(allclose(*A,*B,rtol,atol))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_maximum(const smat* A, const smat* B) { SM_API_TRY return new smat(move(maximum(*A,*B))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_minimum(const smat* A, const smat* B) { SM_API_TRY return new smat(move(minimum(*A,*B))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_max(const smat* A, axis_t axis)       { SM_API_TRY return new smat(move(max(*A,axis)));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_min(const smat* A, axis_t axis)       { SM_API_TRY return new smat(move(min(*A,axis)));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_sum(const smat* A, axis_t axis)       { SM_API_TRY return new smat(move(sum(*A,axis)));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_mean(const smat* A, axis_t axis)      { SM_API_TRY return new smat(move(mean(*A,axis)));  SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_nnz(const smat* A, axis_t axis)       { SM_API_TRY return new smat(move(nnz(*A,axis)));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_all(const smat* A, axis_t axis)       { SM_API_TRY return new smat(move(all(*A,axis)));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_any(const smat* A, axis_t axis)       { SM_API_TRY return new smat(move(any(*A,axis)));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_diff(const smat* A, axis_t axis)      { SM_API_TRY return new smat(move(diff(*A,axis)));  SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_repeat(const smat* A, shape_t& n)     { SM_API_TRY return new smat(move(repeat(*A,n)));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_tile(const smat* A, shape_t& n)       { SM_API_TRY return new smat(move(tile(*A,n)));     SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_trace(const smat* A)                  { SM_API_TRY return new smat(move(trace(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_trans(const smat* A)                  { SM_API_TRY return new smat(move(trans(*A))); SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT smat* api_softmax(const smat* A, axis_t axis)   { SM_API_TRY return new smat(move(softmax(*A,axis)));   SM_API_CATCH_AND_RETURN(0) }
SM_EXPORT void  api_apply_mask(smat* A, const smat* mask) { SM_API_TRY apply_mask(*A,*mask); SM_API_CATCH }

SM_EXPORT void  api_dropout_fp_tr(const smat* X, double rate, smat* Z, smat* M)
{ 
	SM_API_TRY
	EMIT(oc_dropout_fp_tr, X->as_arg(),carray(rate),Z->as_arg(),M->as_arg());
	SM_API_CATCH
}

SM_EXPORT void api_dropout_bp_tr(const smat* dZ, const smat* M, smat* dX)
{ 
	SM_API_TRY
	EMIT(oc_dropout_bp_tr,dZ->as_arg(),M->as_arg(),dX->as_arg());
	SM_API_CATCH
}


}

#pragma warning(pop)

SM_NAMESPACE_END
