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
#ifndef __SM_SMAT_H__
#define __SM_SMAT_H__

#include <smat/vm/instruction.h>
#include <smat/smat_ops.h>
#include <smat/shape.h>
#include <memory>

SM_NAMESPACE_BEGIN

using std::shared_ptr;
struct heap_alloc;

/////////////////////////////// smat ////////////////////////////////////

//
// smat
//   Row-major (C-order) streaming-mode matrix ("s" for "streaming").
//
class SM_EXPORT smat { SM_MOVEABLE(smat) SM_COPYABLE(smat)
public:
	//
	// view
	//    A proxy view into this matrix. 
	//    The entire purpose of this class is to facilititate numpy-like copy semantics:
	//        A(span) = B;    // copy B into A
	//        A       = B;    // A now refers to B
	//
	//    More examples:
	//        A(span) = 1;         // assign 1 all elements to 1
	//        A(10,20) = 0;        // assign 0 to elements in rows 10..19 (exclude 20)
	//        A = 5;               // make A be the 1x1 matrix containing scalar 5
	//
	class view {
	public:
		view& operator= (const smat& A) { smat T(_target,_rows,_cols); T.assign(A); return *this; }
		view& operator+=(const smat& A) { _target(_rows,_cols) += A; return *this; }
		view& operator-=(const smat& A) { _target(_rows,_cols) -= A; return *this; }
		view& operator*=(const smat& A) { _target(_rows,_cols) *= A; return *this; }
		view& operator/=(const smat& A) { _target(_rows,_cols) /= A; return *this; }
		view& operator%=(const smat& A) { _target(_rows,_cols) %= A; return *this; }
		view& operator&=(const smat& A) { _target(_rows,_cols) &= A; return *this; }
		view& operator|=(const smat& A) { _target(_rows,_cols) |= A; return *this; }
		view& operator^=(const smat& A) { _target(_rows,_cols) ^= A; return *this; }
		operator smat() { return smat(_target,_rows,_cols); }

	private:
		view(smat& target, slice_t rows, slice_t cols): _target(target), _rows(rows), _cols(cols) { }
		smat&   _target;
		slice_t _rows;
		slice_t _cols;
		friend class smat;
	};

	smat();
	smat(bool scalar);
	smat(int8_t   scalar);
	smat(uint8_t  scalar);
	smat(int16_t  scalar);
	smat(uint16_t scalar);
	smat(int32_t  scalar);
	smat(uint32_t scalar);
	smat(int64_t  scalar);
	smat(uint64_t scalar);
	smat(float    scalar);
	smat(double   scalar);
	smat(shape_t shape, dtype_t dtype = default_dtype);
	~smat();

	dtype_t  dtype()  const;
	shape_t  shape()  const;
	coord_t  strides() const;
	size_t   size()   const;
	bool     is_scalar() const;
	bool     is_fullstride() const;
	
	void reshape(shape_t shape);
	void resize(shape_t shape);

	void assign(const smat& A); // copy each element of A into this matrix
	smat copy() const;          // create a new copy of this matrix
	smat T() const;             // return transposed copy of this matrix

	      view operator()(slice_t rows, slice_t cols = span);
	const smat operator()(slice_t rows, slice_t cols = span) const;

	const struct argument& as_arg() const;
	template <typename Tx> Tx as_scalar() const;
	template <typename Tx> void copy_from(Tx* src, coord_t strides)       { SM_ASSERT(ctype2dtype(Tx) == _.dtype); copy_from_void(src,strides); }
	template <typename Tx> void copy_to  (Tx* dst, coord_t strides) const { SM_ASSERT(ctype2dtype(Tx) == _.dtype); copy_to_void(  dst,strides); }

	SMAT_OPS_MEMBERS(smat)

private:
	struct zeros_tag {};
	struct ones_tag {};
	struct identity_tag {};
	smat(shape_t shape, dtype_t dt, zeros_tag);
	smat(shape_t shape, dtype_t dt, ones_tag);
	smat(isize_t n, dtype_t dt, identity_tag);
	smat(const smat& src, slice_t rows, slice_t cols);
	void release_alloc();
	void coerce_to_supported_dtype();

	void copy_from_void(void* src, coord_t strides);
	void copy_to_void(  void* dst, coord_t strides) const;

	smat& inplace_op(opcode_t op, const smat& A);

	static smat elemwise(opcode_t opcode, const smat& A, dtype_t dt_out = default_dtype);
	static smat elemwise(opcode_t opcode, const smat& A, const smat& B, dtype_t dt_out = default_dtype);
	static smat reduce(opcode_t opcode, const smat& A);
	static smat diff(opcode_t opcode, const smat& A);

	argument _;                    // the shape, dtype, strides, and data address of this smat
	shared_ptr<heap_alloc> _alloc; // the original allocation that this smat has a view into
	
	SMAT_OPS_GLOBALS_FRIEND(smat)
};

SMAT_OPS_GLOBALS(smat)

/////////////////////////////////////////////////////////////////////////////

SM_INLINE dtype_t smat::dtype()   const { return _.dtype;        }
SM_INLINE size_t  smat::size()    const { return _.size();       }
SM_INLINE shape_t smat::shape()   const { return _.shape;        }
SM_INLINE coord_t smat::strides() const { return _.strides;      }
SM_INLINE bool    smat::is_scalar() const { return _.vtype == vt_carray && _.size() == 1; }
SM_INLINE smat    smat::T()       const { return trans(*this);   }

SM_INLINE       smat::view smat::operator()(slice_t rows, slice_t cols)       { return view(*this,rows,cols); }
SM_INLINE const smat       smat::operator()(slice_t rows, slice_t cols) const { return smat(*this,rows,cols); }

template <> SM_INLINE void smat::copy_from(void* src, coord_t strides)         { copy_from_void(src,strides); }
template <> SM_INLINE void smat::copy_to(  void* dst, coord_t strides)   const { copy_to_void(dst,strides); }


template <typename Tx>
Tx smat::as_scalar() const
{
	SM_ASSERT(_.vtype == vt_carray);
	switch (_.dtype) {
	case b8:  return (Tx)_.get<bool>();
	case i8:  return (Tx)_.get<int8_t>();
	case u8:  return (Tx)_.get<uint8_t>();
	case i16: return (Tx)_.get<int16_t>();
	case u16: return (Tx)_.get<uint16_t>();
	case i32: return (Tx)_.get<int32_t>();
	case u32: return (Tx)_.get<uint32_t>();
	case i64: return (Tx)_.get<int64_t>();
	case u64: return (Tx)_.get<uint64_t>();
	case f32: return (Tx)_.get<float>();
	case f64: return (Tx)_.get<double>();
	}
	SM_UNREACHABLE();
}

SM_NAMESPACE_END

#endif // __SM_SMAT_H__
