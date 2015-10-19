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
#ifndef __SM_SMAT_SHAPE_H__
#define __SM_SMAT_SHAPE_H__

#include <smat/dtypes.h>
#include <base/assert.h>

SM_NAMESPACE_BEGIN

///////////////////////////// coord_t ///////////////////////////////////

struct SM_EXPORT coord_t {
	SM_INLINE coord_t()                               : x(0),y(0),z(0) { }
	SM_INLINE coord_t(index_t x)                      : x(x),y(0),z(0) { }
	SM_INLINE coord_t(index_t x, index_t y)           : x(x),y(y),z(0) { }
	SM_INLINE coord_t(index_t x, index_t y, index_t z): x(x),y(y),z(z) { }
	index_t x,y,z;
};

SM_INLINE bool    operator==(const coord_t& a, const coord_t& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
SM_INLINE bool    operator!=(const coord_t& a, const coord_t& b) { return a.x != b.x || a.y != b.y || a.z != b.z; }
SM_INLINE coord_t operator+ (const coord_t& a, const coord_t& b) { return coord_t(a.x+b.x,a.y+b.y,a.z+b.z); }
SM_INLINE coord_t operator- (const coord_t& a, const coord_t& b) { return coord_t(a.x-b.x,a.y-b.y,a.z-b.z); }

typedef coord_t strides_t;

///////////////////////////// shape_t ///////////////////////////////////

struct SM_EXPORT shape_t {
	SM_INLINE shape_t()                               : x(1),y(1),z(1) { }
	SM_INLINE shape_t(isize_t x)                      : x(x),y(1),z(1) { }
	SM_INLINE shape_t(isize_t x, isize_t y)           : x(x),y(y),z(1) { }
	SM_INLINE shape_t(isize_t x, isize_t y, isize_t z): x(x),y(y),z(z) { }
	SM_INLINE bool   empty()   const { return x*y*z == 0; }
	SM_INLINE usize_t size()   const { SM_DBASSERT(x>=0 && y>=0 && z>=0); return (usize_t)x*(usize_t)y*(usize_t)z; }
	SM_INLINE bool is_scalar() const { return x == 1 && y == 1 && z == 1; }
	SM_INLINE bool is_xvec()   const { return y == 1 && z == 1; }
	SM_INLINE bool is_yvec()   const { return x == 1 && z == 1; }
	SM_INLINE bool is_zvec()   const { return x == 1 && y == 1; }
	isize_t x,y,z;
};

typedef int axis_t;
enum { noaxis=-1, xaxis=0, yaxis=1, zaxis=2 };

SM_INLINE bool operator==(const shape_t& a, const shape_t& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
SM_INLINE bool operator!=(const shape_t& a, const shape_t& b) { return a.x != b.x || a.y != b.y || a.z != b.z; }

SM_INLINE shape_t operator+(const shape_t& a, const shape_t& b) { return shape_t(a.x+b.x,a.y+b.y,a.z+b.z); }
SM_INLINE shape_t operator-(const shape_t& a, const shape_t& b) { return shape_t(a.x-b.x,a.y-b.y,a.z-b.z); }
SM_INLINE shape_t operator*(const shape_t& a, const shape_t& b) { return shape_t(a.x*b.x,a.y*b.y,a.z*b.z); }
SM_INLINE shape_t operator/(const shape_t& a, const shape_t& b) { return shape_t(a.x/b.x,a.y/b.y,a.z/b.z); }

SM_EXPORT std::string shape2str(const shape_t& shape);
SM_EXPORT coord_t     fullstride(const shape_t& shape);

///////////////////////////// slice_t ///////////////////////////////////

struct SM_EXPORT slice_t {
	SM_INLINE slice_t(index_t first): first(first), last(first+1) { }
	SM_INLINE slice_t(index_t first, index_t last): first(first), last(last) { }
	SM_INLINE isize_t size() const { return last-first; }
	void bind(isize_t dim);
	index_t first,last;
};

static const index_t slice_end = sizeof(index_t) == 4 ? (index_t)0x7f000000ll : (index_t)0x7f00000000000000ll;

const slice_t span(0,slice_end);

SM_INLINE bool operator==(const slice_t& a, const slice_t& b) { return a.first == b.first && a.last == b.last; }
SM_INLINE bool operator!=(const slice_t& a, const slice_t& b) { return a.first != b.first || a.last != b.last; }

SM_NAMESPACE_END

#endif // __SM_SMAT_SHAPE_H__
