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
#ifndef __SM_RANGE_H__
#define __SM_RANGE_H__

#include <base/config.h>
#ifdef _MSC_VER
#include <xutility>  // for std::input_iterator_tag
#else
#include <iterator>
#endif

SM_NAMESPACE_BEGIN

template <typename T>
class _range {
public:
	class const_iterator {
	public:
		SM_INLINE const_iterator() {}
		SM_INLINE const_iterator(T pos, T step = 1): _pos(pos), _step(step) {}
		SM_INLINE const_iterator& operator++() { _pos += _step; return *this; }
		SM_INLINE const_iterator& operator--() { _pos -= _step; return *this; }
		SM_INLINE bool operator==(const const_iterator& other) const { return _pos == other._pos; }
		SM_INLINE bool operator!=(const const_iterator& other) const { return _pos != other._pos; }
		SM_INLINE const T& operator*() const { return _pos; }

		typedef std::input_iterator_tag iterator_category;
		typedef T value_type;
		typedef void difference_type;
		typedef const T* pointer;
		typedef const T& reference;
	private:
		T _pos;
		T _step;
	};

	SM_INLINE _range(T begin, T end, T step) : _begin(begin), _end(end), _step(step) {}
	SM_INLINE bool operator==(const _range &other) const { return _begin == other._begin && _end == other._end; }
	SM_INLINE const_iterator begin() const { return const_iterator(_begin,_step); }
	SM_INLINE const_iterator end()   const { return const_iterator(_end,_step); }

private:
	T _begin,_end,_step;
};

template <typename T> _range<T> range(T count)                { return _range<T>(0,count,1); }
template <typename T> _range<T> range(T begin, T end)         { return _range<T>(begin,end,1); }
template <typename T> _range<T> range(T begin, T end, T step) { return _range<T>(begin,end,step); }

SM_NAMESPACE_END

#endif // __SM_RANGE_H__
