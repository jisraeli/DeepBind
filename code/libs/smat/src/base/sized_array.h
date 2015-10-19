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
#ifndef __SM_SIZED_ARRAY_H__
#define __SM_SIZED_ARRAY_H__

#include <base/config.h>
#include <base/assert.h>
#include <new>
#ifdef _MSC_VER
#include <xutility>  // for std::input_iterator_tag
#else
#include <iterator>
#endif


SM_NAMESPACE_BEGIN

//////////////////////////////////////////////////////////////////////////

template <typename T, int N>
class sized_array {
public:
	SM_INLINE sized_array(): _size(0) {}
	~sized_array() {
		while (_size > 0)
			pop_back();
	}

	SM_INLINE int  size()  const { return _size; }
	SM_INLINE bool empty() const { return _size == 0; }
	SM_INLINE bool full()  const { return _size == N; }

	void resize(int newsize) {
		while (_size > newsize)
			pop_back();
		while (_size < newsize)
			push_back();
	}

	SM_INLINE void push_back()
	{
		SM_DBASSERT(_size < N);
		new (((T*)_data) + _size++) T(); // construct inplace
	}

	SM_INLINE void push_back(const T& item)
	{
		SM_DBASSERT(_size < N);
		new (((T*)_data) + _size++) T(item); // construct inplace
	}

	SM_INLINE void push_back(T&& item)
	{
		SM_DBASSERT(_size < N);
		new (((T*)_data) + _size++) T(move(item)); // construct inplace
	}

	SM_INLINE void pop_back()
	{
		SM_DBASSERT(_size > 0);
		((T*)_data)[--_size].~T();  // destruct inplace
	}

	SM_INLINE void push_front()              { insert(0,T()); }
	SM_INLINE void push_front(const T& item) { insert(0,item); }
	SM_INLINE void push_front(T&& item)      { insert(0,move(item)); }
	SM_INLINE void pop_front()               { erase(0);      }

	void insert(int pos, const T& item)
	{
		if (pos == _size) {
			push_back(item);
		} else {
			SM_DBASSERT(pos >= 0 && pos <= _size);
			SM_DBASSERT(_size < N);
			new (((T*)_data) + _size) T(move(((T*)_data)[_size-1])); // move-construct new final item from second-last item
			for (int i = _size-1; i > pos; --i)             // for each other item that follows 'index'
				((T*)_data)[i] = move(((T*)_data)[i-1]);             // shift the item down one slot in memory
			((T*)_data)[pos].~T();                                   // destruct what's left of old item at requested index
			new (((T*)_data) + pos) T(item);                         // copy item to requested index
			++_size;
		}
	}

	void insert(int pos, T&& item)
	{
		if (pos == _size) {
			push_back(move(item));
		} else {
			SM_DBASSERT(pos >= 0 && pos <= _size);
			SM_DBASSERT(_size < N);
			new (((T*)_data) + _size) T(move(((T*)_data)[_size-1])); // move-construct new final item from second-last item
			for (int i = _size-1; i > pos; --i)             // for each other item that follows 'index'
				((T*)_data)[i] = move(((T*)_data)[i-1]);             // shift the item down one slot in memory
			((T*)_data)[pos].~T();                                   // destruct what's left of old item at requested index
			new (((T*)_data)+pos) T(move(item));                   // move item to requested index
			++_size;
		}
	}

	void erase(int pos)
	{
		SM_DBASSERT(pos >= 0 && pos < _size);
		for (int i = pos; i < _size-1; ++i)           // for each item that follows 'index'
			((T*)_data)[i] = move(((T*)_data)[i+1]);  // move the next item down one slot in memory
		((T*)_data)[--_size].~T();                    // destruct the now-empty last item
	}

	SM_INLINE       T& operator[](int i)       { SM_DBASSERT(i >= 0 && i < _size); return ((T*)_data)[i]; }
	SM_INLINE const T& operator[](int i) const { SM_DBASSERT(i >= 0 && i < _size); return ((T*)_data)[i]; }

	SM_INLINE       T& at(int i)       { SM_DBASSERT(i >= 0 && i < _size); return ((T*)_data)[i]; }
	SM_INLINE const T& at(int i) const { SM_DBASSERT(i >= 0 && i < _size); return ((T*)_data)[i]; }

	class const_iterator {
	public:
		SM_INLINE const_iterator& operator++()    { ++_pos; return *this; }
		SM_INLINE const_iterator& operator--()    { --_pos; return *this; }
		SM_INLINE const_iterator  operator++(int) { return const_iterator(_pos++); }
		SM_INLINE const_iterator  operator--(int) { return const_iterator(_pos--); }
		SM_INLINE bool operator==(const const_iterator& other) const { return _pos == other._pos; }
		SM_INLINE bool operator!=(const const_iterator& other) const { return _pos != other._pos; }
		SM_INLINE const T& operator*() const { return *_pos; }

		typedef std::input_iterator_tag iterator_category;
		typedef T value_type;
		typedef void difference_type;
		typedef const T* pointer;
		typedef const T& reference;
	private:
		SM_INLINE const_iterator(const T* pos): _pos(pos) {}
		const T* _pos;
		friend class sized_array;
	};

	class iterator {
	public:
		SM_INLINE iterator& operator++()    { ++_pos; return *this; }
		SM_INLINE iterator& operator--()    { --_pos; return *this; }
		SM_INLINE iterator  operator++(int) { return iterator(_pos++); }
		SM_INLINE iterator  operator--(int) { return iterator(_pos--); }
		SM_INLINE bool operator==(const iterator& other) const { return _pos == other._pos; }
		SM_INLINE bool operator!=(const iterator& other) const { return _pos != other._pos; }
		SM_INLINE T& operator*() { return *_pos; }

		typedef std::input_iterator_tag iterator_category;
		typedef T value_type;
		typedef void difference_type;
		typedef T* pointer;
		typedef T& reference;
	private:
		SM_INLINE iterator(T* pos): _pos(pos) {}
		T* _pos;
		friend class sized_array;
	};

	SM_INLINE iterator begin() { return iterator((T*)_data); }
	SM_INLINE iterator end()   { return iterator((T*)_data+_size); }
	SM_INLINE const_iterator begin() const { return const_iterator((const T*)_data); }
	SM_INLINE const_iterator end()   const { return const_iterator((const T*)_data+_size); }

private:
	int  _size;
	char _padding[8-sizeof(int)]; // to make sure _data is aligned in case T is 64-bit
	char _data[N*sizeof(T)];
};

SM_NAMESPACE_END

#endif // __SM_SIZED_ARRAY_H__
