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
#ifndef __SM_SPECIALIZATION_TABLE_H__
#define __SM_SPECIALIZATION_TABLE_H__

#include <smat/vm/instruction_db.h>
#include <base/typelist.h>
#include <vector>

SM_NAMESPACE_BEGIN

template <int dim, typename exec_fn>
class specialization_table {
public:
	specialization_table()
	{ 
		_size = 1;
		for (int d = 0; d < dim; ++d)
			_size *= num_dtypes;
		_table = new exec_fn[_size];
		memset(_table,0,_size*sizeof(exec_fn));
	}
	specialization_table(const specialization_table& src)
	{ 
		_size = src._size;
		_table = new exec_fn[_size];
		memcpy(_table,src._table,_size*sizeof(exec_fn));
	}
	~specialization_table() { delete[] _table; }
	SM_INLINE exec_fn& operator()(dtype_t dt0)                            { SM_ASSERT(dim==1); return *(exec_fn*)&_table[dt0]; }
	SM_INLINE exec_fn& operator()(dtype_t dt0, dtype_t dt1)               { SM_ASSERT(dim==2); return *(exec_fn*)&_table[dt0+dt1*num_dtypes]; }
	SM_INLINE exec_fn& operator()(dtype_t dt0, dtype_t dt1, dtype_t dt2)  { SM_ASSERT(dim==3); return *(exec_fn*)&_table[dt0+dt1*num_dtypes+dt2*num_dtypes*num_dtypes]; }
protected:
	int _size;
	exec_fn* _table;
};

////////////////////////////// for exec_functor<T> /////////////////////////////////////

template <template <typename T> class exec_functor, typename typesets>
struct specialization_table1_builder {
	template <typename table_type>
	static void build(table_type& table)
	{
		typedef typename typesets::type::type T;       // The dtype is in the first position of the current typeset item.
		table(ctype2dtype(T)) = &exec_functor<T>::execute;  // Store a pointer to this function specialization.
		specialization_table1_builder<exec_functor,typename typesets::next>::build(table); // Move on to the next type in the typeset list, thereby filling out the rest of the table.
	}
};

template <template <typename T> class exec_functor>
struct specialization_table1_builder<exec_functor,null_type> {
	template <typename table_type> static void build(table_type&) { } // We've reached the end of the typeset list, so we're done.
};

template <typename typesets, typename exec_fn, template <typename T> class exec_functor>
specialization_table<1,exec_fn> make_specialization_table()
{
	specialization_table<1,exec_fn> table;
	specialization_table1_builder<exec_functor,typesets>::build(table);
	return table;
}

////////////////////////////// for exec_functor<T0,T1> /////////////////////////////////////


template <template <typename T0, typename T1> class exec_functor, typename typesets>
struct specialization_table2_builder {
	template <typename table_type>
	static void build(table_type& table)
	{
		typedef typename typesets::type::type       T0;       // First dtype being specialized
		typedef typename typesets::type::next::type T1;       // Second dtype being specialized
		table(ctype2dtype(T0),ctype2dtype(T1)) = &exec_functor<T0,T1>::execute;  // Store a pointer to this function specialization.
		specialization_table2_builder<exec_functor,typename typesets::next>::build(table); // Move on to the next type in the typeset list, thereby filling out the rest of the table.
	}
};

template <template <typename T0, typename T1> class exec_functor>
struct specialization_table2_builder<exec_functor,null_type> {
	template <typename table_type> static void build(table_type&) { } // We've reached the end of the typeset list, so we're done.
};

template <typename typesets, typename exec_fn, template <typename T0, typename T1> class exec_functor>
specialization_table<2,exec_fn> make_specialization_table()
{
	specialization_table<2,exec_fn> table;
	specialization_table2_builder<exec_functor,typesets>::build(table);
	return table;
}

/////////////////////////////////////////////////////////

#define DECL_SPECIALIZATION_TABLE(typesets,exec_fn,...) \
	static _SM::specialization_table<typelist_len<typename typesets::type>::value,exec_fn> specialization_table(make_specialization_table<typesets,exec_fn,__VA_ARGS__>());

SM_NAMESPACE_END

#endif // __SM_SPECIALIZATION_TABLE_H__
