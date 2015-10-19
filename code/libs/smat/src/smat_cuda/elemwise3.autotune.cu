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
#include <smat_cuda/reduce_y.cuh>
#include <base/typelist.h>

SM_NAMESPACE_BEGIN

using namespace std;

// Just "sum of floats" to tune kernel launch parameters for all reducers and all dtypes.
typedef execute_reduce_y_typed<reducer_sum>::general<float,float> reduce_y_functor;

size_t reduce_y_setup_args(const autotune_query& query, argument* arg)
{
	isize_t nx = query.q0;
	isize_t ny = query.q1;
	arg[0].shape.x = nx;
	arg[0].shape.y = ny;
	arg[1].shape.x = nx;
	arg[1].shape.y = 1;
	return (ny-1)*nx; // flop count
}

void autotune_reduce_y()
{
	typedef make_typelist<     // <kernel,bdx,wdx,wdy>
		make_intlist4<0, 32,1,1>::type,
		make_intlist4<0, 64,1,1>::type,
		make_intlist4<0,128,1,1>::type,
		make_intlist4<0, 32,2,1>::type,
		make_intlist4<0, 64,2,1>::type,
		make_intlist4<0,128,2,1>::type,

		make_intlist4<1, 32,1,1>::type,
		make_intlist4<1, 64,1,1>::type,
		make_intlist4<1,128,1,1>::type,
		make_intlist4<1, 32,1,2>::type,
		make_intlist4<1, 64,1,2>::type,
		make_intlist4<1,128,1,2>::type
	>::type psets;

	autotune_queries queries;
	const size_t c_max_dim  = 1 << 20;   // nx and ny in range [0,max_dim-1]
	const size_t c_max_size = 1 << 28;   // largest nx*ny matrix to be considered
	for (size_t i = 2; i <= c_max_dim; i *= 2) {
		for (size_t j = 2; j <= c_max_dim; j *= 2) {
			if (i*j <= c_max_size)
				queries.push_back(autotune_query((index_t)i,(index_t)j));
		}
	}
	queries.clear();
	queries.push_back(autotune_query(10,1000000));
	queries.push_back(autotune_query(1000000,10));

	autotuner tuner(oc_sum_y);
	tuner.sample<reduce_y_functor::launch,psets>(queries,reduce_y_setup_args);
	tuner.print();

	//tuner.save(__FILE__);
}

SM_NAMESPACE_END