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
#ifndef __SM_AUTOTUNE_H__
#define __SM_AUTOTUNE_H__

#include <smat/dtypes.h>
#include <smat/shape.h>
#include <smat/vm/instruction_db.h>
#include <smat/vm/heap.h>
#include <smat/vm/context.h>
#include <base/typelist.h>
#include <base/time.h>
#include <map>
#include <limits>
#include <algorithm>

SM_NAMESPACE_BEGIN

using std::vector;
using std::map;

// autotune_query:
//    Specifies input conditions of an operation.
//    It's up to the user to make sure ophash is unique for their purposes.
//    The user values q0..q3 are used to describe the conditions;
//    For example, if the operation being tuned were matrix multiply, then
//    (q0,q1) can be set to the size of an input matrix.
//
struct autotune_query {
	SM_INLINE autotune_query()                                              : q0( 0),q1( 0),q2( 0),q3( 0) { }
	SM_INLINE autotune_query(index_t q0)                                    : q0(q0),q1( 0),q2( 0),q3( 0) { }
	SM_INLINE autotune_query(index_t q0, index_t q1)                        : q0(q0),q1(q1),q2( 0),q3( 0) { }
	SM_INLINE autotune_query(index_t q0, index_t q1, index_t q2)            : q0(q0),q1(q1),q2(q2),q3( 0) { }
	SM_INLINE autotune_query(index_t q0, index_t q1, index_t q2, index_t q3): q0(q0),q1(q1),q2(q2),q3(q3) { }
	index_t q0,q1,q2,q3;
};

SM_INLINE bool operator<(const autotune_query& a, const autotune_query& b)
{ 
	if (a.q0 != b.q0) return a.q0 < b.q0;
	if (a.q1 != b.q1) return a.q1 < b.q1;
	if (a.q2 != b.q2) return a.q2 < b.q2;
	if (a.q3 != b.q3) return a.q3 < b.q3;
	return false;
}

// autotune_pset:
//    Holds configuration parameters (p0..p3) for some operation.
//    For example, if the operation being tuned were CUDA matrix multiply,
//    then p0 might identify the particular kernel version to use, 
//    and (p1,p2) might be the block size for a launch.
//
struct autotune_pset {
	SM_INLINE autotune_pset()                                              : p0( 0),p1( 0),p2( 0),p3( 0) { }
	SM_INLINE autotune_pset(index_t p0)                                    : p0(p0),p1( 0),p2( 0),p3( 0) { }
	SM_INLINE autotune_pset(index_t p0, index_t p1)                        : p0(p0),p1(p1),p2( 0),p3( 0) { }
	SM_INLINE autotune_pset(index_t p0, index_t p1, index_t p2)            : p0(p0),p1(p1),p2(p2),p3( 0) { }
	SM_INLINE autotune_pset(index_t p0, index_t p1, index_t p2, index_t p3): p0(p0),p1(p1),p2(p2),p3(p3) { }
	index_t p0,p1,p2,p3;
};

typedef vector<autotune_query>  autotune_queries;
typedef vector<autotune_pset>   autotune_psets;
typedef vector<double>          autotune_scores;

class SM_EXPORT autotuner { SM_NOCOPY(autotuner)
	typedef map<autotune_query,autotune_scores> score_map;
public:
	autotuner(opcode_t opcode);
	~autotuner();


	template <template <unsigned p0, unsigned p1, unsigned p2, unsigned p3> class exec_fn, typename psets>
	void sample(const autotune_queries& queries, size_t (*setup_args)(const autotune_query&, argument*))
	{
		add_psets_4<psets>();

		instruction instr(_opcode);
		argument* arg = instr.arg;
		heap_alloc allocs[instruction::max_arg];
		int narg = get_instruction_info(_opcode).narg;
		for (int i = 0; i < narg; ++i) {
			arg[i].dtype = f32;
			arg[i].vtype = vt_darray; // defaults that config_args can override
			arg[i].shape = shape_t();
		}

		// For each query, generate an appropriate set of arguments, and 
		// calculate the performance of the execute function for each 
		// compile-time parameterset 
		for (autotune_queries::const_iterator q = queries.begin(); q != queries.end(); ++q) {

			// Configure the args and allocate any memory they need
			size_t flop = setup_args(*q,arg);
			for (int i = 0; i < narg; ++i) {
				arg[i].strides = coord_t(1,arg[i].shape.x,arg[i].shape.y);
				if (arg[i].vtype == vt_darray && !arg[i].shape.empty()) {
					allocs[i] = thread_ctx().alloc(arg[i].shape,arg[i].dtype);
					arg[i].set(allocs[i].addr);
				}
			}

			// We have now configured and allocated some dummy arguments that
			// the execute function can operate on, so run exec_fn with each pset entry
			// and record the GFLOPS.
			autotune_scores results;
			sampler<exec_fn,psets>::sample(*q,instr,narg,flop,results);
			_scores.insert(make_pair(*q,results));

			// Release any memory that was allocated.
			for (int i = 0; i < narg; ++i) {
				if (arg[i].vtype == vt_darray && !arg[i].shape.empty()) {
					arg[i].set((void*)0);
					thread_ctx().free(allocs[i]);
				}
			}
		}
	}

	void print_best();
	void print_all();

private:

	template <typename psets> void add_psets_4();

	template <template <unsigned p0, unsigned p1, unsigned p2, unsigned p3> class exec_fn, typename psets>
	struct sampler {
		typedef typename psets::type pset;
		static void sample(const autotune_query& query, const instruction& instr, int narg, size_t flop, autotune_scores& scores)
		{
			const unsigned p0 = typelist_get<pset,0>::type::value;
			const unsigned p1 = typelist_get<pset,1>::type::value;
			const unsigned p2 = typelist_get<pset,2>::type::value;
			const unsigned p3 = typelist_get<pset,3>::type::value;
			double score = 0.0;
			void* validate = (void*)&exec_fn<p0,p1,p2,p3>::validate;
			if (call_validate_fn(validate,instr,narg)) {
				void* exec = (void*)&exec_fn<p0,p1,p2,p3>::execute;
				call_execute_fn(exec,instr,narg);  // execute a couple of times and throw away result, to reduce timing variance
				const int ntrial = 5;
				const int nrepeat = 25;
				ticks_t trial_ticks[ntrial];
				for (int t = 0; t < ntrial; ++t) {
					thread_ctx().sync();
					ticks_t tic = ticks();
					for (int i = 0; i < nrepeat; ++i)
						call_execute_fn(exec,instr,narg);
					thread_ctx().sync();
					ticks_t toc = ticks();
					trial_ticks[t] = toc-tic;
				}
				std::sort(trial_ticks,trial_ticks+ntrial);
				ticks_t median = trial_ticks[ntrial/2];
				score = flop / (duration(median)/nrepeat) / 1e9; // GFLOPS
			}
			scores.push_back(score);
			sampler<exec_fn,typename psets::next>::sample(query,instr,narg,flop,scores); 
		}
	};

	template <template <unsigned p0, unsigned p1, unsigned p2, unsigned p3> class exec_fn>
	struct sampler<exec_fn,null_type> { static void sample(const autotune_query&, const instruction&, int, size_t, autotune_scores&) { } };

	opcode_t       _opcode;
	autotune_psets _psets;
	score_map      _scores;
};


template <typename psets>
void autotuner::add_psets_4()
{
	typedef typename psets::type pset;
	const unsigned p0 = typelist_get<pset,0>::type::value;
	const unsigned p1 = typelist_get<pset,1>::type::value;
	const unsigned p2 = typelist_get<pset,2>::type::value;
	const unsigned p3 = typelist_get<pset,3>::type::value;
	_psets.push_back(autotune_pset(p0,p1,p2,p3));
	add_psets_4<typename psets::next>();
}
template <> SM_INLINE void autotuner::add_psets_4<null_type>() { }


///////////////////////////////////////////////////////////////////////

class SM_EXPORT autotune_table_base {
protected:
	template <typename T> SM_INLINE void add_fn(T fn) { _fn.push_back((void*)fn); }
	void   insert(index_t q0, index_t q1, uint8_t index);
	void*  lookup(index_t q0, index_t q1);
	
	struct entry: public autotune_query {
		SM_INLINE entry(const autotune_query& query, int8_t index): autotune_query(query), index(index) { }
		int8_t         index;
	};

	vector<entry>    _db;
	vector<void*>    _fn;
};

template <typename exec_fn>
class autotune_table4: public autotune_table_base {
public:
	SM_INLINE exec_fn operator()(index_t q0, index_t q1)  { return (exec_fn)lookup(q0,q1); }
};


//////////////////////////////////////////////////////////////////////////////////

SM_NAMESPACE_END

#endif // __SM_AUTOTUNE_H__
