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
#include <smat/vm/util/autotune.h>
#include <base/assert.h>
#include <base/util.h>
#include <base/os.h>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <fstream>

SM_NAMESPACE_BEGIN

using namespace std;

autotuner::autotuner(opcode_t opcode)
: _opcode(opcode)
{
}

autotuner::~autotuner()
{
}

void autotuner::print_all()
{
	for (score_map::const_iterator it = _scores.begin(); it != _scores.end(); ++it) {
		const autotune_query& q = it->first;
		printf("query(%d,%d,%d,%d):\n",q.q0,q.q1,q.q2,q.q3);
		const autotune_scores& r = it->second;
		for (size_t i = 0; i < r.size(); ++i) {
			const autotune_pset& pset = _psets[i];
			printf("     %6.2f @ (%d,%d,%d,%d)\n",r[i],pset.p0,pset.p1,pset.p2,pset.p3);
		}
		printf("\n");
	}
}

void autotuner::print_best()
{
	vector<vector<pair<autotune_query,double>>> series(_psets.size());
	for (score_map::const_iterator it = _scores.begin(); it != _scores.end(); ++it) {
		const autotune_scores& r = it->second;
		int best = (int)(max_element(r.begin(),r.end()) - r.begin());
		series[best].push_back(make_pair(it->first,r[best]));
	}
	for (size_t i = 0; i < series.size(); ++i) {
		auto& pset = _psets[i];
		printf("\n<%d,%d,%d,%d>\n",pset.p0,pset.p1,pset.p2,pset.p3);
		for (auto& q : series[i]) {
			printf("%d\t%d\t%.6f\n",q.first.q0,q.first.q1,q.second);
		}
	}
}

void autotune_table_base::insert(index_t q0, index_t q1, uint8_t index)
{
	_db.emplace_back(autotune_query(q0,q1),index);
}

void* autotune_table_base::lookup(index_t q0, index_t q1)
{
	size_t min_index = 0;
	long long min_dist = 1ll<<40;
	for (size_t i = 0; i < _db.size(); ++i) {
		const entry& e = _db[i];
		long long dist = abs(e.q0-q0) + abs(e.q1-q1);
		if (dist < min_dist) {
			min_index = e.index;
			min_dist = dist;
		}
	}
	SM_ASSERTMSG(min_index < _fn.size(),"IndexError: Autotune lookup returned function index that was not added to the database; use add_fn.");
	return _fn[min_index];
}

/*
void load_autotuner()
{
	static bool s_was_read = false;
	if (s_was_read)
		return;
	SM_ASSERT(g_autotuner.empty());
	string path = format("%s/%s/%s",user_home_dir(),g_autotuner_dir,g_autotuner_file);
	ifstream file(path,ios::in|ios::binary);
	if (!file.is_open())
		return;
	while (!file.eof()) {
		char buf[sizeof(autotuner::value_type)];
		file.read(buf,sizeof(autotuner::value_type));
		g_autotuner.insert(*(autotuner::value_type*)buf);
	}
}

bool add_autotune_entry(autotune_query query, autotune_entry entry)
{
	load_autotuner();
	auto iter = g_autotuner.find(query);
	if (iter == g_autotuner.end()) {
		g_autotuner.insert(make_pair(query,entry)); // make a new entry
		return true;
	} else if (entry.performance > iter->second.performance) {
		iter->second = entry;  // replace the old entry with a new best-performing one
		return true;
	}
	return false; // don't use the new entry
}

const autotune_entry* get_autotune_entry(autotune_query query)
{
	load_autotuner();
	auto iter = g_autotuner.find(query);
	if (iter == g_autotuner.end())
		return 0;
	return &iter->second;
}

void save_autotuner()
{
	load_autotuner();
	string path = format("%s/%s",user_home_dir(),g_autotuner_dir);
	mkdir(path.c_str());
	path += "/";
	path += g_autotuner_file;
	ofstream file(path,ios::out|ios::binary);
	if (!file.is_open())
		SM_ERROR(format("OSError: Failed to open \"%s\" for writing.",path.c_str()).c_str());
	for (auto item : g_autotuner) {
		file.write((const char*)&item,sizeof(autotuner::value_type));
	}
}
*/

SM_NAMESPACE_END
