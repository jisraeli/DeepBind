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
#include <base/optionset.h>
#include <base/util.h>
#include <algorithm>
#include <cstring>

SM_NAMESPACE_BEGIN

using namespace std;

optionset::optionset()
{
}

optionset::optionset(const vector<string>& argv)
{
	add(argv);
}

optionset::optionset(int argc, const char** argv)
{
	add(vector<string>(argv,argv+argc));
}

optionset::optionset(const optionset& src)
: _boolset(src._boolset)
, _intset(src._intset)
, _doubleset(src._doubleset)
, _stringset(src._stringset)
, _stringlistset(src._stringlistset)
, _optionsetset(src._optionsetset)
, _optionsetlistset(src._optionsetlistset)
{
}

optionset::optionset(optionset&& src)
: _boolset(move(src._boolset))
, _intset(move(src._intset))
, _doubleset(move(src._doubleset))
, _stringset(move(src._stringset))
, _stringlistset(move(src._stringlistset))
, _optionsetset(move(src._optionsetset))
, _optionsetlistset(move(src._optionsetlistset))
{
}

optionset::~optionset()
{
}

void optionset::add(const vector<string>& argv)
{
	for (auto& arg : argv) {
		size_t i = arg.find("=");
		if (i == -1)
			i = arg.size();
		string name = arg.substr(0,i);
		string val  = arg.substr(i+1);
		set_all(name.c_str(),val);
	}
}

void optionset::set(const char* name, bool   value) { _boolset[name]   = value; }
void optionset::set(const char* name, int    value) { _intset[name]    = value; }
void optionset::set(const char* name, double value) { _doubleset[name] = value; }
void optionset::set(const char* name, const char* value)         { _stringset[name] = value; }
void optionset::set(const char* name, const std::string& value)  { _stringset[name] = value; }
void optionset::set(const char* name, const optionset& value)    { _optionsetset[name] = value; }
void optionset::add(const char* name, const string& item)        { _stringlistset[name].push_back(item); }
void optionset::add(const char* name, const optionset& item)     { _optionsetlistset[name].push_back(item); }

void optionset::set_all(const char* name, const string& val)
{
	if (val.empty()) {
		set(name,true);
		return;
	}
	if (val == "true" || val == "True" || val == "yes" || val == "1") {
		set(name,true);
		set(name,1);
		set(name,1.0);
	} else if (val == "false" || val == "False" || val == "no" || val == "0") {
		set(name,false);
		set(name,0);
		set(name,0.0);
	} else {
		double dval = atof(val.c_str());
		set(name,(int)dval);
		set(name,dval);
	}
	set(name,val.c_str());
	stringlist items = split(val,",");
	for (auto& item : items)
		add(name,item);
}

template <typename V> void merge(V& dst, const V& src) { dst = src; }
template <> void merge<optionset>(optionset& dst, const optionset& src) { dst.merge(src); }

template <typename V>
void merge(map<string,V>& dst, const map<string,V>& src, const char* prefix)
{
	int prelen = prefix ? (int)strlen(prefix) : 0;
	for (typename map<string,V>::const_iterator i = src.begin(); i != src.end(); ++i) {
		const string& name = i->first;
		if ((int)name.size() >= prelen && 0 == strncmp(name.c_str(),prefix,prelen))
			merge(dst[name],i->second);
	}
}

void optionset::merge(const optionset& src, const char* prefix)
{
	_SM::merge(_boolset  ,src._boolset  ,prefix);
	_SM::merge(_intset   ,src._intset   ,prefix);
	_SM::merge(_doubleset,src._doubleset,prefix);
	_SM::merge(_stringset,src._stringset,prefix);
	_SM::merge(_stringlistset,src._stringlistset,prefix);
	_SM::merge(_optionsetset,src._optionsetset,prefix);
	_SM::merge(_optionsetlistset,src._optionsetlistset,prefix);
}

optionset& optionset::operator=(const optionset& src)
{
	_boolset = src._boolset;
	_intset = src._intset;
	_doubleset = src._doubleset;
	_stringset = src._stringset;
	_stringlistset = src._stringlistset;
	_optionsetset = src._optionsetset;
	_optionsetlistset = src._optionsetlistset;
	return *this;
}

optionset& optionset::operator=(optionset&& src)
{
	_boolset = move(src._boolset);
	_intset = move(src._intset);
	_doubleset = move(src._doubleset);
	_stringset = move(src._stringset);
	_stringlistset = move(src._stringlistset);
	_optionsetset = move(src._optionsetset);
	_optionsetlistset = move(src._optionsetlistset);
	return *this;
}

SM_NAMESPACE_END

