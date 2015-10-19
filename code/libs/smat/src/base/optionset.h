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
#ifndef __SM_OPTIONSET_H__
#define __SM_OPTIONSET_H__

#include <base/assert.h>
#include <base/util.h>
#include <vector>
#include <map>
#include <string>

SM_NAMESPACE_BEGIN

using std::vector;
using std::map;
using std::string;

class optionset;
typedef vector<optionset> optionsetlist;

class BASE_EXPORT optionset {
public:
	typedef vector<string> stringlist;
	typedef map<string,bool> boolset;
	typedef map<string,int> intset;
	typedef map<string,double> doubleset;
	typedef map<string,string>   stringset;
	typedef map<string,stringlist>    stringlistset;
	typedef map<string,optionset>     optionsetset;
	typedef map<string,optionsetlist> optionsetlistset;

	optionset();
	optionset(const vector<string>& argv);
	optionset(int argc, const char** argv);
	~optionset();
	void add(const vector<string>& argv);
	void set(const char* name, bool   value);
	void set(const char* name, int    value);
	void set(const char* name, double value);
	void set(const char* name, const char* value);
	void set(const char* name, const string& value);
	void set(const char* name, const optionset& value);
	void add(const char* name, const string& item);  // add item to stringlist 'name'
	void add(const char* name, const optionset& item);      // add item to optlist 'name'

	// valid T types are <bool>, <int>, <double>, <string>, <stringlist>, and <optionset>
	template <typename T> bool     contains(const char* name) const;
	template <typename T> const T& get(const char* name) const;
	template <typename T> const T& get(const char* name, const T& default_val) const;
	const stringlist get_strings(const char* name) const { return contains<stringlist>(name) ? get<stringlist>(name) : stringlist(); }

	// optionset in 'src' are copied and overwrite the current optionset;
	// if 'src' has any suboptions, they are recursively merged (not replaced)
	void merge(const optionset& src, const char* prefix = 0);

	optionset(const optionset& src);
	optionset(optionset&& src);
	optionset& operator=(const optionset& src);
	optionset& operator=(optionset&& src);

private:
	template <typename T> const map<string,T>& getset() const { /* should not compile */ }

	void set_all(const char* name, const string& val);

	boolset          _boolset;
	intset           _intset;
	doubleset        _doubleset;
	stringset        _stringset;
	stringlistset    _stringlistset;
	optionsetset     _optionsetset;
	optionsetlistset _optionsetlistset;
};

template <> SM_INLINE const map<string,bool>&       optionset::getset<bool>() const { return _boolset; }
template <> SM_INLINE const map<string,int>&        optionset::getset<int>() const  { return _intset; }
template <> SM_INLINE const map<string,double>&     optionset::getset<double>() const { return _doubleset; }
template <> SM_INLINE const map<string,string>&     optionset::getset<string>() const { return _stringset; }
template <> SM_INLINE const map<string,optionset::stringlist>&    optionset::getset<optionset::stringlist>() const { return _stringlistset; }
template <> SM_INLINE const map<string,optionset>&                optionset::getset<optionset>() const { return _optionsetset; }
template <> SM_INLINE const map<string,optionsetlist>&            optionset::getset<optionsetlist>() const { return _optionsetlistset; }

//////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool optionset::contains(const char* name) const
{
	typedef map<string,T> settype;
	const settype& set = getset<T>();
	typename settype::const_iterator i = set.find(name);
	return i != set.end();
}

template <typename T>
const T& optionset::get(const char* name) const
{
	typedef map<string,T> settype;
	const settype& set = getset<T>();
	typename settype::const_iterator i = set.find(name);
	//SM_ASSERTMSG(i != set.end(),format("option '%s' not found",name).c_str());
	SM_ASSERT(i != set.end());
	return i->second;
}

template <typename T>
const T& optionset::get(const char* name, const T& default_value) const
{
	typedef map<string,T> settype;
	const settype& set = getset<T>();
	typename settype::const_iterator i = set.find(name);
	return (i != set.end()) ? i->second : default_value;
}

SM_NAMESPACE_END

#define OPTION(type,name) type name = opt.get<type>(#name);
#define OPTION_DEFAULT(type,name,defval) type name = opt.get<type>(#name,defval);

#endif // __SM_OPTIONSET_H__
