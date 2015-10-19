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
#include <smat/vm/extension.h>
#include <smat/vm/instruction_db.h>
#include <base/assert.h>
#include <base/os.h>
#include <base/util.h>
#include <base/logging.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <stdexcept>

#if defined _WIN32 || defined __CYGWIN__
#define DLLEXT    ".dll"
#define DLLPREFIX ""
#elif defined __linux__ or defined __unix__
#define DLLEXT    ".so"
#define DLLPREFIX "lib"
#include <elf.h>
#include <link.h>
#else
#error not yet implemented for this platform
#endif

SM_NAMESPACE_BEGIN

using namespace std;

////////////////////////////////////////////////////////////////////

map<string,dllhandle_t> g_extensions;
dllhandle_t             g_extension_loading = 0;
dllhandle_t             g_extension_unloading = 0;


dllhandle_t load_extension(const char* name)
{
	auto entry = g_extensions.find(name);
	if (entry != g_extensions.end())
		return entry->second;

	get_last_os_error();
	
	vector<string> search_dirs;
	vector<string> failure_reasons;
#ifdef _DEBUG
	string builddir = "build/debug/bin/";
#else
	string builddir = "build/release/bin/";
#endif

/*
#if defined __linux__ || defined __unix__
	const ElfW(Dyn) *dyn = _DYNAMIC;
	const ElfW(Dyn) *rpath = NULL;
	const char *strtab = NULL;
	for (; dyn->d_tag != DT_NULL; ++dyn) {
		if (dyn->d_tag == DT_RPATH) {
			rpath = dyn;
		} else if (dyn->d_tag == DT_STRTAB) {
			strtab = (const char *)dyn->d_un.d_val;
		}
	}
	if (strtab != NULL && rpath != NULL) {
		builddir = strtab + rpath->d_un.d_val;
		builddir += "/";
		//printf("RPATH: %s\n", strtab + rpath->d_un.d_val);
	}
#endif
*/
	search_dirs.push_back(builddir);
	search_dirs.push_back("../smat/" + builddir);
	search_dirs.push_back("libs/smat/" + builddir);
	search_dirs.push_back("libs/deepity/" + builddir);
	search_dirs.push_back("libs/kangaroo/" + builddir);
	/*const char* dev_dir = getenv("DEV");
	if (dev_dir)
		search_dirs.push_back(string(dev_dir) + "/smat/" + builddir);*/
	const char* path_var = getenv("PATH");
	if (path_var) {
#ifdef _WIN32
		const char* delims = ";";
#else
		const char* delims = ":";
#endif
		std::vector<std::string> path_dirs = split(path_var, delims);
		for (size_t i = 0; i < path_dirs.size(); ++i)
			search_dirs.push_back(path_dirs[i] + "/");
	}
	if (strstr(name,"_smat")) {
		string prefix(name);
		prefix = prefix.substr(0,strstr(name,"_smat")-name);
		search_dirs.push_back("../" + prefix + "/" + builddir);
		//if (dev_dir)
		//	search_dirs.push_back(string(dev_dir) + "/" + prefix + "/" + builddir);
	}

	dllhandle_t handle = 0;
	for (auto dir : search_dirs) {
		// the imat backend is located inside smat.dll ... there is no separete smat_imat.dll
		string dllname = name;
		dllname = dir + DLLPREFIX + dllname + DLLEXT;
		SM_LOG("ext","Trying to load extension %s.",dllname.c_str());
		handle = load_dll(dllname.c_str());
		if (handle)
			break;
		const char* err_msg = get_last_os_error();
		if (!err_msg)
			err_msg = "None";
		failure_reasons.push_back(err_msg);
	}
	if (!handle) {
		printf("Failed to load extension %s:\n",name);
		for (size_t i = 0; i < search_dirs.size(); ++i) {
			printf("   attempt: %s\n",search_dirs[i].c_str());
			printf("    reason: %s\n",failure_reasons[i].c_str());
		}
		throw runtime_error(format("IOError: Failed to load module '%s'.\n",name).c_str());
	}
	g_extensions[name] = handle;

	// If the extension has a register_ext function, call it right away.
	auto register_ext = (void (*)())get_dll_proc(handle,"register_ext");
	if (register_ext) {
		g_extension_loading = handle;
		try {
			register_ext();
		} catch (...) {
			g_extension_loading = 0;
			throw;
		}
		g_extension_loading = 0;
	}

	return handle;
}

void unload_extension(dllhandle_t handle)
{
	auto entry = find_if(g_extensions.begin(),g_extensions.end(),
		[&](const pair<string,dllhandle_t>& x) { return x.second == handle; });
	if (entry == g_extensions.end())
		SM_ERROR("AssertionError: Cannot unload extension (handle not recognized).");
	g_extensions.erase(entry);
	auto unregister_ext = (void (*)())get_dll_proc(handle,"unregister_ext");
	if (unregister_ext) {
		g_extension_unloading = handle;
		try {
			unregister_ext();
		} catch (...) {
			g_extension_unloading = 0;
			throw;
		}
		g_extension_unloading = 0;
	}
	unload_dll(handle);
	remove_instructions(handle);
	remove_instruction_impls(handle);
}

///////////////////////////////////////////////////////////////////////////

#pragma warning(push)
#pragma warning(disable : 4190 4297)  // disable warning about C linkage of shape_t, and about throwing exceptions from C functions

extern "C" {

SM_EXPORT dllhandle_t api_load_extension(const char* dllname)  { SM_API_TRY; return load_extension(dllname); SM_API_CATCH_AND_RETURN(0); }
SM_EXPORT void        api_unload_extension(dllhandle_t handle) { SM_API_TRY; unload_extension(handle);       SM_API_CATCH; }

}

SM_NAMESPACE_END
