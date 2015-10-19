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
#include <base/os.h>
#include <base/assert.h>
#include <base/util.h>
#include <string>
#include <cstring>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#elif defined __linux__ 
#include <unistd.h>
#include <sys/sysinfo.h>
#include <dlfcn.h>
#include <sys/stat.h>
#else
#error not yet implemented for this platform
#endif

SM_NAMESPACE_BEGIN

using namespace std;

BASE_EXPORT size_t get_system_memory_avail()
{
#ifdef _WIN32
	MEMORYSTATUSEX memstat = { sizeof(MEMORYSTATUSEX) };
	GlobalMemoryStatusEx(&memstat);
	return (size_t)memstat.ullAvailPhys;
#else
	struct sysinfo info;
	sysinfo(&info);
	return info.totalram;
#endif
}

BASE_EXPORT size_t get_system_memory_total()
{
#ifdef _WIN32
	MEMORYSTATUSEX memstat = { sizeof(MEMORYSTATUSEX) };
	GlobalMemoryStatusEx(&memstat);
	return (size_t)memstat.ullTotalPhys;
#else
	return (size_t)sysconf(_SC_PHYS_PAGES)*(size_t)sysconf(_SC_PAGE_SIZE);
#endif
}

BASE_EXPORT size_t get_process_memory_used()
{
#ifdef _WIN32
	PROCESS_MEMORY_COUNTERS counters;
	GetProcessMemoryInfo(GetCurrentProcess(),&counters,sizeof(counters));
	return (size_t)counters.WorkingSetSize;
#else
	struct sysinfo info;
	sysinfo(&info);
	return info.totalram - info.freeram;
#endif
}

BASE_EXPORT const char* get_last_os_error()
{
	static char buffer[1024];
#ifdef _WIN32
	auto err = GetLastError();
	if (err == 0)
		return NULL;
	char* temp_buffer = 0;
	FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
	              FORMAT_MESSAGE_FROM_SYSTEM |
	              FORMAT_MESSAGE_IGNORE_INSERTS,
	              NULL,
	              err,
	              MAKELANGID(LANG_NEUTRAL,SUBLANG_DEFAULT),
	              (LPTSTR)&buffer,1023,NULL);
#else
	const char* err = dlerror();
	if (err == NULL)
		return NULL;
	strncpy(buffer,err,1024); buffer[1023]=0;
#endif
	return buffer;
}

BASE_EXPORT dllhandle_t load_dll(const char* name)
{
#ifdef _WIN32
	HMODULE h = LoadLibrary(wstring(name,name+strlen(name)).c_str());
	return (dllhandle_t)h;
#else
	void* handle = dlopen(name,RTLD_NOW);
	return (dllhandle_t)handle;
#endif
}

BASE_EXPORT void unload_dll(dllhandle_t handle)
{
#ifdef _WIN32
	FreeLibrary((HMODULE)handle);
#else
	dlclose((void*)handle);
#endif
}

BASE_EXPORT void* get_dll_proc(dllhandle_t dll, const char* procname)
{
#ifdef _WIN32
	return (void*)GetProcAddress((HMODULE)dll,procname);
#else
	return (void*)dlsym((void*)dll,procname);
#endif
}

BASE_EXPORT const char* user_home_dir()
{
	static char s_path[512] = {0};
	if (!s_path[0]) {
#ifdef _WIN32
		strcat(s_path,getenv("HOMEDRIVE"));
		strcat(s_path,getenv("HOMEPATH"));
#else
		strcat(s_path,getenv("HOME"));
#endif
	}
	return s_path;
}

BASE_EXPORT void mkdir(const char* dir)
{
	auto parts = split(dir,"/\\");
	string path;
	for (auto part : parts) {
		path += part;
		if (!isdir(path.c_str())) {
#ifdef _WIN32
			BOOL success = CreateDirectoryA(path.c_str(),NULL);
#else
			bool success = ::mkdir(path.c_str(),S_IRUSR|S_IXUSR|S_IWUSR) == 0;
#endif
			if (!success)
				SM_ERROR(format("OSError: Failed to create directory \"%s\".",path.c_str()).c_str());
		}
	}
}

BASE_EXPORT bool isdir(const char* dir)
{
#ifdef _WIN32
	DWORD fattrib = GetFileAttributesA(dir);
	return (fattrib != INVALID_FILE_ATTRIBUTES) && (fattrib & FILE_ATTRIBUTE_DIRECTORY);
#else
	struct stat sb;
	return stat(dir, &sb) == 0 && S_ISDIR(sb.st_mode);
#endif
}


SM_NAMESPACE_END
