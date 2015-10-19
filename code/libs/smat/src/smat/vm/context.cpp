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
#include <smat/vm/context.h>
#include <smat/vm/machine.h>
#include <smat/vm/heap.h>
#include <smat/vm/extension.h>
#include <smat/vm/instruction_db.h>
#include <base/assert.h>
#include <base/util.h>
#include <base/range.h>
#include <base/random.h>
#include <base/optionset.h>
#include <base/logging.h>
#include <base/sized_array.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <stdexcept>

SM_NAMESPACE_BEGIN

using namespace std;

argument make_alloc_arg(void* addr, size_t size)
{
	SM_ASSERT(size == (isize_t)size); // make sure nothing got lost
	argument arg;
	arg.shape.x = (isize_t)size;
	arg.dtype = u8;
	arg.vtype = vt_darray;
	arg.set(addr);
	return arg;
}

////////////////////////////////////////////////////////////////////

const char* const c_default_backend = "cuda";
static SM_THREADLOCAL context* s_thread_ctx = 0;
static SM_THREADLOCAL char s_thread_ctx_backend_name[32] = { 0 };
static bool s_warn_debug = true;

void set_backend(const char* backend_name) { set_backend(backend_name,optionset()); }
void set_backend(const char* backend_name, const optionset& opt)
{
#ifdef _DEBUG
	if (s_warn_debug) {
		s_warn_debug = false;
		printf("PerformanceWarning: using a debug build of smat.\n");
	}
#endif
	destroy_backend();
	dllhandle_t dll = load_extension(format("smat_%s",backend_name).c_str());
	auto create_backend_context = (context* (*)())get_dll_proc(dll,"create_backend_context");
	SM_ASSERTMSG(create_backend_context,"OSError: Module did not contain extern \"C\" function 'create_backend_context'.\n");
	s_thread_ctx = create_backend_context();
	strncpy(s_thread_ctx_backend_name,backend_name,32);
	s_thread_ctx->set_options(opt);
}

void reset_backend() { reset_backend(optionset()); }

void reset_backend(const optionset& opt)
{
	if (strlen(s_thread_ctx_backend_name) == 0)
		strncpy(s_thread_ctx_backend_name,c_default_backend,32);
	set_backend(s_thread_ctx_backend_name,opt);
}

static bool g_force_destroy_backend = false;

void destroy_backend(bool force)
{
	if (s_thread_ctx) {
		g_force_destroy_backend = force;
		delete s_thread_ctx;
		s_thread_ctx = 0;
		g_force_destroy_backend = false;
	}
}

void autotune_backend()
{
	thread_ctx().autotune();
}

context& thread_ctx()
{
	if (!s_thread_ctx) {
		set_backend(c_default_backend);
	}
	return *s_thread_ctx;
}

////////////////////////////////////////////////////////////////////

context::context(const _SM::backend_info& info, _SM::machine* machine, _SM::heap* heap)
: _backend_info(info)
, _machine(machine)
, _heap(heap)
, _queue(new instruction_list())
, _queuesize(0)
, _max_queuesize(0)
, _verbose(false)
, _sanitycheck(false)
, _randseed(0)
{
}

context::context(const context&) { SM_UNREACHABLE(); }
context& context::operator=(const context&) { SM_UNREACHABLE(); }

context::~context()
{
	destroy_heap();
	delete _queue;   _queue = 0;
	delete _machine; _machine = 0;
}

void context::destroy_heap()
{
	if (_heap) {
		auto count = _heap->alloc_count();
		if (count > 0 && g_smat_last_error.empty() && !g_force_destroy_backend)
			SM_ERROR(format("Cannot destroy smat context with %d allocations still in its heap.",count).c_str());
		delete _heap;
		_heap = 0;
	}
}

const backend_info& context::backend_info() const { ensure_initialized(); return _backend_info; }
      machine&      context::machine()          { ensure_initialized(); return *_machine; }
const machine&      context::machine() const    { ensure_initialized(); return *_machine; }
      heap&         context::heap()        { ensure_initialized(); return *_heap; }
const heap&         context::heap() const  { ensure_initialized(); return *_heap; }

bool context::is_supported(dtype_t dt) const
{
	ensure_initialized(); 
	switch (dt) {
	case b8:  return SM_WANT_BOOL;
	case i8:  return SM_WANT_INT;
	case u8:  return SM_WANT_UINT;
	case i16: return SM_WANT_INT;
	case u16: return SM_WANT_UINT;
	case i32: return SM_WANT_INT;
	case u32: return SM_WANT_UINT;
	case i64: return SM_WANT_INT;
	case u64: return SM_WANT_UINT;
	case f32: return true;
	case f64: return SM_WANT_DOUBLE;
	}
	return true; 
}
void context::set_verbose(int verbose) { _verbose = verbose; }
void context::set_sanitycheck(int level)
{
	if (level != _sanitycheck) {
		if (_heap->size() > 0)
			SM_ERROR("ValueError: Cannot set sanitycheck if there are allocations on the smat heap.");
		_sanitycheck = level;
	}
}
void context::set_max_queuesize(size_t size) { _max_queuesize = size; }
void context::set_randseed(size_t seed)
{
	_randseed = (int)seed;
	_SM::set_rand_seed(seed);
	srand((unsigned)seed*3);
}
void context::set_options(const optionset& opt)
{
	if (opt.contains<int>("seed"))         set_randseed(opt.get<int>("seed"));
	if (opt.contains<int>("verbose"))      set_verbose(opt.get<int>("verbose"));
	if (opt.contains<int>("queue"))        set_max_queuesize(opt.get<int>("queue"));
	if (opt.contains<int>("sanitycheck"))  set_sanitycheck(opt.get<int>("sanitycheck"));

	// Set which context/memory events get logged
	auto log_ids = opt.get_strings("log");
	for (auto id : log_ids)
		set_log_policy(id.c_str(),_verbose ? lp_print : lp_ignore);

	_machine->set_options(opt);
}

void context::sync()
{
	ensure_initialized();
	flush();
}

heap_alloc context::alloc(shape_t s, dtype_t dtype)
{
	return alloc((size_t)s.size()*dtype_size(dtype));
}

heap_alloc context::alloc(size_t size)
{
	size_t align = _heap->alignment();
	size_t pitch = _heap->pitch();

	size = rndup(size, pitch);

	if (_sanitycheck) {
		// Add padding before and after the allocation so that we can 
		// check for device heap corruption later on, during execution
		heap_alloc alloc_padded = _heap->alloc(size+2*pitch);

		// Emit an instruction to mark the moment in the execution stream 
		// where the allocation is to take place. This will allow the machine
		// to fill the padding (if any) with magic numbers.
		emit(oc_alloc,make_alloc_arg(alloc_padded.addr, alloc_padded.size));

		// The address returned should not include the magic number padding,
		// which is of size 'pitch' bytes at the start of the allocation.
		heap_alloc alloc((size_t)alloc_padded.addr+pitch, alloc_padded.size-2*pitch, alloc_padded.bookkeeping);
		return alloc;
	} else {
		return _heap->alloc(size);
	}
}

void context::free(heap_alloc& alloc)
{
	size_t align = _heap->alignment();
	size_t pitch = _heap->pitch();

	if (_sanitycheck) {
		// Calculate the actual allocation that included padding.
		heap_alloc alloc_padded((size_t)alloc.addr-pitch, alloc.size+2*pitch, alloc.bookkeeping);
		_heap->free(alloc_padded);

		// Emitting the "free" instruction, but make sure it 
		// sees the original unpadded address range.
		emit(oc_free,make_alloc_arg(alloc_padded.addr,alloc_padded.size));
	} else {
		_heap->free(alloc);
	}
}

void context::autotune() { }

void context::ensure_initialized() const { } // subclass should override if necessary

void context::emit(opcode_t opcode, argument arg0)                                              { emit(instruction(opcode,move(arg0))); }
void context::emit(opcode_t opcode, argument arg0, argument arg1)                               { emit(instruction(opcode,move(arg0),move(arg1))); }
void context::emit(opcode_t opcode, argument arg0, argument arg1, argument arg2)                { emit(instruction(opcode,move(arg0),move(arg1),move(arg2))); }
void context::emit(opcode_t opcode, argument arg0, argument arg1, argument arg2, argument arg3) { emit(instruction(opcode,move(arg0),move(arg1),move(arg2),move(arg3))); }
void context::emit(opcode_t opcode, argument arg0, argument arg1, argument arg2, argument arg3, argument arg4) { emit(instruction(opcode,move(arg0),move(arg1),move(arg2),move(arg3),move(arg4))); }

void context::emit(instruction&& instr)
{
	if (std::uncaught_exception())  // if an exception was thrown, we may be unwinding the stack, 
		return;                     // so ignore any attempts to execute further instructions during
	                                // unwind (e.g. smat::~smat destructor)
	_machine->validate(instr,get_instruction_info(instr.opcode));
	_queue->push_back(move(instr));
	_queuesize++;

	if (_queuesize > _max_queuesize)
		flush();
}

void context::flush()
{
	SM_DBASSERT(_queuesize == _queue->size());

	// execute each instruction, starting at current program 
	// counter and continuing until the end of the program
	while (!_queue->empty()) {
		instruction instr(std::move(_queue->front())); 
		_queue->pop_front(); _queuesize--;
		_machine->execute(move(instr),get_instruction_info(instr.opcode),get_instruction_impl(_backend_info.uuid,instr.opcode));
	}
}

///////////////////////////////////////////////////////////////////////////

#pragma warning(push)
#pragma warning(disable : 4190 4297)  // disable warning about C linkage of shape_t, and about throwing exceptions from C functions

extern "C" {

SM_EXPORT void  api_set_backend(const char* backend_name, int argc, const char** argv) { SM_API_TRY; set_backend(backend_name,optionset(argc,argv)); SM_API_CATCH; }
SM_EXPORT void  api_set_backend_options(int argc, const char** argv)  { SM_API_TRY; thread_ctx().set_options(optionset(argc,argv)); SM_API_CATCH; }
SM_EXPORT void  api_reset_backend(int argc, const char** argv)        { SM_API_TRY; reset_backend(optionset(argc,argv)); SM_API_CATCH; }
SM_EXPORT void  api_destroy_backend(bool force)                       { SM_API_TRY; destroy_backend(force); SM_API_CATCH; }
SM_EXPORT void  api_autotune_backend()                                { SM_API_TRY; autotune_backend(); SM_API_CATCH; }
SM_EXPORT void  api_get_backend_info(backend_info& info)              { SM_API_TRY; info   = thread_ctx().backend_info();  SM_API_CATCH; }
SM_EXPORT void  api_get_heap_status(heap_status& status)              { SM_API_TRY; status = thread_ctx().heap().status(); SM_API_CATCH; }
SM_EXPORT bool  api_is_dtype_supported(dtype_t dt)                    { SM_API_TRY; return thread_ctx().is_supported(dt);  SM_API_CATCH_AND_RETURN(false); }
SM_EXPORT void  api_sync()                                            { SM_API_TRY; thread_ctx().sync(); SM_API_CATCH }
SM_EXPORT void  api_set_rand_seed(size_t seed)                        { SM_API_TRY; thread_ctx().set_randseed(seed); SM_API_CATCH }

}

SM_NAMESPACE_END
