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
#include <smat_cuda/cuda_context.h>
#include <smat_cuda/cuda_errors.h>
#include <smat_cuda/launch_util.h>
#include <smat/vm/heap.h>
#include <smat/vm/machine.h>
#include <smat/vm/extension.h>
#include <base/optionset.h>
#include <cuda.h>
#include <cstring>

SM_NAMESPACE_BEGIN

backend_info s_backend_info = {
	cuda_uuid,
	"cuda",
	"0.1alpha",
	""
};

class cuda_block_allocator: public block_allocator {
public:
	virtual void* alloc_block(size_t size)
	{
		void* daddr = 0;
		cudaError_t error = cudaMalloc(&daddr,size);
		if (error != cudaSuccess)
			SM_ERROR(format("AssertionError: cudaMalloc failed allocating %llu bytes; CUDA error is \"%s\".",(unsigned long long)size,cudaGetErrorString(error)).c_str());
		return daddr;
	}
	
	virtual void  free_block(void* ptr)
	{
		ccu(Free,ptr);
	}
	
	virtual size_t get_total_memory()
	{
		return thread_cudactx().deviceprop().totalGlobalMem;
	}

	virtual size_t get_avail_memory()
	{
		// cudaGetMemInfo tells us what we need.
		size_t free = 0, total = 0;
		ccu(MemGetInfo, &free, &total);
		return free;
	}
};

// TODO: add code to ensure that each device is used by at most one thread's cuda_context.

//////////////////////////////////////////////////////////////
	
cuda_context::cuda_context()
: context(s_backend_info,
          new _SM::machine(),
          new _SM::heap(0,8,new cuda_block_allocator()))
, _device(0)
, _want_stream(false)
, _curr_device(-1)
, _cublas(0)
, _stream(0)
, _curand(0)
, _curand_state(0)
{
	memset(&_deviceprop,0,sizeof(cudaDeviceProp));
	//printf("CREATING CONTEXT %d/%d with heap size %llu\n",_curr_device,_device,_heap->size());
}

cuda_context::~cuda_context()
{
	//printf("DESTROYING CONTEXT %d/%d with heap size %llu\n",_curr_device,_device,_heap->size());
	cce();
	sync();
	cce();
	destroy_heap();
	if (_cublas) {
		ccb(SetStream,_cublas,0);
		ccb(Destroy,_cublas);
	}
	if (_stream) ccu(StreamDestroy,_stream);
	if (_curand) ccr(DestroyGenerator,_curand);
	if (_curand_state) ccu(Free,_curand_state);
	ccu(DeviceReset);
	cce();
}

int                   cuda_context::device() const { ensure_initialized(); return _device; }
cudaStream_t          cuda_context::stream() const { ensure_initialized(); return _stream; }
cublasHandle_t        cuda_context::cublas() const { ensure_initialized(); return _cublas; }
curandGenerator_t     cuda_context::curand() const { ensure_initialized(); return _curand; }
curandState*          cuda_context::curand_state() const { ensure_initialized(); return _curand_state; }
const cudaDeviceProp& cuda_context::deviceprop() const { ensure_initialized(); return _deviceprop; }

void execute_curand_init(cudaStream_t, curandState*, int seed, unsigned gdim, unsigned bdim);

void cuda_context::ensure_initialized() const
{
	if (_curr_device >= 0)
		return;

	//printf("INITIALIZING CONTEXT %d/%d with heap size %llu\n",_curr_device,_device,_heap->size());

	// Initialize CUDA runtime, with our own non-default stream.
	ccu(SetDevice,_device);
	ccu(GetDeviceProperties,&_deviceprop,_device);
	cce();
	sprintf(_backend_info.device,"%s (arch=%d.%d)",_deviceprop.name,_deviceprop.major,_deviceprop.minor); // update device name in backend_info
	ccu(GetDevice,&_curr_device);
	ccu(SetDeviceFlags, cudaDeviceScheduleBlockingSync);
	SM_ASSERTMSG(_device == _curr_device,"AssertionError: SetDevice seems to have failed.");

	// Set the heap allocation pitch, in bytes. 
	// Currently fixed assuming CUDA 2.x or 3.x, but could depend on _deviceprop
	_heap->set_pitch(64);

	// Create our own stream if necessary.
	if (_want_stream) ccu(StreamCreateWithFlags,&_stream,cudaStreamNonBlocking)
	else              _stream = 0;

	// Initialize CUBLAS, and bind it to our stream.
	ccb(Create,&_cublas);
	ccb(SetStream,_cublas,_stream);
	cce();

	// Initialize CURAND, and bind it to our stream.
	set_curand_seed();

	// Set device limits AFTER seeding CURAND. 
	// The reason is because CURAND sets things like StackSize to large values,
	// and doesn't bother resetting them back to normal.
	ccu(DeviceSetLimit, cudaLimitStackSize, 512);            // Stack space per thread. CUDA's default is 1024 bytes.
	ccu(DeviceSetLimit, cudaLimitPrintfFifoSize,   1024*1024); // Shared memory for printing. CUDA's default is 1MB.
	ccu(DeviceSetLimit, cudaLimitMallocHeapSize, 2*1024*1024); // Global memory for heap structure. CUDA's default is 9MB.
}

void cuda_context::set_device(int device)
{
	SM_ASSERTMSG(_curr_device < 0 || device == _device,"RuntimeError: Cannot change device on an smat context that is already in use; must create a new context instead.");
	_device = device;
}

void cuda_context::set_randseed(size_t seed)
{
	context::set_randseed(seed);
	if (_curand)
		set_curand_seed();
}

void cuda_context::set_options(const optionset& opt)
{
	context::set_options(opt);
	if (opt.contains<int>("device")) set_device(opt.get<int>("device"));
}

bool cuda_context::is_supported(dtype_t dt) const
{
	ensure_initialized();
	if (dt == f64)
		return context::is_supported(dt) && (_deviceprop.major > 1 || (_deviceprop.major == 1 && _deviceprop.minor >= 3)); // Double only supported by compute capability >= 1.3
	return context::is_supported(dt);
}

void cuda_context::sync()
{
	context::sync();
	ccu(StreamSynchronize,_stream);
}

void cuda_context::autotune()
{
	dllhandle_t handle = load_extension("smat_cuda_autotune");
	unload_extension(handle);
}

void cuda_context::set_curand_seed() const
{
	if (_curand)
		ccr(DestroyGenerator,_curand);
	ccr(CreateGenerator,&_curand, CURAND_RNG_PSEUDO_DEFAULT);
	ccr(SetStream, _curand, _stream);
	ccr(SetPseudoRandomGeneratorSeed, _curand, _randseed);
	
	// Force CURAND to set up random number generation completely.
	// This is so that the setup overhead is finished with, and also
	// so that CURAND allocates whatever memory it needs.
	void* buffer = 0;
	ccu(Malloc,&buffer,512*sizeof(double));
	ccr(GenerateUniform,_curand,(float*)buffer,512);
	ccr(GenerateNormal,_curand,(float*)buffer,512,0.0f,1.0f);
	if (is_supported(f64)) {
		ccr(GenerateUniformDouble,_curand,(double*)buffer,512);
		ccr(GenerateNormalDouble,_curand,(double*)buffer,512,0.0,1.0);
	}
	ccu(Free,buffer);
	cce();

	// Initialize CURAND device state for random number generation within kernels
	// We need an array of curandState values, one for every possible thread
	// that might execute on the GPU.
//	unsigned max_resident_threads = _deviceprop.maxThreadsPerMultiProcessor*_deviceprop.multiProcessorCount;
	launchcfg cfg = make_elemwise_launchcfg(1<<30);  // get the biggest possible launch configuration, with the most blocks/threads
	unsigned max_resident_threads = cfg.gdim.x*cfg.bdim.x;
	if (!_curand_state)
		ccu(Malloc,&_curand_state, max_resident_threads*sizeof(curandState));
	execute_curand_init(_stream, _curand_state, _randseed, max_resident_threads/_deviceprop.maxThreadsPerBlock, _deviceprop.maxThreadsPerBlock);

}

// Called by smat.dll to create a context instance from this backend (cuda_context, in our case).
extern "C" SM_DLLEXPORT context* create_backend_context()
{
	return new cuda_context();
}

SM_NAMESPACE_END
