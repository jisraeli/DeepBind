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
#include <smat_cuda/elemwise2.cuh>
#include <smat_cuda/cuda_errors.h>
#include <smat/vm/instruction_db.h>
#include <smat/vm/util/specialization_table.h>
#include <smat/vm/util/specialization_typelists.h>

SM_NAMESPACE_BEGIN

DEF_AB_GENERIC(copy, b[j] = (B)a[i]) // copy function just casts element of type A to destination type B

#define KERNEL_COPY_DD_PARAMS   const S* src, D* dst
#define KERNEL_COPY_DD_ARGS     src,i,dst,i
#define KERNEL_COPY_DD_PREAMBLE 

#define KERNEL_COPY_RD_PARAMS   const S* src, D* dst, usize_t m
#define KERNEL_COPY_RD_ARGS     src,i%m,dst,i
#define KERNEL_COPY_RD_PREAMBLE 

#define KERNEL_COPY_CD_PARAMS   const S* src, D* dst, usize_t m
#define KERNEL_COPY_CD_ARGS     src,i/m,dst,i
#define KERNEL_COPY_CD_PREAMBLE 

#define KERNEL_COPY_SD_PARAMS   const S _src, D* dst
#define KERNEL_COPY_SD_ARGS     src,0,dst,i
#define KERNEL_COPY_SD_PREAMBLE const S src[1] = { _src };   // turn scalar into an "array" for the sake of generic copy code; should be optimized away

// TODO: specialize this for smaller-than-32-bit copy, 
// since right now it's significantly slower, and working with
// bytes will be important (converting to/from).
//    Note that if source bytes can be bound to a texture,
//    they can be auto-converted to range [0,1].

#define DEF_KERNEL_COPY(broadcast) \
	template <typename S, typename D>                                                        \
	__global__ void kernel_copy_##broadcast(KERNEL_COPY_##broadcast##_PARAMS, usize_t size)  \
	{                                                                                        \
		DECL_KERNEL_VARS                                                                     \
		KERNEL_COPY_##broadcast##_PREAMBLE                                                   \
		for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)                         \
			k_copy<S,D>::apply(KERNEL_COPY_##broadcast##_ARGS);                              \
	}


///////////////////////////////////////////////////////////////////
//                COPY VARIANTS:
//                   dd = device matrix   -> device matrix
//                   rd = device rowvec   -> device matrix
//                   cd = device colvec   -> device matrix
//                   sd = host scalar     -> device matrix
//                   id = identity matrix -> device matrix
///////////////////////////////////////////////////////////////////

DEF_KERNEL_COPY(DD)  // no broadcasting
DEF_KERNEL_COPY(RD)  // broadcast row vec arg
DEF_KERNEL_COPY(CD)  // broadcast col vec arg
DEF_KERNEL_COPY(SD)  // broadcast scalar arg

#define LAUNCH_KERNEL_COPY(broadcast) \
	kernel_copy_##broadcast<<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>

template <typename S, typename D>
struct execute_copy_typed_xd {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		if (src.vtype != vt_darray || dst.vtype != vt_darray)
			SM_ERROR(format("NotImplementedError: Unsupported combination of argument value types in copy: %s:%s -> %s:%s.\n", \
								vtype2str(src.vtype),dtype2str(src.dtype), \
								vtype2str(dst.vtype),dtype2str(dst.dtype)).c_str()); 
		usize_t size = (usize_t)dst.size();
		if (size > 0) {
			launchcfg cfg = make_elemwise_launchcfg(size);
			// Launch either an elementwise or a row/col broadcasted version of the functor
			if (src.shape == dst.shape) {
				LAUNCH_KERNEL_COPY(DD)(src.get<const S*>(),dst.get<D*>(),size);                   // direct elementwise copy // TODO: make this faster for small matching types
			} else if (src.shape.y == 1 && src.shape.x == dst.shape.x) {
				LAUNCH_KERNEL_COPY(RD)(src.get<const S*>(),dst.get<D*>(),dst.shape.x,size);            // broadcast row vector on left
			} else if (src.shape.x == 1 && src.shape.y == dst.shape.y) {
				LAUNCH_KERNEL_COPY(CD)(src.get<const S*>(),dst.get<D*>(),dst.shape.x,size);            // broadcast col vector on left
			} else 
				SM_ERROR("NotImplementedError: incompatible broadcasting dimensions at kernel launch.\n"); 
		}
	}
};

// Launch kernel to write scalar value to dst; we can assume the types are the same
// because the context base class tries to coerce the dtypes of constant arguments.
template <typename D>
struct execute_copy_typed_sd {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		usize_t size = (usize_t)dst.size();
		if (size > 0) {
			launchcfg cfg = make_elemwise_launchcfg(size);
			LAUNCH_KERNEL_COPY(SD)(src.get<D>(),dst.get<D*>(),size);
		}
	}
};

template <typename D>
__global__ void kernel_copy_id(D* dst, usize_t m_plus_one, usize_t size)
{
	DECL_KERNEL_VARS
	for (usize_t i = (usize_t)bdx*bx+tx; i < size; i += bdx*gdx)
		dst[i] = (i % m_plus_one) == 0 ? (D)1 : (D)0;
}

// Launch kernel to write identity matrix to dst
template <typename D>
struct execute_copy_typed_id {
	static void execute(opcode_t opcode, const argument& src, const argument& dst)
	{
		usize_t size = (usize_t)dst.size();
		if (size > 0) {
			launchcfg cfg = make_elemwise_launchcfg(size);
			kernel_copy_id<D><<<cfg.gdim,cfg.bdim,cfg.smem,cfg.stream>>>(dst.get<D*>(),dst.shape.x+1,size);
		}
	}
};

template <typename T>
void copy_ch_typed(T val, T* dst, usize_t size)
{
	for (usize_t i = 0; i < size; ++i)
		dst[i] = val;
}

void copy_ch(const argument& src, const argument& dst)
{
	switch (dst.dtype) {case b8:  copy_ch_typed(src.get<bool>()    ,dst.get<bool*>()    ,dst.size());
	case i8:  copy_ch_typed(src.get<int8_t>()  ,dst.get<int8_t*>()  ,dst.size());
	case u8:  copy_ch_typed(src.get<uint8_t>() ,dst.get<uint8_t*>() ,dst.size());
	case i16: copy_ch_typed(src.get<int16_t>() ,dst.get<int16_t*>() ,dst.size());
	case u16: copy_ch_typed(src.get<uint16_t>(),dst.get<uint16_t*>(),dst.size());
	case i32: copy_ch_typed(src.get<int32_t>() ,dst.get<int32_t*>() ,dst.size());
	case u32: copy_ch_typed(src.get<uint32_t>(),dst.get<uint32_t*>(),dst.size());
	case i64: copy_ch_typed(src.get<int64_t>() ,dst.get<int64_t*>() ,dst.size());
	case u64: copy_ch_typed(src.get<uint64_t>(),dst.get<uint64_t*>(),dst.size());
	case f32: copy_ch_typed(src.get<float>()   ,dst.get<float*>()   ,dst.size());
	case f64: copy_ch_typed(src.get<double>()  ,dst.get<double*>()  ,dst.size());
	}
}

void execute_copy(opcode_t opcode, const argument& src, const argument& dst)
{
	SM_ASSERT(dst.vtype == vt_harray || dst.vtype == vt_darray);
	if (dst.shape.size() == 0)
		return;
	isize_t dtsize = dtype_size(src.dtype);
	SM_ASSERTMSG(dst.vtype == vt_darray || dst.vtype == vt_harray,"AssertionError: Output must be host address or device address.");
	bool all_full_stride = (dst.strides == src.strides) && (src.strides.y == src.shape.x*src.strides.x);
	if (src.vtype == vt_harray && dst.vtype == vt_darray) {
		// HOST -> DEVICE
		SM_ASSERT(src.dtype == dst.dtype);    // dtypes must match if transferring to/from host.
		SM_ASSERT(src.shape == dst.shape);    // sizes must match (no broadcasting when copy to device)
		if (all_full_stride) { ccu(MemcpyAsync  ,dst.get<void*>(),src.get<const void*>(),src.size()*dtsize,cudaMemcpyHostToDevice,thread_cudactx().stream()); } // try to do 1D copies since they can be overlapped if we eventually use streams
		else                 { ccu(Memcpy2DAsync,dst.get<void*>(),dst.strides.y*dtsize,src.get<const void*>(),src.strides.y*dtsize,src.shape.x*src.strides.x*dtsize,src.shape.y,cudaMemcpyHostToDevice,thread_cudactx().stream()); }
	} else if (src.vtype == vt_darray && dst.vtype == vt_harray) {
		// HOST <- DEVICE
		SM_ASSERT(src.dtype == dst.dtype);    // dtypes must match if transferring to/from host.
		SM_ASSERT(src.shape == dst.shape);    // sizes must match (no broadcasting when copy to device)
		if (all_full_stride) { ccu(MemcpyAsync  ,dst.get<void*>(),src.get<const void*>(),src.size()*dtsize,cudaMemcpyDeviceToHost,thread_cudactx().stream()); } // try to do 1D copies since they can be overlapped if we eventually use streams
		else                 { ccu(Memcpy2DAsync,dst.get<void*>(),dst.strides.y*dtsize,src.get<const void*>(),src.strides.y*dtsize,src.shape.x*src.strides.x*dtsize,src.shape.y,cudaMemcpyDeviceToHost,thread_cudactx().stream()); }
	} else if (src.vtype == vt_darray && dst.vtype == vt_darray) {
		// DEVICE -> DEVICE
		if (src.shape != dst.shape || src.dtype != dst.dtype) {
			SM_ASSERTMSG(src.strides.y == src.shape.x*src.strides.x,"NotImplementedError: Cannot perform broadcasting/conversion on column-sliced input array.")
			SM_ASSERTMSG(dst.strides.y == dst.shape.x*dst.strides.x,"NotImplementedError: Cannot perform broadcasting/conversion on column-sliced output array.")
			DECL_SPECIALIZATION_TABLE(T_GxG,execute_fn2,execute_copy_typed_xd); // TODO: Several of these kernels will have identical machine code, so don't generate redundant kernels; TODO: when data is of small but matching type (e.g. int8->int8), copy with larger size when possible
			specialization_table(src.dtype,dst.dtype)(opcode,src,dst); // copy src to dst, casting type as necessary
		} else {
			if (all_full_stride) { ccu(MemcpyAsync  ,dst.get<void*>(),src.get<const void*>(),src.size()*dtsize,cudaMemcpyDeviceToDevice,thread_cudactx().stream()); } // try to do 1D copies since they can be overlapped if we eventually use streams
			else                 { ccu(Memcpy2DAsync,dst.get<void*>(),dst.strides.y*dtsize,src.get<const void*>(),src.strides.y*dtsize,src.shape.x*src.strides.x*dtsize,src.shape.y,cudaMemcpyDeviceToDevice,thread_cudactx().stream()); }
		}
	} else {
		SM_ASSERTMSG(dst.strides.x == 1, "NotImplementedError: Column slicing not yet supported for this operation."); // TODO
		SM_ASSERTMSG(dst.strides.y == dst.strides.x*dst.shape.x || dst.shape.y == 1,"NotImplementedError: Column slicing not yet supported for this operation."); // TODO
		if (src.vtype == vt_carray && dst.vtype == vt_darray) {
			DECL_SPECIALIZATION_TABLE(T_G,execute_fn2,execute_copy_typed_sd)
			specialization_table(dst.dtype)(opcode,src,dst);  // broadcast scalar, for any matching type
		} else if (src.vtype == vt_carray && dst.vtype == vt_harray) {
			copy_ch(src,dst);                                 // broadcast scalar to host array
		} else if (src.vtype == vt_iarray && dst.vtype == vt_darray) {
			DECL_SPECIALIZATION_TABLE(T_G,execute_fn2,execute_copy_typed_id);
			specialization_table(dst.dtype)(opcode,src,dst);  // broadcast scalar, for any matching type
		} else {
			SM_ERROR(format("Unsupported combination of argument value types in exec_copy: %s:%s -> %s:%s.\n",vtype2str(src.vtype),dtype2str(src.dtype),vtype2str(dst.vtype),dtype2str(dst.dtype)).c_str());
		}
	}
}

SM_NAMESPACE_END
