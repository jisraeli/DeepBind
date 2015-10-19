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
#include <smat_cuda/launch_util.h>
#include <smat_cuda/cuda_context.h>
#include <smat/vm/instruction_db.h>
#include <smat/smat.h>
#include "kangaroo_smat.h"
#include <cstring>
using namespace sm;

void launch_corr1ord(cudaStream_t stream, dtype_t dtype,
                     const void*    W, usize_t m, usize_t nfilter,
                     const uint8_t* X, usize_t n, usize_t nchannel,
                           void*    Z);

void launch_corr1ord_bprop_W(cudaStream_t stream, dtype_t dtype,
                                   void*    dW, usize_t m, usize_t nfilter,
                             const uint8_t*  X, usize_t n, usize_t nchannel,
                             const void*    dZ);

void launch_corr1ord_bprop_X(cudaStream_t stream, dtype_t dtype,
                                   void*    dX, usize_t n, usize_t nchannel,
                             const void*    W, usize_t m, usize_t nfilter,
                             const void*    dZ);

void launch_convseq(cudaStream_t stream, dtype_t dtype,
                    const uint8_t*  samples, usize_t nsample, usize_t nchannel,
                    const void*     filters, usize_t nfilter, usize_t filter_size,
                    const uindex_t* segments,usize_t nsegment,
                          void*     featuremaps);


void launch_convseq_bprop(cudaStream_t stream, dtype_t dtype,
                          const uint8_t*  samples, usize_t nsample, usize_t nchannel,
                                void*     filters, usize_t nfilter, usize_t filter_size,
                          const uindex_t* segments,usize_t nsegment,
                          const void*     deltamaps);

void launch_poolrgn(cudaStream_t stream, dtype_t dtype,
					const void*     featuremaps, usize_t nfeaturemap,
					const uindex_t* regions,     usize_t nregion, bool per_region_step,
					      void*     pooledmap,   uindex_t* pooledmap_argmax, pooltype_t ptype);

void launch_poolrgn_bprop(cudaStream_t stream, dtype_t dtype,
                                void*     unpooledmaps, usize_t nfeaturemap,
                          const uindex_t* regions,      usize_t nregion, bool per_region_step,
                          const void*     pooledmaps,   const uindex_t* pooledmaps_argmax, pooltype_t ptype);

void launch_dropoutord_fp_tr(cudaStream_t stream,
                             const uint8_t* X, uint8_t* Z, bool* M, usize_t n, float rate);

struct corr1ord_options_t {
	padmode_t padmode;
	usize_t   nchannel;
};

struct convseq_options_t {
	usize_t nchannel;
};

struct poolrgn_options_t {
	pooltype_t ptype;
};


void execute_corr1ord(opcode_t opcode, const argument& W,
                                       const argument& X,
                                       const argument& Z,
                                       const argument& options)
{
	auto opt = options.get<corr1ord_options_t*>();
	auto m = W.shape.y/opt->nchannel;
	auto n = X.size()-m+1;  // here n is the number of ACTUAL OUTPUTS that will get written, not number of input elements
	auto nfilter = W.shape.x;
	launch_corr1ord(thread_cudactx().stream(),Z.dtype,
	                W.get<const void*>(),m,nfilter,
	                X.get<const uint8_t*>(),n,opt->nchannel,
	                Z.get<void*>());
}

void validate_corr1ord(opcode_t opcode,const argument& W,
                                       const argument& X,
                                       const argument& Z,
                                       const argument& options)
{
	auto nchannel = options.get<corr1ord_options_t*>()->nchannel;
	SM_ASSERT(W.shape.y % nchannel == 0);
	SM_ASSERT(W.dtype == Z.dtype);
	SM_ASSERT(X.dtype == u8);
	SM_ASSERT(Z.shape.x == W.shape.x);
	SM_ASSERT(Z.shape.y == X.size());
}

void execute_corr1ord_bp_W(opcode_t opcode, const argument& dW,
                                            const argument& X,
                                            const argument& dZ,
                                            const argument& options)
{
	auto opt = options.get<corr1ord_options_t*>();
	auto m = dW.shape.y/opt->nchannel;
	auto n = X.size()-m+1;  // here n is the number of ACTUAL OUTPUTS that will get written, not number of input elements
	auto nfilter = dW.shape.x;
	launch_corr1ord_bprop_W(thread_cudactx().stream(),dZ.dtype,
	                        dW.get<void*>(),m,nfilter,
	                         X.get<const uint8_t*>(),n,opt->nchannel,
	                        dZ.get<const void*>());
}

void validate_corr1ord_bp_X(opcode_t opcode,const argument& dX,
                                            const argument& W,
                                            const argument& dZ,
                                            const argument& options)
{
	auto nchannel = options.get<corr1ord_options_t*>()->nchannel;
	SM_ASSERT(W.shape.y % nchannel == 0);
	SM_ASSERT(W.dtype == dZ.dtype);
	SM_ASSERT(dX.dtype == dZ.dtype);
	SM_ASSERT(dX.shape.x == nchannel);
	SM_ASSERT(dZ.shape.x == W.shape.x);
	SM_ASSERT(dZ.shape.y == dX.shape.y);
}

void execute_corr1ord_bp_X(opcode_t opcode, const argument& dX,
                                            const argument& W,
                                            const argument& dZ,
                                            const argument& options)
{
	auto opt = options.get<corr1ord_options_t*>();
	auto m = W.shape.y/opt->nchannel;
	auto n = dX.shape.y;
	auto nfilter = W.shape.x;
	launch_corr1ord_bprop_X(thread_cudactx().stream(),dZ.dtype,
	                        dX.get<void*>(),n,opt->nchannel,
	                         W.get<const void*>(),m,nfilter,
	                        dZ.get<const void*>());
}

void execute_convseq(opcode_t opcode, const argument& samples,
                                      const argument& filters,
                                      const argument& segments,
                                      const argument& featuremaps,
                                      const argument& options)
{
	auto opt = options.get<convseq_options_t*>();
	auto nsample = samples.shape.x;
	auto nfilter = filters.shape.y;
	auto filter_size = filters.shape.x/opt->nchannel;
	launch_convseq(thread_cudactx().stream(),featuremaps.dtype,
	               samples.get<const uint8_t*>(),nsample,opt->nchannel,
	               filters.get<const void*>(),nfilter,filter_size,
	               segments.get<const uindex_t*>(),segments.shape.x,
	               featuremaps.get<void*>());
}

void validate_convseq(opcode_t opcode,const argument& samples,
                                      const argument& filters,
                                      const argument& segments,
                                      const argument& featuremaps,
                                      const argument& options)
{
	auto nsample  = samples.shape.x;
	auto nchannel = options.get<convseq_options_t*>()->nchannel;
	SM_ASSERT(filters.shape.x % nchannel == 0);
	SM_ASSERT(samples.dtype == u8);
	SM_ASSERT(filters.dtype == featuremaps.dtype);
	SM_ASSERT(featuremaps.shape.x == filters.shape.y);
	SM_ASSERT(featuremaps.shape.y == nsample);
}

void execute_convseq_bprop(opcode_t opcode, const argument& samples,
                                            const argument& filters,
                                            const argument& segments,
                                            const argument& deltamaps,
                                            const argument& options)
{
	auto opt = options.get<convseq_options_t*>();
	usize_t nsample = samples.shape.x;
	usize_t nfilter = filters.shape.y;
	auto filter_size = filters.shape.x/opt->nchannel;
	launch_convseq_bprop(thread_cudactx().stream(),deltamaps.dtype,
	                     samples.get<const uint8_t*>(),nsample,opt->nchannel,
	                     filters.get<void*>(),nfilter,filter_size,
	                     segments.get<const uindex_t*>(),segments.shape.x,
	                     deltamaps.get<const void*>());
}

void execute_poolrgn(opcode_t opcode, const argument& unpooledmaps,
                                      const argument& regions,
                                      const argument& pooledmaps,
                                      const argument& pooledmaps_argmax,
                                      const argument& options)
{
	auto opt = options.get<poolrgn_options_t*>();
	launch_poolrgn(thread_cudactx().stream(),unpooledmaps.dtype,
	               unpooledmaps.get<const void*>(),unpooledmaps.shape.x,
	               regions.get<const uindex_t*>(),regions.shape.y,regions.shape.x == 3,
	               pooledmaps.get<void*>(),pooledmaps_argmax.get<uindex_t*>(),opt->ptype);
}

void validate_poolrgn(opcode_t opcode, const argument& unpooledmaps,
                                       const argument& regions,
                                       const argument& pooledmaps,
                                       const argument& pooledmaps_argmax,
                                       const argument& options)
{
	auto nfeaturemap = unpooledmaps.shape.x;
	auto nregion = regions.shape.y;
	auto opt = options.get<poolrgn_options_t*>();
	SM_ASSERTMSG(unpooledmaps.dtype == pooledmaps.dtype,"TypeError: featuremaps dtype must match pooledmaps dtype.");
	SM_ASSERTMSG(regions.shape.x == 2 || regions.shape.x == 3,"ValueError: regions array must have 2 or 3 columns.");
	if (opt->ptype == pt_all) {
		SM_ASSERTMSG(pooledmaps.shape.x == 2*nfeaturemap,"ValueError: pooledmaps for \"all\" pooling must have twice as many columns as there are feature maps.");
	} else {
		SM_ASSERTMSG(pooledmaps.shape.x == nfeaturemap,"ValueError: pooledmaps must have as many columns as there are feature maps.");
	}
	SM_ASSERTMSG(pooledmaps.shape.y == nregion,"ValueError: pooledmaps must have as many rows as there are regions.");
	if (pooledmaps_argmax.vtype != vt_none) {
		SM_ASSERTMSG(opt->ptype == pt_max || opt->ptype == pt_all,"ValueError: pooledmaps_argmax can only be used for max/all pooling.");
		SM_ASSERTMSG(pooledmaps_argmax.shape.x == nfeaturemap,"ValueError: pooledmaps_argmax must have as many columns as there are feature maps.");
		SM_ASSERTMSG(pooledmaps_argmax.dtype == ctype2dtype(uindex_t),"TypeError: pooledmaps_argmax dtype must be same type as uindex_t.");
		SM_ASSERTMSG(pooledmaps_argmax.shape.y == nregion,"ValueError: pooledmaps_argmax must have as many rows as there are regions.");
	}	
}

void execute_poolrgn_bprop(opcode_t opcode, const argument& unpooledmaps,
                                            const argument& regions,
                                            const argument& pooledmaps,
                                            const argument& pooledmaps_argmax,
                                            const argument& options)
{
	auto opt = options.get<poolrgn_options_t*>();
	launch_poolrgn_bprop(thread_cudactx().stream(),unpooledmaps.dtype,
	                     unpooledmaps.get<void*>(),unpooledmaps.shape.x,
	                     regions.get<const uindex_t*>(),regions.shape.y,regions.shape.x == 3,
	                     pooledmaps.get<const void*>(),pooledmaps_argmax.get<const uindex_t*>(),opt->ptype);
}

void validate_poolrgn_bprop(opcode_t opcode, const argument& unpooledmaps,
                                             const argument& regions,
                                             const argument& pooledmaps,
                                             const argument& pooledmaps_argmax,
                                             const argument& options)
{
	auto nfeaturemap = unpooledmaps.shape.x;
	auto nregion = regions.shape.y;
	auto opt = options.get<poolrgn_options_t*>();
	SM_ASSERTMSG(unpooledmaps.dtype == pooledmaps.dtype,"TypeError: unpooledmaps dtype must match pooledmaps dtype.");
	SM_ASSERTMSG(regions.shape.x == 2 || regions.shape.x == 3,"ValueError: regions array must have 2 or 3 columns.");
	if (opt->ptype == pt_all) {
		SM_ASSERTMSG(pooledmaps.shape.x == 2*nfeaturemap,"ValueError: pooledmaps for \"all\" pooling must have twice as many columns as there are feature maps.");
	} else {
		SM_ASSERTMSG(pooledmaps.shape.x == nfeaturemap,"ValueError: pooledmaps must have as many columns as there are feature maps.");
	}
	SM_ASSERTMSG(pooledmaps.shape.y == nregion,"ValueError: pooledmaps must have as many rows as there are regions.");
	if (opt->ptype == pt_max || opt->ptype == pt_all) {
		SM_ASSERTMSG(pooledmaps_argmax.vtype != vt_none,"ValueError: pooledmaps_argmax must be specified for max pooling.")
		SM_ASSERTMSG(pooledmaps_argmax.shape.x == nfeaturemap,"ValueError: pooledmaps_argmax must have as many columns as there are feature maps.");
		SM_ASSERTMSG(pooledmaps_argmax.dtype == ctype2dtype(uindex_t),"TypeError: pooledmaps_argmax dtype must be same type as uindex_t.");
		SM_ASSERTMSG(pooledmaps_argmax.shape.y == nregion,"ValueError: pooledmaps_argmax must have as many rows as there are regions.");
	} else
		SM_ASSERTMSG(pooledmaps_argmax.vtype == vt_none,"ValueError: pooledmaps_argmax can only be specified for max pooling.")
}

void validate_dropoutord_fp_tr(opcode_t opcode, const argument& X, const argument& Z, const argument& M, const argument& rate)
{
	SM_ASSERT(X.shape.x == Z.shape.x);
	SM_ASSERT(X.shape.y == Z.shape.y);
	SM_ASSERT(M.shape.x == X.shape.x);
	SM_ASSERT(M.shape.y == X.shape.y);
	SM_ASSERT(X.dtype == u8);
	SM_ASSERT(Z.dtype == u8);
}

void execute_dropoutord_fp_tr(opcode_t opcode, const argument& X, const argument& Z, const argument& M, const argument& rate)
{
	auto n = X.shape.size();
	launch_dropoutord_fp_tr(thread_cudactx().stream(),
	                        X.get<const uint8_t*>(), Z.get<uint8_t*>(), M.get<bool*>(), n, rate.get<float>());
}



opcode_t oc_corr1ord   = -1;
opcode_t oc_corr1ord_bp_W= -1;
opcode_t oc_corr1ord_bp_X= -1;
opcode_t oc_convseq    = -1;
opcode_t oc_convseq_bp = -1;
opcode_t oc_poolrgn    = -1;
opcode_t oc_poolrgn_bp = -1;
opcode_t oc_dropoutord_fp_tr = -1;

void corr1ord_options_deleter(void* ptr) { delete (corr1ord_options_t*)ptr; };
void convseq_options_deleter(void* ptr)  { delete (convseq_options_t*)ptr; };
void poolrgn_options_deleter(void* ptr)  { delete (poolrgn_options_t*)ptr; };


#pragma warning(disable : 4190 4297)  // disable warning about C linkage of shape_t, and about throwing exceptions from C functions

extern "C" {

SM_DLLEXPORT void api_corr1ord(const smat* W, const smat* X, smat* Z, const corr1ord_options_t* options)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_corr1ord,W->as_arg(),X->as_arg(),Z->as_arg(),user_arg(new corr1ord_options_t(*options),corr1ord_options_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_corr1ord_bprop_W(smat* dW, const smat* X, const smat* dZ, const corr1ord_options_t* options)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_corr1ord_bp_W,dW->as_arg(),X->as_arg(),dZ->as_arg(),user_arg(new corr1ord_options_t(*options),corr1ord_options_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_corr1ord_bprop_X(smat* dX, const smat* W, const smat* dZ, const corr1ord_options_t* options)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_corr1ord_bp_X,dX->as_arg(),W->as_arg(),dZ->as_arg(),user_arg(new corr1ord_options_t(*options),corr1ord_options_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_convseq(const smat* samples,
                              const smat* filters,
                              const smat* segments,
                                    smat* featuremaps, const convseq_options_t* options)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_convseq,
	                      samples->as_arg(),
	                      filters->as_arg(),
	                      segments ? segments->as_arg() : unused_arg(),
	                      featuremaps->as_arg(),
	                      user_arg(new convseq_options_t(*options),convseq_options_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_convseq_bprop(const smat* samples,
                                          smat* filters,
                                    const smat* segments,
                                    const smat* deltamaps, const convseq_options_t* options)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_convseq_bp,
	                      samples->as_arg(),
	                      filters->as_arg(),
	                      segments ? segments->as_arg() : unused_arg(),
	                      deltamaps->as_arg(),
	                      user_arg(new convseq_options_t(*options),convseq_options_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_poolrgn(const smat* unpooledmaps, const smat* regions, smat* pooledmaps, smat* pooledmaps_argmax, const poolrgn_options_t* options)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_poolrgn,
	                      unpooledmaps->as_arg(),
	                      regions->as_arg(),
	                      pooledmaps->as_arg(),
	                      pooledmaps_argmax ? pooledmaps_argmax->as_arg() : unused_arg(),
	                      user_arg(new poolrgn_options_t(*options),poolrgn_options_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_poolrgn_bprop(smat* unpooledmaps, const smat* regions, const smat* pooledmaps, const smat* pooledmaps_argmax, const poolrgn_options_t* options)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_poolrgn_bp,
	                      unpooledmaps->as_arg(),
	                      regions->as_arg(),
	                      pooledmaps->as_arg(),
	                      pooledmaps_argmax ? pooledmaps_argmax->as_arg() : unused_arg(),
	                      user_arg(new poolrgn_options_t(*options),poolrgn_options_deleter));
	SM_API_CATCH
}


SM_DLLEXPORT void api_dropoutord_fp_tr(const smat* X, smat* Z, smat* M, float rate)
{ 
	SM_API_TRY
	thread_cudactx().emit(oc_dropoutord_fp_tr,
	                      X->as_arg(),
	                      Z->as_arg(),
	                      M->as_arg(),
	                      carray(rate));
	SM_API_CATCH
}

SM_DLLEXPORT void register_ext()
{
	oc_corr1ord = add_instruction("corr1ord",
			iprop_none,
			aprop_in |aprop_float,      // (W)
			aprop_in |aprop_uint,       // (X)
			aprop_out|aprop_float,      // (Z)
			aprop_in |aprop_user        // (options)
		);

	oc_corr1ord_bp_W = add_instruction("corr1ord_bp_W",
			iprop_none,
			aprop_out|aprop_float,      // (W)
			aprop_in |aprop_uint,       // (X)
			aprop_in |aprop_float,      // (Z)
			aprop_in |aprop_user        // (options)
		);

	oc_corr1ord_bp_X = add_instruction("corr1ord_bp_X",
			iprop_none,
			aprop_out|aprop_float,      // (dX)
			aprop_in |aprop_float,      // (W)
			aprop_in |aprop_float,      // (dZ)
			aprop_in |aprop_user        // (options)
		);

	oc_convseq = add_instruction("convseq",
			iprop_none,
			aprop_in |aprop_uint,       // (samples)
			aprop_in |aprop_float,      // (filters)
			aprop_in |aprop_uint,       // (segments)
			aprop_out|aprop_float,      // (featuremaps)
			aprop_in |aprop_user        // (options)
		);

	oc_convseq_bp = add_instruction("convseq_bp",
			iprop_none,
			aprop_in |aprop_uint,       // (samples)
			aprop_out|aprop_float,      // (filters)
			aprop_in |aprop_uint,       // (segments)
			aprop_in |aprop_float,      // (deltamaps)
			aprop_in |aprop_user        // (options)
		);

	oc_poolrgn = add_instruction("poolrgn",
			iprop_none,
			aprop_in |aprop_float,      // (unpooledmaps)
			aprop_in |aprop_uint,        // (regions)
			aprop_out|aprop_float,      // (pooledmaps)
			aprop_out|aprop_uint,       // (pooledmaps_argmax)
			aprop_in |aprop_user        // (options)
		);

	oc_poolrgn_bp = add_instruction("poolrgn_bp",
			iprop_none,
			aprop_out|aprop_float,      // (unpooledmaps)
			aprop_in |aprop_uint,        // (regions)
			aprop_in |aprop_float,      // (pooledmaps)
			aprop_in |aprop_uint,       // (pooledmaps_argmax)
			aprop_in |aprop_user        // (options)
		);

	oc_dropoutord_fp_tr = add_instruction("dropoutord_fp_tr",
			iprop_none,
			aprop_in  |aprop_uint,      // (X)
			aprop_out |aprop_uint,      // (Z)
			aprop_out |aprop_bool,      // (M)
			aprop_in  |aprop_float      // (rate)
		);

	add_instruction_impl(cuda_uuid,oc_corr1ord  ,execute_corr1ord     ,validate_corr1ord);
	add_instruction_impl(cuda_uuid,oc_corr1ord_bp_W,execute_corr1ord_bp_W ,validate_corr1ord);
	add_instruction_impl(cuda_uuid,oc_corr1ord_bp_X,execute_corr1ord_bp_X ,validate_corr1ord_bp_X);
	add_instruction_impl(cuda_uuid,oc_convseq   ,execute_convseq      ,validate_convseq);
	add_instruction_impl(cuda_uuid,oc_convseq_bp,execute_convseq_bprop,validate_convseq);
	add_instruction_impl(cuda_uuid,oc_poolrgn   ,execute_poolrgn      ,validate_poolrgn);
	add_instruction_impl(cuda_uuid,oc_poolrgn_bp,execute_poolrgn_bprop,validate_poolrgn_bprop);
	add_instruction_impl(cuda_uuid,oc_dropoutord_fp_tr,execute_dropoutord_fp_tr,validate_dropoutord_fp_tr);
}

} // extern "C"
