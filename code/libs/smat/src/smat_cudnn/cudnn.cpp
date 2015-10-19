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
#include <smat/vm/instruction_db.h>
#include <smat/vm/context.h>
#include <smat/smat.h>
#include <cstring>
#include <cudnn.h>

using namespace sm;

#define ccd(func, ...) { cudnnStatus_t status = cudnn##func(__VA_ARGS__); if (status != CUDNN_STATUS_SUCCESS) { SM_ERROR(format("AssertionError: CUDNN error in %s: %s.", #func, cudnnGetErrorString(status)).c_str()); } }

#define SETUP_CONV2_DESCRIPTORS(src, filters, dst) \
	conv2cfg_t& cfg = *_cfg.get<conv2cfg_t*>();    \
	int filter_w = cfg.filter_w;                   \
	int filter_h = cfg.filter_h;                   \
	int filter_c = (int)filters.shape.x / (filter_w*filter_h); \
	int filter_k = (int)filters.shape.y;           \
	int src_w = cfg.src_w;                         \
	int src_h = cfg.src_h;                         \
	int src_n = (int)src.shape.y;                  \
	int src_c = (int)src.shape.x / (src_w*src_h);  \
	int dst_w = (src_w-filter_w)/cfg.stride + 1;   \
	int dst_h = (src_h-filter_h)/cfg.stride + 1;   \
	int dst_n = (int)dst.shape.y;                  \
	int dst_k = (int)dst.shape.x / (dst_w*dst_h);  \
	int n = src_n;                                 \
	int k = dst_k;                                 \
	int c = src_c;                                 \
	SM_ASSERTMSG(dst.shape.x == dst_k*dst_w*dst_h, "Number of columns in dst must be dst_w*dst_h*dst_channels"); \
	SM_ASSERTMSG((int)filters.shape.x % (filter_w*filter_h) == 0, "Number of columns in filters must be divisible by filter_w*filter_h"); \
	SM_ASSERTMSG((int)src.shape.x % (src_w*src_h) == 0, "Number of columns in src must be divisible by src_w*src_h"); \
	SM_ASSERTMSG((int)dst.shape.x % (dst_w*dst_h) == 0, "Number of columns in dst must be divisible by dst_w*dst_h"); \
	SM_ASSERTMSG(src_n == dst_n, "Number of src rows must match number of dst rows"); \
	SM_ASSERTMSG(filter_c == src_c, "Number of filter channels must match number of src channels"); \
	SM_ASSERTMSG(filter_k == dst_k, "Number of filters must match number of dst channels"); \
	cudnnDataType_t cudnn_dt = (src.dtype == f32) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE; \
	ccd(SetTensor4dDescriptor, dst_desc, CUDNN_TENSOR_NCHW, cudnn_dt, n, k, dst_h, dst_w); \
	ccd(SetTensor4dDescriptor, src_desc, CUDNN_TENSOR_NCHW, cudnn_dt, n, c, src_h, src_w); \
	ccd(SetFilter4dDescriptor, filter_desc, cudnn_dt, k, c, filter_h, filter_w); \
	ccd(SetConvolution2dDescriptor, conv_desc, 0, 0, cfg.stride, cfg.stride, 1, 1, CUDNN_CROSS_CORRELATION);

#define SETUP_POOL2_DESCRIPTORS(src, dst) \
	pool2cfg_t& cfg = *_cfg.get<pool2cfg_t*>();    \
	int window_w = cfg.window_w;                   \
	int window_h = cfg.window_h;                   \
	int src_w = cfg.src_w;                         \
	int src_h = cfg.src_h;                         \
	int src_n = (int)src.shape.y;                  \
	int src_c = (int)src.shape.x / (src_w*src_h);  \
	int dst_w = (src_w-window_w)/cfg.stride + 1;   \
	int dst_h = (src_h-window_h)/cfg.stride + 1;   \
	int dst_n = (int)dst.shape.y;                  \
	int dst_c = (int)dst.shape.x / (dst_w*dst_h);  \
	int n = src_n;                                 \
	int c = src_c;                                 \
	SM_ASSERTMSG(dst.shape.x == dst_c*dst_w*dst_h, "Number of columns in dst must be dst_w*dst_h*dst_channels"); \
	SM_ASSERTMSG((int)src.shape.x % (src_w*src_h) == 0, "Number of columns in src must be divisible by src_w*src_h"); \
	SM_ASSERTMSG((int)dst.shape.x % (dst_w*dst_h) == 0, "Number of columns in dst must be divisible by dst_w*dst_h"); \
	SM_ASSERTMSG(src_n == dst_n, "Number of src rows must match number of dst rows"); \
	SM_ASSERTMSG(src_c == dst_c, "Number of src channels must match number of dst channels"); \
	cudnnDataType_t cudnn_dt = (src.dtype == f32) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE; \
	cudnnPoolingMode_t mode = cfg.mode ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_MAX; \
	ccd(SetTensor4dDescriptor, dst_desc, CUDNN_TENSOR_NCHW, cudnn_dt, n, c, dst_h, dst_w); \
	ccd(SetTensor4dDescriptor, src_desc, CUDNN_TENSOR_NCHW, cudnn_dt, n, c, src_h, src_w); \
	ccd(SetPooling2dDescriptor, pool_desc, mode, window_h, window_w, 0, 0, cfg.stride, cfg.stride);

// macro to define "void* alpha" and "void* beta" to correspond to the accumulate mode (true or false)
#define SETUP_CUDNN_ACCUM_ALPHABETA(accum) \
	float  alpha32 = 1.0f, beta32 = accum ? 1.0f : 0.0f; \
	double alpha64 = 1.0,  beta64 = accum ? 1.0  : 0.0;  \
	void* alpha = (cudnn_dt == CUDNN_DATA_FLOAT) ? (void*)&alpha32 : (void*)&alpha64; \
	void* beta  = (cudnn_dt == CUDNN_DATA_FLOAT) ? (void*)&beta32  : (void*)&beta64;


// Obviously this is not going to work with multiple threads using cuDNN but
// who cares, this is intended to be used from Python for now anyways.
static cudnnHandle_t handle = 0;
static cudnnTensorDescriptor_t dst_desc = 0;
static cudnnTensorDescriptor_t src_desc = 0;
static cudnnFilterDescriptor_t filter_desc = 0;
static cudnnConvolutionDescriptor_t conv_desc = 0;
static cudnnPoolingDescriptor_t pool_desc = 0;
static void* workspace = 0;
static size_t workspace_size = 0;

struct conv2cfg_t {
	int src_w, src_h;
	int filter_w, filter_h;
	int stride;
	int accumulate;
	int cpu_check;
};

struct featuremap_bias_cfg_t {
	int dims;
	int accumulate;
	int cpu_check;
};

struct pool2cfg_t {
	int mode; // 0 == max, 1 == avg
	int src_w, src_h;
	int window_w, window_h;
	int stride;
	int accumulate;
	int cpu_check;
};

 // provided so that smat vm can delete this thing cleanly if it's used as instruction argument.
static void conv2cfg_deleter(void* ptr) { delete (conv2cfg_t*)ptr; }
static void featuremap_bias_cfg_deleter(void* ptr) { delete (featuremap_bias_cfg_t*)ptr; }
static void pool2cfg_deleter(void* ptr) { delete (pool2cfg_t*)ptr; }

static void initialize_cudnn()
{
	ccd(Create, &handle);
	ccd(CreateTensorDescriptor, &dst_desc);
	ccd(CreateTensorDescriptor, &src_desc);
	ccd(CreateFilterDescriptor, &filter_desc);
	ccd(CreateConvolutionDescriptor, &conv_desc);
	ccd(CreatePoolingDescriptor, &pool_desc);
}

// All cpu_check functions assume use of CUDNN_TENSOR_NCHW format and convolution in CUDNN_CROSS_CORRELATION mode.

// Perform CPU check of a GPU forward 2D convolution, where dst was computed from src and filters.
template <typename T>
static void cpu_check_conv2(const T* src, const T* filters, const T* dst,
                            int n,  // number of input images (size of minibatch)
                            int k,  // number of output channels (number of filters)
                            int c,  // number of input channels (dimensionality of src)
                            int dst_w, int dst_h, // size of output image (computed from src_h,src_w,filter_w,filter_h, and stride)
                            int src_w, int src_h, // size of input image
                            int filter_w, int filter_h, // size of filter
                            int stride) // stride between applications of filter (i.e. bigger stride => smaller dst size)
{
	// Get host copy (_src,_filters,_dst) of device arrays (src,filters,dst) so that we 
	// can compare each _dst[i] to a reference that we will compute from _src and _filters.
	T* _src = new T[n*c*src_h*src_w];
	T* _dst = new T[n*k*dst_h*dst_w];
	T* _filters = new T[k*c*filter_h*filter_w];

	cudaMemcpy(_src, src, sizeof(T)*n*c*src_h*src_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_dst, dst, sizeof(T)*n*k*dst_h*dst_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_filters, filters, sizeof(T)*k*c*filter_h*filter_w, cudaMemcpyDeviceToHost);

	// Variables to keep track of relative error seen so far
	T max_rel_err = 0;
	T z_of_max_rel_err_cpu = 0;
	T z_of_max_rel_err_gpu = 0;

	for (int img = 0; img < n; ++img) {
		for (int out_channel = 0; out_channel < k; ++out_channel) {
			for (int out_y = 0; out_y < dst_h; ++out_y) {
				for (int out_x = 0; out_x < dst_w; ++out_x) {
					// Calculate convolution response z for output position (out_x, out_y)
					// of filter (out_channel) and image (img).
					T z = 0;
					for (int j = 0; j < filter_h; ++j) {
						for (int i = 0; i < filter_w; ++i) {
							for (int in_channel = 0; in_channel < c; ++in_channel) {
								// x = src[img][in_channel][out_y*stride+j][out_x*stride+i]
								// w = filters[out_channel][in_channel][j][i]
								T x = _src[(img)*(c*src_h*src_w) + (in_channel)*(src_h*src_w) + (out_y*stride+j)*(src_w) + (out_x*stride+i)];
								T w = _filters[(out_channel)*(c*filter_h*filter_w) + (in_channel)*(filter_h*filter_w) + (j)*(filter_w) + i];
								z += x*w;
							}
						}
					}
					// Compare the reference value z to the corresponding value from _z
					// that was computed on the GPU.
					// _z = dst[img][out_channel][out_y][out_x]
					T _z = _dst[(img)*(dst_h*dst_w*k) + (out_channel)*(dst_h*dst_w) + (out_y)*(dst_w) + out_x];
					T rel_err = abs(_z-z)/max(abs(z),(T)1e-12);
					if (max_rel_err < rel_err) {
						max_rel_err = rel_err;
						z_of_max_rel_err_gpu = _z;
						z_of_max_rel_err_cpu = z;
					}
				}
			}
		}
	}

	delete[] _filters;
	delete[] _src;
	delete[] _dst;

	SM_ASSERTMSG(max_rel_err <= 0.01, format("ValueError: conv2 implementation had relative error of %f compared to CPU implementation (%f gpu, %f cpu).\n", (double)max_rel_err, (double)z_of_max_rel_err_gpu, (double)z_of_max_rel_err_cpu).c_str());
}

// Perform CPU check of a GPU back-propagation (of 2D convolution) to the incoming 
// feature maps, where srcgrad (src differentials) was computed from filters and 
// dstgrad (dst differentials).
template <typename T>
static void cpu_check_conv2_srcgrad(const T* srcgrad, const T* filters, const T* dstgrad,
                                   int n, int k, int c,
                                   int dst_w, int dst_h,
                                   int src_w, int src_h,
                                   int filter_w, int filter_h,
                                   int stride)
{
	T* _srcgrad = new T[n*c*src_h*src_w];
	T* _dstgrad = new T[n*k*dst_h*dst_w];
	T* _filters = new T[k*c*filter_h*filter_w];
	cudaMemcpy(_srcgrad, srcgrad, sizeof(T)*n*c*src_h*src_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_dstgrad, dstgrad, sizeof(T)*n*k*dst_h*dst_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_filters, filters, sizeof(T)*k*c*filter_h*filter_w, cudaMemcpyDeviceToHost);

	T max_rel_err = 0;
	T dx_of_max_rel_err_cpu = 0;
	T dx_of_max_rel_err_gpu = 0;

	T* srcgrad_ref = new T[n*c*src_h*src_w];
	for (int i = 0; i < n*c*src_h*src_w; ++i)
		srcgrad_ref[i] = 0; // initialize each element of temporary array to 0, accumulate into it, then compare to _srcgrad when finished

	for (int img = 0; img < n; ++img) {
		for (int out_channel = 0; out_channel < k; ++out_channel) {
			for (int out_y = 0; out_y < dst_h; ++out_y) {
				for (int out_x = 0; out_x < dst_w; ++out_x) {
					// dz = dstgrad[img][out_channel][out_y][out_x]
					T dz = _dstgrad[(img)*(dst_h*dst_w*k) + (out_channel)*(dst_h*dst_w) + out_y*(dst_w) + out_x];
					for (int j = 0; j < filter_h; ++j) {
						for (int i = 0; i < filter_w; ++i) {
							for (int in_channel = 0; in_channel < c; ++in_channel) {
								// dx = srcgrad_ref[img][in_channel][out_y*stride+j][out_x*stride+i]
								// w  = filters[out_channel][in_channel][j][i]
								T& dx = srcgrad_ref[(img)*(c*src_h*src_w) + (in_channel)*(src_h*src_w) + (out_y*stride+j)*(src_w) + (out_x*stride+i)];
								T   w = _filters[(out_channel)*(c*filter_h*filter_w) + (in_channel)*(filter_h*filter_w) + (j)*(filter_w) + i];
								dx += dz*w;
							}
						}
					}
				}
			}
		}
	}

	// Now that we've accumulated all the dx values into tmp, 
	for (int i = 0; i < n*c*src_h*src_w; ++i) {
		T dx = srcgrad_ref[i];
		T _dx = _srcgrad[i];
		T rel_err = abs(_dx-dx)/max(abs(dx),(T)1e-12);
		if (max_rel_err < rel_err) {
			max_rel_err = rel_err;
			dx_of_max_rel_err_gpu = _dx;
			dx_of_max_rel_err_cpu = dx;
		}
	}

	delete[] srcgrad_ref;
	delete[] _filters;
	delete[] _dstgrad;
	delete[] _srcgrad;

	SM_ASSERTMSG(max_rel_err <= 0.01, format("ValueError: conv2_srcgrad implementation had relative error of %f compared to CPU implementation (%f gpu, %f cpu).\n", (double)max_rel_err, (double)dx_of_max_rel_err_gpu, (double)dx_of_max_rel_err_cpu).c_str());
}

// Perform CPU check of a GPU back-propagation (of 2D convolution) to the incoming 
// filter weights, where filtersgrad was computed from src and dstgrad (dst differentials).
template <typename T>
static void cpu_check_conv2_filtersgrad(const T* src, const T* filtersgrad, const T* dstgrad,
                                        int n, int k, int c,
                                        int dst_w, int dst_h,
                                        int src_w, int src_h,
                                        int filter_w, int filter_h,
                                        int stride)
{
	T* _filtersgrad = new T[k*c*filter_h*filter_w];
	T* _dstgrad = new T[n*k*dst_h*dst_w];
	T* _src     = new T[n*c*src_h*src_w];
	cudaMemcpy(_filtersgrad, filtersgrad, sizeof(T)*k*c*filter_h*filter_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_dstgrad, dstgrad, sizeof(T)*n*k*dst_h*dst_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_src    , src    , sizeof(T)*n*c*src_h*src_w, cudaMemcpyDeviceToHost);

	T max_rel_err = 0;
	T dw_of_max_rel_err_cpu = 0;
	T dw_of_max_rel_err_gpu = 0;

	T* filtersgrad_ref = new T[k*c*filter_h*filter_w];
	for (int i = 0; i < k*c*filter_h*filter_w; ++i)
		filtersgrad_ref[i] = 0; // initialize each element of temporary array to 0, accumulate into it, then compare to _filtersgrad when finished

	for (int img = 0; img < n; ++img) {
		for (int out_channel = 0; out_channel < k; ++out_channel) {

			for (int out_y = 0; out_y < dst_h; ++out_y) {
				for (int out_x = 0; out_x < dst_w; ++out_x) {
					// dz = dstgrad[img][out_channel][out_y][out_x]
					T dz = _dstgrad[(img)*(dst_h*dst_w*k) + (out_channel)*(dst_h*dst_w) + out_y*(dst_w) + out_x];

					for (int j = 0; j < filter_h; ++j) {
						for (int i = 0; i < filter_w; ++i) {
							for (int in_channel = 0; in_channel < c; ++in_channel) {
								// x  = src[img][in_channel][out_y*stride+j][out_x*stride+i]
								// dw = filtersgrad_ref[out_channel][in_channel][j][i]
								T   x = _src[(img)*(src_h*src_w*c) + (in_channel)*(src_h*src_w) + (out_y*stride+j)*(src_w) + (out_x*stride+i)];
								T& dw = filtersgrad_ref[(out_channel)*(c*filter_h*filter_w) + (in_channel)*(filter_h*filter_w) + (j)*(filter_w) + i];
								dw += x*dz;
							}
						}
					}
				}
			}

		}
	}

	for (int i = 0; i < k*c*filter_h*filter_w; ++i) {
		T dw = filtersgrad_ref[i];
		T _dw = _filtersgrad[i];
		T rel_err = abs(_dw-dw)/max(abs(dw),(T)1e-12);
		if (max_rel_err < rel_err) {
			max_rel_err = rel_err;
			dw_of_max_rel_err_gpu = _dw;
			dw_of_max_rel_err_cpu = dw;
		}
	}

	delete[] filtersgrad_ref;
	delete[] _src;
	delete[] _dstgrad;
	delete[] _filtersgrad;

	SM_ASSERTMSG(max_rel_err <= 0.01, format("ValueError: conv2_filtersgrad implementation had relative error of %f compared to CPU implementation (%f gpu, %f cpu).\n", (double)max_rel_err, (double)dw_of_max_rel_err_gpu, (double)dw_of_max_rel_err_cpu).c_str());
}


// Perform CPU check of a GPU featuremap bias operation, where fmaps is incremented by bias.
template <typename T>
static void cpu_check_featuremap_bias(
                            const T* _old_fmaps,  // host pointer to copy of fmaps before function was complete (before they were 'biased')
                            const T* fmaps, const T* bias, // device pointers to fmaps/bias state after function is complete
                            int n,  // number of input images (size of minibatch)
                            int c,  // number of feature maps
                            int d)  // number of elements per feature map
{
	// Get host copy (_fmaps,_bias) of device arrays (fmaps,bias) so that we 
	// can compare _fmaps to the correct sums of _old_fmaps and bias.
	T* _fmaps = new T[n*c*d];
	T* _bias  = new T[c];

	cudaMemcpy(_fmaps, fmaps, sizeof(T)*n*c*d, cudaMemcpyDeviceToHost);
	cudaMemcpy(_bias,  bias,  sizeof(T)*c,     cudaMemcpyDeviceToHost);

	// Variables to keep track of relative error seen so far
	T max_rel_err = 0;
	T x_of_max_rel_err_cpu = 0;
	T x_of_max_rel_err_gpu = 0;

	for (int img = 0; img < n; ++img) {
		for (int channel = 0; channel < c; ++channel) {
			T b = _bias[channel];
			for (int pixel = 0; pixel < d; ++pixel) {
				// old_x = old_fmaps[img][channel][pixel]
				// new_x = fmaps[img][channel][pixel]
				T old_x = _old_fmaps[(img)*(c*d) + (channel)*(d) + pixel];
				T new_x = _fmaps[(img)*(c*d) + (channel)*(d) + pixel];
				T ref_x = old_x + b;

				// Compare the reference value ref_x to the corresponding value new_x
				// that was computed on the GPU.
				T rel_err = abs(new_x-ref_x)/max(abs(ref_x),(T)1e-12);
				if (max_rel_err < rel_err) {
					max_rel_err = rel_err;
					x_of_max_rel_err_gpu = new_x;
					x_of_max_rel_err_cpu = ref_x;
				}
			}
		}
	}

	delete[] _bias;
	delete[] _fmaps;
	delete[] _old_fmaps;

	SM_ASSERTMSG(max_rel_err <= 0.01, format("ValueError: featuremap_bias implementation had relative error of %f compared to CPU implementation (%f gpu, %f cpu).\n", (double)max_rel_err, (double)x_of_max_rel_err_gpu, (double)x_of_max_rel_err_cpu).c_str());
}

// Perform CPU check of a GPU featuremap bias operation, where fmaps is incremented by bias.
template <typename T>
static void cpu_check_featuremap_bias_grad(
                            const T* fmapsgrad, const T* biasgrad, // device pointers to fmapsgrad/biasgrad state after function is complete, assuming no accumulation
                            int n,  // number of input images (size of minibatch)
                            int c,  // number of feature maps
                            int d)  // number of elements per feature map
{
	// Get host copy (_fmaps,_bias) of device arrays (fmaps,bias) so that we 
	// can compare _fmaps to the correct sums of _old_fmaps and bias.
	T* _fmapsgrad = new T[n*c*d];
	T* _biasgrad  = new T[c];

	cudaMemcpy(_fmapsgrad, fmapsgrad, sizeof(T)*n*c*d, cudaMemcpyDeviceToHost);
	cudaMemcpy(_biasgrad,  biasgrad,  sizeof(T)*c, cudaMemcpyDeviceToHost);

	// Variables to keep track of relative error seen so far
	T max_rel_err = 0;
	T db_of_max_rel_err_cpu = 0;
	T db_of_max_rel_err_gpu = 0;

	for (int channel = 0; channel < c; ++channel) {
		T db = 0;
		for (int img = 0; img < n; ++img) {
			for (int pixel = 0; pixel < d; ++pixel) {
				// dx = fmapsgrad[img][channel][pixel]
				T dx = _fmapsgrad[(img)*(c*d) + (channel)*(d) + pixel];
				db += dx;
			}
		}
		// Compare the reference value ref_x to the corresponding value new_x
		// that was computed on the GPU.
		T _db = _biasgrad[channel];
		T rel_err = abs(_db-db)/max(abs(db),(T)1e-12);
		if (max_rel_err < rel_err) {
			max_rel_err = rel_err;
			db_of_max_rel_err_gpu = _db;
			db_of_max_rel_err_cpu = db;
		}
	}

	delete[] _biasgrad;
	delete[] _fmapsgrad;

	SM_ASSERTMSG(max_rel_err <= 0.01, format("ValueError: featuremap_bias_grad implementation had relative error of %f compared to CPU implementation (%f gpu, %f cpu).\n", (double)max_rel_err, (double)db_of_max_rel_err_gpu, (double)db_of_max_rel_err_cpu).c_str());
}

// Perform CPU check of a GPU forward 2D pooling operation, where dst was computed from src and filters.
template <typename T>
static void cpu_check_pool2(const T* src, const T* dst,
                            int n,  // number of input images (size of minibatch)
                            int c,  // number of input and output channels (feature maps)
                            int dst_w, int dst_h, // size of output image (computed from src_h,src_w,window_w,window_h, and stride)
                            int src_w, int src_h, // size of input image
                            int window_w, int window_h, // size of pooling region
                            int stride,  // stride between applications of filter (i.e. bigger stride => smaller dst size)
                            int mode)
{
	// Get host copy (_src,_dst) of device arrays (src,dst) so that we 
	// can compare each _dst[i] to a reference that we will compute from _src.
	T* _src = new T[n*c*src_h*src_w];
	T* _dst = new T[n*c*dst_h*dst_w];

	cudaMemcpy(_src, src, sizeof(T)*n*c*src_h*src_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_dst, dst, sizeof(T)*n*c*dst_h*dst_w, cudaMemcpyDeviceToHost);

	// Variables to keep track of relative error seen so far
	T max_rel_err = 0;
	T z_of_max_rel_err_cpu = 0;
	T z_of_max_rel_err_gpu = 0;

	for (int img = 0; img < n; ++img) {
		for (int channel = 0; channel < c; ++channel) {
			for (int out_y = 0; out_y < dst_h; ++out_y) {
				for (int out_x = 0; out_x < dst_w; ++out_x) {
					// Calculate response z for output position (out_x, out_y) of filter (channel) and image (img).
					T z = (mode == 0) ? 
						     _src[(img)*(c*src_h*src_w) + (channel)*(src_h*src_w) + (out_y*stride)*(src_w) + (out_x*stride)] // first value in this channel
					         : 0;

					for (int j = 0; j < window_h; ++j) {
						for (int i = 0; i < window_w; ++i) {
							// x = src[img][channel][out_y*stride+j][out_x*stride+i]
							T x = _src[(img)*(c*src_h*src_w) + (channel)*(src_h*src_w) + (out_y*stride+j)*(src_w) + (out_x*stride+i)];
							if (mode == 0) {
								if (x > z)  // take max
									z = x;
							} else if (mode == 1) {
								z += x;  // take sum
							}
						}
					}
					if (mode == 1)
						z /= (window_w * window_h);

					// Compare the reference value z to the corresponding value from _z
					// that was computed on the GPU.
					// _z = dst[img][channel][out_y][out_x]
					T _z = _dst[(img)*(dst_h*dst_w*c) + (channel)*(dst_h*dst_w) + (out_y)*(dst_w) + out_x];

					T rel_err = abs(_z-z)/max(abs(z),(T)1e-12);
					if (max_rel_err < rel_err) {
						max_rel_err = rel_err;
						z_of_max_rel_err_gpu = _z;
						z_of_max_rel_err_cpu = z;
					}
				}
			}
		}
	}

	delete[] _dst;
	delete[] _src;

	SM_ASSERTMSG(max_rel_err <= 0.01, format("ValueError: pool2 implementation had relative error of %f compared to CPU implementation (%f gpu, %f cpu).\n", (double)max_rel_err, (double)z_of_max_rel_err_gpu, (double)z_of_max_rel_err_cpu).c_str());
}


// Perform CPU check of a GPU backpropagated 2D pooling, where dst was computed from src and filters.
template <typename T>
static void cpu_check_pool2_grad(const T* src, const T* srcgrad,  const T* dst, const T* dstgrad,
                                 int n,  // number of input images (size of minibatch)
                                 int c,  // number of input and output channels (feature maps)
                                 int dst_w, int dst_h, // size of output image (computed from src_h,src_w,window_w,window_h, and stride)
                                 int src_w, int src_h, // size of input image
                                 int window_w, int window_h, // size of pooling region
                                 int stride,  // stride between applications of filter (i.e. bigger stride => smaller dst size)
                                 int mode)
{
	// Get host copy (_src,_dst) of device arrays (src,dst) so that we 
	// can compare each _dst[i] to a reference that we will compute from _src.
	T* _src     = new T[n*c*src_h*src_w];
	T* _srcgrad = new T[n*c*src_h*src_w];
	T* _dst     = new T[n*c*dst_h*dst_w];
	T* _dstgrad = new T[n*c*dst_h*dst_w];

	cudaMemcpy(_src,     src,     sizeof(T)*n*c*src_h*src_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_srcgrad, srcgrad, sizeof(T)*n*c*src_h*src_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_dst,     dst,     sizeof(T)*n*c*dst_h*dst_w, cudaMemcpyDeviceToHost);
	cudaMemcpy(_dstgrad, dstgrad, sizeof(T)*n*c*dst_h*dst_w, cudaMemcpyDeviceToHost);

	// Variables to keep track of relative error seen so far
	T max_rel_err = 0;
	T dx_of_max_rel_err_cpu = 0;
	T dx_of_max_rel_err_gpu = 0;

	T* srcgrad_ref = new T[n*c*src_h*src_w];
	for (int i = 0; i < n*c*src_h*src_w; ++i)
		srcgrad_ref[i] = 0; // initialize each element of temporary array to 0, accumulate into it, then compare to _srcgrad when finished

	for (int img = 0; img < n; ++img) {
		for (int channel = 0; channel < c; ++channel) {
			for (int out_y = 0; out_y < dst_h; ++out_y) {
				for (int out_x = 0; out_x < dst_w; ++out_x) {
					// dz = dstgrad[img][channel][out_y][out_x]
					// z  = dst    [img][channel][out_y][out_x]
					T dz = _dstgrad[(img)*(dst_h*dst_w*c) + (channel)*(dst_h*dst_w) + out_y*(dst_w) + out_x];
					T z  = _dst    [(img)*(dst_h*dst_w*c) + (channel)*(dst_h*dst_w) + out_y*(dst_w) + out_x];
					for (int j = 0; j < window_h; ++j) {
						for (int i = 0; i < window_w; ++i) {
							// dx = srcgrad_ref[img][channel][out_y*stride+j][out_x*stride+i]
							//  x = src        [img][channel][out_y*stride+j][out_x*stride+i]
							T& dx = srcgrad_ref[(img)*(c*src_h*src_w) + (channel)*(src_h*src_w) + (out_y*stride+j)*(src_w) + (out_x*stride+i)];
							T   x = _src       [(img)*(c*src_h*src_w) + (channel)*(src_h*src_w) + (out_y*stride+j)*(src_w) + (out_x*stride+i)];
							if (mode == 0) {
								if (z == x)
									dx = dz; // for max pooling, set dx only if this position (i,j) in src (x) was chosen in forward pooling as the max value in dst (z)
							} else {
								dx += dz; // for average pooling, accumulate
							}
						}
					}
				}
			}
		}
	}

	// If average pooling, divide by number of elements per window
	if (mode == 1) {
		for (int i = 0; i < n*c*src_h*src_w; ++i)
			srcgrad_ref[i] /= (window_w * window_h);
	}


	// Now that we've accumulated all the dx values into tmp, 
	for (int i = 0; i < n*c*src_h*src_w; ++i) {
		T dx = srcgrad_ref[i];
		T _dx = _srcgrad[i];
		T rel_err = abs(_dx-dx)/max(abs(dx),(T)1e-12);
		if (max_rel_err < rel_err) {
			max_rel_err = rel_err;
			dx_of_max_rel_err_gpu = _dx;
			dx_of_max_rel_err_cpu = dx;
		}
	}

	delete[] _dstgrad;
	delete[] _dst;
	delete[] _srcgrad;
	delete[] _src;

	SM_ASSERTMSG(max_rel_err <= 0.01, format("ValueError: pool2_grad implementation had relative error of %f compared to CPU implementation (%f gpu, %f cpu).\n", (double)max_rel_err, (double)dx_of_max_rel_err_gpu, (double)dx_of_max_rel_err_cpu).c_str());
}


/////////////////////////////////////////////////////////////////////////////////////////////////////

#define CPUCHECK_CONV2_RESULT(funcname, src, filters, dst) \
	if (cfg.cpu_check) { \
		SM_ASSERTMSG(!cfg.accumulate, "Cannot use cpu_check when accumulate is enabled"); \
		if (src.dtype == f32) cpu_check_##funcname<float >(src.get<float* >(), filters.get<float* >(), dst.get<float* >(), n, k, c, dst_w, dst_h, src_w, src_h, filter_w, filter_h, cfg.stride); \
		else                  cpu_check_##funcname<double>(src.get<double*>(), filters.get<double*>(), dst.get<double*>(), n, k, c, dst_w, dst_h, src_w, src_h, filter_w, filter_h, cfg.stride); \
	} 


// Shorthand:
//  w = width
//  h = height
//  n = number of images
//  c = number of src channels (colors)
//  k = number of dst channels (feature maps)
//
static void execute_conv2(opcode_t,
                          const argument& src,      // (n) x (c*src_h*src_w),       where memory indexed as src[img][in_channel][pixel_y][pixel_x]
                          const argument& filters,  // (k) x (n*filter_h*filter_w), where memory indexed as filters[out_channel][in_channel][filter_y][filter_x]
                          const argument& dst,      // (n) x (k*dst_h*dst_w),       where memory indexed as dst[img][out_channel][pixel_y][pixel_x]
                          const argument& _cfg)
{
	SETUP_CONV2_DESCRIPTORS(src, filters, dst)

	// Choose fastest algorithm
	cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
	ccd(GetConvolutionForwardAlgorithm, handle, src_desc, filter_desc, conv_desc, dst_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

	// Grow the algorithm's workspace if necessary.
	size_t new_workspace_size = 0;
	ccd(GetConvolutionForwardWorkspaceSize, handle, src_desc, filter_desc, conv_desc, dst_desc, algo, &new_workspace_size);
	if (new_workspace_size > workspace_size) {
		if (workspace)
			cudaFree(workspace);
		workspace_size = new_workspace_size;
		cudaMalloc(&workspace, workspace_size);
	}

	SETUP_CUDNN_ACCUM_ALPHABETA(cfg.accumulate)
	ccd(ConvolutionForward, handle, alpha, src_desc, src.get<void*>(), filter_desc, filters.get<void*>(), conv_desc, algo, workspace, workspace_size, beta, dst_desc, dst.get<void*>());
	CPUCHECK_CONV2_RESULT(conv2, src, filters, dst);
}

///////////////////////////////////////////////////////

static void execute_conv2_srcgrad(opcode_t,
                                  const argument& srcgrad, // (n) x (c*src_h*src_w),       where memory indexed as srcgrad[img][in_channel][pixel_y][pixel_x]
                                  const argument& filters, // (k) x (n*filter_h*filter_w), where memory indexed as filters[out_channel][in_channel][filter_y][filter_x]
                                  const argument& dstgrad, // (n) x (k*dst_h*dst_w),       where memory indexed as dstgrad[img][out_channel][pixel_y][pixel_x]
                                  const argument& _cfg)
{
	SETUP_CONV2_DESCRIPTORS(srcgrad, filters, dstgrad)
	SETUP_CUDNN_ACCUM_ALPHABETA(cfg.accumulate)
	ccd(ConvolutionBackwardData, handle, alpha, filter_desc, filters.get<void*>(), dst_desc, dstgrad.get<void*>(), conv_desc, beta, src_desc, srcgrad.get<void*>());
	CPUCHECK_CONV2_RESULT(conv2_srcgrad, srcgrad, filters, dstgrad);
}

static void execute_conv2_filtersgrad(opcode_t,
                                      const argument& src,         // (n) x (c*src_h*src_w),       where memory indexed as src[img][in_channel][pixel_y][pixel_y]
                                      const argument& filtersgrad, // (k) x (n*filter_h*filter_w), where memory indexed as filtersgrad[out_channel][in_channel][filter_y][filter_x]
                                      const argument& dstgrad,     // (n) x (k*dst_h*dst_w),       where memory indexed as dstgrad[img][out_channel][pixel_y][pixel_x]
                                      const argument& _cfg)
{
	SETUP_CONV2_DESCRIPTORS(src, filtersgrad, dstgrad)
	SETUP_CUDNN_ACCUM_ALPHABETA(cfg.accumulate)
	ccd(ConvolutionBackwardFilter, handle, alpha, src_desc, src.get<void*>(), dst_desc, dstgrad.get<void*>(), conv_desc, beta, filter_desc, filtersgrad.get<void*>());
	CPUCHECK_CONV2_RESULT(conv2_filtersgrad, src, filtersgrad, dstgrad);
}

// Shorthand:
//  n = number of images
//  c = number of feature maps (channels)
//  d = number of elements per feature map
//
static void execute_featuremap_bias(opcode_t,
                          const argument& fmaps,   // (n) x (c*d),       where memory indexed as src[img][channel][pixel]
                          const argument& bias,    // (1) x (c)
                          const argument& _cfg)
{
	featuremap_bias_cfg_t& cfg = *_cfg.get<featuremap_bias_cfg_t*>();
	int n = fmaps.shape.y;  // number of images
	int c = bias.shape.y;   // number of feature maps
	int d = cfg.dims;       // number of elements per feature map
	cudnnDataType_t cudnn_dt = (fmaps.dtype == f32) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	ccd(SetTensor4dDescriptor, dst_desc, CUDNN_TENSOR_NCHW, cudnn_dt, n, c, d, 1);  // fmaps tensor is (n,c,d,1)
	ccd(SetTensor4dDescriptor, src_desc, CUDNN_TENSOR_NCHW, cudnn_dt, 1, c, 1, 1);  // bias  tensor is (1,c,1,1) for CUDNN_ADD_SAME_C
	SETUP_CUDNN_ACCUM_ALPHABETA(cfg.accumulate)

	// If we've been asked to do cpu_check, then copy the original featuremaps to CPU before calling the cuDNN kernel
	void* _old_fmaps = 0;
	if (cfg.cpu_check) {
		if (fmaps.dtype == f32) { _old_fmaps = new float[n*c*d];   cudaMemcpy(_old_fmaps, fmaps.get<float* >(), sizeof(float )*n*c*d, cudaMemcpyDeviceToHost); }
		else                    { _old_fmaps = new double[n*c*d];  cudaMemcpy(_old_fmaps, fmaps.get<double*>(), sizeof(double)*n*c*d, cudaMemcpyDeviceToHost); }
	}

	// Call cudnnAddTensor with CUDNN_ADD_SAME_C, which treats the bias tensor as k bias values, 
	// each broadcast across its entire corresponding featuremap (all d elements).
	ccd(AddTensor, handle, CUDNN_ADD_SAME_C, alpha, src_desc, bias.get<const void*>(), beta, dst_desc, fmaps.get<void*>());

	if (cfg.cpu_check) {
		if (fmaps.dtype == f32) cpu_check_featuremap_bias<float >((float* )_old_fmaps, fmaps.get<float* >(), bias.get<float* >(), n, c, d);
		else                    cpu_check_featuremap_bias<double>((double*)_old_fmaps, fmaps.get<double*>(), bias.get<double*>(), n, c, d);
	}
}


static void execute_featuremap_bias_grad(opcode_t,
                          const argument& fmapsgrad,   // (n) x (c*d),       where memory indexed as src[img][featuremap][pixel]
                          const argument& biasgrad,    // (1) x (c)
                          const argument& _cfg)
{
	featuremap_bias_cfg_t& cfg = *_cfg.get<featuremap_bias_cfg_t*>();
	int n = fmapsgrad.shape.y; // number of images
	int c = biasgrad.shape.y;  // number of feature maps
	int d = cfg.dims;          // number of elements per feature map
	cudnnDataType_t cudnn_dt = (fmapsgrad.dtype == f32) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	ccd(SetTensor4dDescriptor, src_desc, CUDNN_TENSOR_NCHW, cudnn_dt, n, c, d, 1);
	ccd(SetTensor4dDescriptor, dst_desc, CUDNN_TENSOR_NCHW, cudnn_dt, 1, c, 1, 1);
	SETUP_CUDNN_ACCUM_ALPHABETA(cfg.accumulate)
	ccd(ConvolutionBackwardBias, handle, alpha, src_desc, fmapsgrad.get<const void*>(), beta, dst_desc, biasgrad.get<void*>());
	if (cfg.cpu_check) {
		SM_ASSERTMSG(!cfg.accumulate, "Cannot use cpu_check when accumulate is enabled");
		if (fmapsgrad.dtype == f32) cpu_check_featuremap_bias_grad<float >(fmapsgrad.get<float* >(), biasgrad.get<float* >(), n, c, d);
		else                        cpu_check_featuremap_bias_grad<double>(fmapsgrad.get<double*>(), biasgrad.get<double*>(), n, c, d);
	}
}


// Shorthand:
//  w = width
//  h = height
//  n = number of images
//  c = number of src and dst channels (feature maps)
//
static void execute_pool2(opcode_t,
                          const argument& src,      // (n) x (c*src_h*src_w), where memory indexed as src[img][channel][pixel_y][pixel_x]
                          const argument& dst,      // (n) x (c*dst_h*dst_w), where memory indexed as dst[img][channel][pixel_y][pixel_x]
                          const argument& _cfg)
{
	SETUP_POOL2_DESCRIPTORS(src, dst)
	SETUP_CUDNN_ACCUM_ALPHABETA(cfg.accumulate)
	ccd(PoolingForward, handle, pool_desc, alpha, src_desc, src.get<void*>(), beta, dst_desc, dst.get<void*>());
	if (cfg.cpu_check) {
		SM_ASSERTMSG(!cfg.accumulate, "Cannot use cpu_check when accumulate is enabled");
		if (src.dtype == f32) cpu_check_pool2<float >(src.get<float* >(), dst.get<float* >(), n, c, dst_w, dst_h, src_w, src_h, window_w, window_h, cfg.stride, cfg.mode);
		else                  cpu_check_pool2<double>(src.get<double*>(), dst.get<double*>(), n, c, dst_w, dst_h, src_w, src_h, window_w, window_h, cfg.stride, cfg.mode);
	}
}

static void execute_pool2_grad(opcode_t,
                          const argument& src,        // (n) x (c*src_h*src_w), where memory indexed as src[img][channel][pixel_y][pixel_x] (input)
                          const argument& srcgrad,    // (n) x (c*src_h*src_w), where memory indexed as src[img][channel][pixel_y][pixel_x] (output)
                          const argument& dst,        // (n) x (c*dst_h*dst_w), where memory indexed as dst[img][channel][pixel_y][pixel_x] (input)
                          const argument& dstgrad,    // (n) x (c*dst_h*dst_w), where memory indexed as dst[img][channel][pixel_y][pixel_x] (input)
                          const argument& _cfg)
{
	SETUP_POOL2_DESCRIPTORS(src, dst)
	SETUP_CUDNN_ACCUM_ALPHABETA(cfg.accumulate)
	ccd(PoolingBackward, handle, pool_desc, 
		 alpha,
		 dst_desc, dst.get<const void*>(), 
		 dst_desc, dstgrad.get<const void*>(), 
		 src_desc, src.get<const void*>(),
		 beta,
		 src_desc, srcgrad.get<void*>());
	if (cfg.cpu_check) {
		SM_ASSERTMSG(!cfg.accumulate, "Cannot use cpu_check when accumulate is enabled");
		if (src.dtype == f32) cpu_check_pool2_grad<float >(src.get<float* >(), srcgrad.get<float* >(), dst.get<float* >(), dstgrad.get<float* >(), n, c, dst_w, dst_h, src_w, src_h, window_w, window_h, cfg.stride, cfg.mode);
		else                  cpu_check_pool2_grad<double>(src.get<double*>(), srcgrad.get<double*>(), dst.get<double*>(), dstgrad.get<double*>(), n, c, dst_w, dst_h, src_w, src_h, window_w, window_h, cfg.stride, cfg.mode);
	}
}

//////////////////////////////////////////////////////////////////////////////////////

opcode_t oc_conv2  = -1;
opcode_t oc_conv2_sgrad  = -1;
opcode_t oc_conv2_fgrad  = -1;
opcode_t oc_fmap_bias       = -1;
opcode_t oc_fmap_bias_grad  = -1;
opcode_t oc_pool2  = -1;
opcode_t oc_pool2_grad  = -1;

extern "C" {

#define CHECK_CONV2_ARGS(src, filters, dst) \
	SM_ASSERT(cfg->stride >= 1); \
	SM_ASSERT(src->dtype() == filters->dtype()); \
	SM_ASSERT(src->dtype() == dst->dtype());     \
	int dst_w = (cfg->src_w-cfg->filter_w)/cfg->stride + 1; \
	int dst_h = (cfg->src_h-cfg->filter_h)/cfg->stride + 1; \
	int dst_k = filters->shape().y;              \
	int dst_n = src->shape().y;                  \
	SM_ASSERTMSG(dst->shape().x == dst_k*dst_w*dst_h, "Number of columns in dst must be dst_w*dst_h*dst_channels"); \
	SM_ASSERTMSG(dst->shape().y == src->shape().y, "Number of rows in src and dst must match");

#define CHECK_POOL2_ARGS(src, dst) \
	SM_ASSERT(cfg->stride >= 1); \
	SM_ASSERT(src->dtype() == dst->dtype());     \
	int dst_w = (cfg->src_w-cfg->window_w)/cfg->stride + 1; \
	int dst_h = (cfg->src_h-cfg->window_h)/cfg->stride + 1; \
	int dst_c = dst->shape().x / (dst_w*dst_h);  \
	int dst_n = src->shape().y;                  \
	SM_ASSERTMSG(dst->shape().x % (dst_w*dst_h) == 0, "Number of columns in dst must be divisible by dst_w*dst_h"); \
	SM_ASSERTMSG(dst->shape().y == src->shape().y, "Number of rows in src and dst must match");

//  Called from Python via ctypes
SM_DLLEXPORT void api_conv2(const smat* src, const smat* filters, smat* dst, const conv2cfg_t* cfg)
{ 
	SM_API_TRY
	CHECK_CONV2_ARGS(src, filters, dst)
	thread_ctx().emit(oc_conv2, src->as_arg(), filters->as_arg(), dst->as_arg(), user_arg(new conv2cfg_t(*cfg), conv2cfg_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_conv2_srcgrad(smat* srcgrad, const smat* filters, const smat* dstgrad, const conv2cfg_t* cfg)
{ 
	SM_API_TRY
	CHECK_CONV2_ARGS(srcgrad, filters, dstgrad)
	thread_ctx().emit(oc_conv2_sgrad, srcgrad->as_arg(), filters->as_arg(), dstgrad->as_arg(), user_arg(new conv2cfg_t(*cfg), conv2cfg_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_conv2_filtersgrad(const smat* src, smat* filtersgrad, const smat* dstgrad, const conv2cfg_t* cfg)
{ 
	SM_API_TRY
	CHECK_CONV2_ARGS(src, filtersgrad, dstgrad)
	thread_ctx().emit(oc_conv2_fgrad, src->as_arg(), filtersgrad->as_arg(), dstgrad->as_arg(), user_arg(new conv2cfg_t(*cfg), conv2cfg_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_featuremap_bias(smat* fmaps, const smat* bias, const featuremap_bias_cfg_t* cfg)
{ 
	SM_API_TRY
	SM_ASSERT(fmaps->dtype() == bias->dtype());
	SM_ASSERT(fmaps->shape().x % cfg->dims == 0);
	SM_ASSERTMSG((fmaps->shape().x / cfg->dims) == bias->shape().y, "Number of rows in bias must equal number of feature maps");
	SM_ASSERTMSG(1 == bias->shape().x, "Bias must be (k x 1) where k is the number of feature maps");
	thread_ctx().emit(oc_fmap_bias, fmaps->as_arg(), bias->as_arg(), user_arg(new featuremap_bias_cfg_t(*cfg), featuremap_bias_cfg_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_featuremap_bias_grad(smat* fmapsgrad, const smat* biasgrad, const featuremap_bias_cfg_t* cfg)
{ 
	SM_API_TRY
	SM_ASSERT(fmapsgrad->dtype() == biasgrad->dtype());
	SM_ASSERT(fmapsgrad->shape().x % cfg->dims == 0);
	SM_ASSERTMSG((fmapsgrad->shape().x / cfg->dims) == biasgrad->shape().y, "Number of rows in bias must equal number of feature maps");
	SM_ASSERTMSG(1 == biasgrad->shape().x, "Bias must be (k x 1) where k is the number of feature maps");
	thread_ctx().emit(oc_fmap_bias_grad, fmapsgrad->as_arg(), biasgrad->as_arg(), user_arg(new featuremap_bias_cfg_t(*cfg), featuremap_bias_cfg_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_pool2(const smat* src, smat* dst, const pool2cfg_t* cfg)
{ 
	SM_API_TRY
	CHECK_POOL2_ARGS(src, dst)
	thread_ctx().emit(oc_pool2, src->as_arg(), dst->as_arg(), user_arg(new pool2cfg_t(*cfg), pool2cfg_deleter));
	SM_API_CATCH
}

SM_DLLEXPORT void api_pool2_grad(const smat* src, smat* srcgrad, const smat* dst, const smat* dstgrad, const pool2cfg_t* cfg)
{ 
	SM_API_TRY
	CHECK_POOL2_ARGS(srcgrad, dstgrad)
	thread_ctx().emit(oc_pool2_grad, src->as_arg(), srcgrad->as_arg(), dst->as_arg(), dstgrad->as_arg(), user_arg(new pool2cfg_t(*cfg), pool2cfg_deleter));
	SM_API_CATCH
}


//////////////////////////////////////////////////////////


SM_DLLEXPORT void register_ext()
{
	initialize_cudnn();

	oc_conv2  = add_instruction("conv2",
			iprop_none,
			aprop_in |aprop_float,      // src
			aprop_in |aprop_float,      // filters
			aprop_out|aprop_float,      // dst
			aprop_in |aprop_user        // cfg
		);

	oc_conv2_sgrad  = add_instruction("conv2_sgrad",
			iprop_none,
			aprop_out|aprop_float,      // srcgrad
			aprop_in |aprop_float,      // filters
			aprop_in |aprop_float,      // dstgrad
			aprop_in |aprop_user        // cfg
		);

	oc_conv2_fgrad  = add_instruction("conv2_fgrad",
			iprop_none,
			aprop_in |aprop_float,      // src
			aprop_out|aprop_float,      // filtersgrad
			aprop_in |aprop_float,      // dstgrad
			aprop_in |aprop_user        // cfg
		);

	oc_fmap_bias = add_instruction("fmap_bias",
			iprop_none,
			aprop_in |aprop_float,      // fmaps
			aprop_out|aprop_float,      // bias
			aprop_in |aprop_user        // cfg
		);

	oc_fmap_bias_grad = add_instruction("fmap_bias_grad",
			iprop_none,
			aprop_out|aprop_float,      // fmapsgrad
			aprop_in |aprop_float,      // biasgrad
			aprop_in |aprop_user        // cfg
		);

	oc_pool2  = add_instruction("pool2",
			iprop_none,
			aprop_in |aprop_float,      // src
			aprop_out|aprop_float,      // dst
			aprop_in |aprop_user        // cfg
		);

	oc_pool2_grad  = add_instruction("pool2_grad",
			iprop_none,
			aprop_in |aprop_float,      // src
			aprop_out|aprop_float,      // srcgrad
			aprop_in |aprop_float,      // dst
			aprop_in |aprop_float,      // dstgrad
			aprop_in |aprop_user        // cfg
		);

	add_instruction_impl(cuda_uuid, oc_conv2, execute_conv2, 0);
	add_instruction_impl(cuda_uuid, oc_conv2_sgrad, execute_conv2_srcgrad, 0);
	add_instruction_impl(cuda_uuid, oc_conv2_fgrad, execute_conv2_filtersgrad, 0);
	add_instruction_impl(cuda_uuid, oc_fmap_bias, execute_featuremap_bias, 0);
	add_instruction_impl(cuda_uuid, oc_fmap_bias_grad, execute_featuremap_bias_grad, 0);
	add_instruction_impl(cuda_uuid, oc_pool2, execute_pool2, 0);
	add_instruction_impl(cuda_uuid, oc_pool2_grad, execute_pool2_grad, 0);
}

}
