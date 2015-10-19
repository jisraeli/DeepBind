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
#include <base/random.h>
#include <random>
#include <memory>

using namespace std;

SM_NAMESPACE_BEGIN

size_t g_rand_seed = 0;
static mt19937 g_rand_gen;
static uniform_real_distribution<double> g_uniform_dbl;
static uniform_real_distribution<float> g_uniform_flt;
static uniform_int_distribution<int> g_uniform_int;
static normal_distribution<double> g_normal_dbl;
static normal_distribution<float> g_normal_flt;

BASE_EXPORT void    set_rand_seed(size_t seed)  { g_rand_seed = seed; g_rand_gen.seed((unsigned long)seed); }
BASE_EXPORT size_t  bump_rand_seed()            { return g_rand_seed *= 1234; }
BASE_EXPORT double  rand_double()  { return g_uniform_dbl(g_rand_gen); }
BASE_EXPORT float   rand_float()   { return g_uniform_flt(g_rand_gen); }
BASE_EXPORT int     rand_int()     { return g_uniform_int(g_rand_gen); }
BASE_EXPORT double  randn_double() { return g_normal_dbl(g_rand_gen);  }
BASE_EXPORT float   randn_float()  { return g_normal_flt(g_rand_gen);  }

SM_NAMESPACE_END
