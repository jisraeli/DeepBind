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
#include <smat/shape.h>
#include <base/util.h>
#include <base/assert.h>

SM_NAMESPACE_BEGIN

using namespace std;

string shape2str(const shape_t& shape)
{
	if (shape.z != 0) return format("(%lld,%lld,%lld)",(long long)shape.x,(long long)shape.y,(long long)shape.z);
	if (shape.y != 0) return format("(%lld,%lld)",(long long)shape.x,(long long)shape.y);
	return format("(%lld,)",(long long)shape.x);
}

coord_t fullstride(const shape_t& shape)
{
	return coord_t(shape.x == 0 ? 0 : 1,
	               shape.y == 0 ? 0 : shape.x,
	               shape.z == 0 ? 0 : shape.x*shape.y);
}

void slice_t::bind(isize_t dim)
{
	// "Bind" a slice to a specific dimension size, 
	// meaning negative indices get wrapped, and 
	// slice_end gets mapped to the actual dim size.
	if (first < 0) {
		first += dim;
		if (last <= 0) 
			last += dim;
	} else if (last < 0)
		last += dim;
	if (last == slice_end) 
		last = dim;
	SM_ASSERTMSG(first >= 0,"IndexError: Index out of range.\n");
	SM_ASSERTMSG(first <= last,"IndexError: Invalid slice.\n");
	SM_ASSERTMSG(last  <= dim,"IndexError: Index out of range.\n");
}

SM_NAMESPACE_END
