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
#ifndef __SM_HEAP_H__
#define __SM_HEAP_H__

#include <smat/config.h>

SM_NAMESPACE_BEGIN

class heap;

// block_allocator:
//    Whenever another block of committed memory is needed, a 
//    request for another large chunk of memory will be passed to 
//    a subclass of block_allocator. 
//    By default, allocations come from the system heap.
//
class SM_EXPORT block_allocator {
public:
	virtual ~block_allocator();
	virtual void* alloc_block(size_t size) = 0;
	virtual void  free_block(void* ptr) = 0;
	virtual size_t get_total_memory() = 0;
	virtual size_t get_avail_memory() = 0;
};

// heap_alloc:
//    Information about an allocation, needed to properly free it later.
//    A typical heap allocation would store bookkeeping information in the
//    bytes immediately preceeding the allocation itself, e.g. addr[-16..-1].
//    However, we must assume that "addr" may by a device address.
//    So, instead it is the client code's responsibility to store this 
//    "bookkeeping" information around until the memory can be freed.
//
struct heap_alloc {
	SM_INLINE heap_alloc(): addr(0), size(0), bookkeeping(0) { }
	SM_INLINE heap_alloc(size_t addr, size_t size, size_t bookkeeping): addr((void*)addr), size(size), bookkeeping(bookkeeping) { }
	void*  addr;
	size_t size;
	size_t bookkeeping;
};

struct heap_status {
	size_t host_total;
	size_t host_avail;
	size_t host_used;
	size_t device_total;
	size_t device_avail;
	size_t device_used;
	size_t device_committed;
};

// heap:
//    A private heap defined in the machine's address space, which
//    may be host or device depending on which machine backend.
//    The underlying implementation is designed for relatively
//    large allocations (>1kb), not small ones. The heap bookkeeping
//    is always stored in host memory, NOT in machine memory.
//
class SM_EXPORT heap { SM_NOCOPY(heap)
public:
	// Ownership of the block_allocator instance IS transferred to this heap instance,
	// (When the heap is destroyed, the block_allocator instance will be deleted as well.)
	// Can't use unique_ptr because linux nvcc currently chokes on it.
	heap(size_t max_capacity=0, size_t align=8, block_allocator* balloc=0);
	~heap();

	void       set_pitch(size_t pitch);   // sets the pitch alignment of all "large" allocations, for devices that require very large alignments, like CUDA; must be called while heap is empty
	heap_alloc alloc(size_t size);        // allocate size bytes from one of the blocks
	void       free(heap_alloc& alloc);   // free an allocation; this also zeros out the allocation_t argument

	bool    empty()  const;       // have any allocations been made?
	size_t  capacity()  const;    // committed memory has been allocated from the block_allocator, but may be available for use by alloc()
	size_t  size()  const;        // how much of committed memory has been explicitly reserved through alloc()
	size_t  alloc_count() const;  // counts the number of distinct allocations still on the heap
	size_t  alignment() const;    // returns the current alignment
	size_t  pitch() const;        // returns the current pitch

	heap_status status() const;

protected:
	bool alloc_from_block(struct block_alloc& block, size_t& size, heap_alloc& result);

	size_t  _max_capacity;
	size_t  _capacity;
	size_t  _size;
	size_t  _align;
	size_t  _pitch;
	size_t  _alloc_count;
	block_allocator*  _balloc;
	class block_list* _buckets;
};

////////////////////////////////////////////////////////////////

SM_INLINE bool    heap::empty()       const { return _size == 0; }
SM_INLINE size_t  heap::capacity()    const { return _capacity;  }
SM_INLINE size_t  heap::size()        const { return _size;  }
SM_INLINE size_t  heap::alloc_count() const { return _alloc_count;  }
SM_INLINE size_t  heap::alignment()   const { return _align; }
SM_INLINE size_t  heap::pitch()       const { return _pitch; }

////////////////////////////////////////////////////////////////

SM_NAMESPACE_END

#endif // __SM_VIRTUAL_HEAP_H__
