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
#include <smat/vm/heap.h>
#include <base/util.h>
#include <base/os.h>
#include <base/range.h>
#include <base/logging.h>
#include <base/assert.h>
#include <list>
#include <set>

SM_NAMESPACE_BEGIN

using namespace std;

// Maximum size of allocation that can go in each bucket.
const size_t c_bucket_allocsizes[] = {
	(1ull << 12) - 64,  //   4 KB
	(1ull << 16) - 64,  //  64 KB
	(1ull << 21) - 64,  //   2 MB
	(1ull << 25) - 64,  //  32 MB
	0xffffffffffffffffull,  //  last bucket allows allocation of has any size
};
const size_t c_bucket_blocksizes[] = {
	1ull << 22,  //   4 MB
	1ull << 24,  //  16 MB
	1ull << 26,  //  64 MB
	1ull << 27,  // 128 MB
	0,           //  last bucket's blocks must match each individual allocation size exactly
};
const int c_nbucket = sizeof(c_bucket_allocsizes) / sizeof(c_bucket_allocsizes[0]);

const int c_block_end_padding = 8; // For performance reasons, always leave some padding at end of a 
                                   // device-allocated block so that device operations can read (but not write!) 
                                   // past the end of an allocation, without worrying about an access violation.

// free_slot:
//    A contiguous range of memory, inside a block_alloc's memory range, 
//    that is free to be allocated by the client code.
//
struct free_slot {
	SM_INLINE free_slot(size_t base, size_t size): base(base), size(size) { }
	size_t base;
	size_t size;
};

struct compare_free_slot {
	SM_INLINE bool operator()(const free_slot& a, const free_slot& b) const { return a.base < b.base; }
};

class free_list: public set<free_slot,compare_free_slot> { };

// block_alloc:
//    A raw allocation of a large block of memory 
//    (several MB at least) from the device.
//    
struct block_alloc { SM_NOCOPY(block_alloc)
public:
	block_alloc(size_t base, size_t size): base(base), size(size) { slots.insert(free_slot(base,size)); }
	size_t base;
	size_t size;
	free_list slots; 
	// class block_list* bucket; // TODO: a block_alloc should also know what bucket it belongs to, 
	//                            so that when an allocation is freed it gets promoted to a higher-level bucket
};

class block_list: public list<block_alloc> { };

//////////////////////////////////////////////////////////////////////

block_allocator::~block_allocator() { }

class host_block_allocator: public block_allocator {
public:
	virtual void* alloc_block(size_t size) { return malloc(size); }
	virtual void  free_block(void* ptr)    { free(ptr); }
	virtual size_t get_total_memory()      { return get_system_memory_total(); }
	virtual size_t get_avail_memory()      { return get_system_memory_avail(); }
};

host_block_allocator s_default_block_allocator;

heap::heap(size_t max_capacity, size_t align, block_allocator* balloc)
: _max_capacity(max_capacity)
, _capacity(0)
, _size(0)
, _align(align)
, _pitch(align)
, _alloc_count(0)
, _balloc(balloc ? balloc : &s_default_block_allocator)
, _buckets(new block_list[c_nbucket])
{
}

heap::~heap()
{
	// Free all the blocks by calling the _free_block callback
	for (auto i : range(c_nbucket))
		for (auto& block : _buckets[i])
			_balloc->free_block((void*)block.base);
	delete[] _buckets;
	if (_balloc != &s_default_block_allocator)
		delete _balloc;
}

void heap::set_pitch(size_t pitch)
{
	SM_ASSERTMSG(empty(),"RuntimeError: set_pitch cannot be called on a non-empty heap.");
	_pitch = pitch;
}

heap_alloc heap::alloc(size_t size)
{
	if (size == 0)
		size = _align;
	heap_alloc result;

	// Find a bucket based on the requested allocation size
	for (size_t i = 0; i < c_nbucket && !result.addr; ++i) {
		// Only look at blocks at least as big as the alignment size
		if (size > c_bucket_allocsizes[i])
			continue;

		// Try to find a slot in an existing block
		for (auto& block : _buckets[i])
			if (alloc_from_block(block,size,result))
				break;
	}

	// If that failed, add a new block and allocate from that instead.
	if (!result.addr) {
		// Find the best initial bucket for the new block that we'll need.
		size_t i = 0;
		while (size > c_bucket_allocsizes[i])
			++i;
		bool is_last_bucket = (i == c_nbucket-1);

		// First figure out how big the new block should be.
		size_t block_base = 0;
		size_t block_size = is_last_bucket ? size : c_bucket_blocksizes[i];
		if (_max_capacity > 0 && block_size + _capacity > _max_capacity)
			block_size = _max_capacity - _capacity;
		if (block_size < size)
			SM_ERROR(format("MemoryError: Heap reached its max_capacity at %llu bytes.",_max_capacity).c_str());

		// Allocate the new block from the block allocator.
		block_base = (size_t)_balloc->alloc_block(block_size+c_block_end_padding);
		if (!block_base)
			SM_ERROR(format("MemoryError: Failed to allocate block size (%llu bytes) from backend machine.",size).c_str());
		if (block_base % _pitch != 0)
			SM_ERROR(format("MemoryError: Backend machine returned address that is not aligned with pitch; the pitch must be incorrect for the device.",size).c_str());
		_capacity += block_size;
		_buckets[i].emplace_back(block_base,block_size);

		SM_LOG("heap","cmmt[0x%010llx]   0x%010llx bytes  \t(%.2f MB capacity)",block_base,(long long)block_size,(double)_capacity/1024/1024);

		// Now get an allocation from within that block.
		bool success = alloc_from_block(_buckets[i].back(),size,result);
		SM_ASSERT(success);
	}

	_alloc_count++;
	_size += size;
	SM_LOG("heap","allo[0x%010llx]   0x%010llx bytes  \t(%.2f MB used)",result.addr,(long long)size,(double)_size/1024/1024);
	return result;
}

void heap::free(heap_alloc& alloc)
{
	// NOTE: it is very important that this function only free the private heap
	// allocation, and NOT release any block memory to the device via _balloc->free_block.
	// This is because heap::free is called when an smat instance is destructed:
	//     {
	//         smat A(3,3);     // smat::smat calls heap::alloc, which may allocate a new block AND a slot within that block
	//         A *= 5;          // emit "mul A,5,A" instruction
	//     }                    // smat::~smat calls heap::free, which should free the slot but NOT free the block
	// Here the virtual machine executition "lags behind" and may not execute 
	// the corresponding "mul" instruction until long after the CPU has
	// finished destructing A (thereby calling heap::free). The memory must
	// still be owned by smat's process though, and not returned to the device/OS!
	// TODO: provide function to release all unused device memory without needing 
	//       to reset the entire backend.
	SM_ASSERT(alloc.addr)
	size_t base = (size_t)alloc.addr;
	size_t size = alloc.size;
	block_alloc* block = (block_alloc*)alloc.bookkeeping;  // Find the block that this allocation came from.
	SM_ASSERT(base >= block->base && base+size <= block->base+block->size);

	// Check if there's a free slot adjacent to the newly freed one.
	free_list& slots = block->slots;
	free_slot slot(base,size);
	auto right = slots.lower_bound(slot);
	auto left  = right;
	if (right != slots.end())
		SM_ASSERTMSG(base+size <= right->base,format("MemoryError: Attempt to free invalid device addr 0x%010llx. (already freed?)\n",(long long)base).c_str());
	if (right != slots.begin()) {
		--left;
		SM_ASSERTMSG(base >= left->base+left->size,format("MemoryError: Attempt to free invalid device addr 0x%010llx. (already freed?)\n",(long long)base).c_str());
	}
	bool coalesce_right = (right != slots.end() && right->base == base+size);
	bool coalesce_left  = (left  != right && left->base+left->size == base);

	if (!coalesce_left && !coalesce_right) {
		// Just insert the new free slot, since there is a gap 
		// between the new slot and the adjacent left/right slots
		slots.insert(right,slot);
	} else if (coalesce_left && coalesce_right) {
		// The new slot abutts both the left and right free slots,
		// so coalesce into a single large slot.
		slot.base = left->base;
		slot.size = left->size+size+right->size;
		slots.erase(left);
		slots.insert(right,slot);
		slots.erase(right);
	} else if (coalesce_left) {
		// Only coalesce with the left slot
		slot.base = left->base;
		slot.size = left->size+size;
		slots.erase(left);
		slots.insert(right,slot);
	} else {
		// Only coalesce with the right slot
		slot.size = size+right->size;
		slots.insert(right,slot);
		slots.erase(right);
	}

	_size -= size;
	_alloc_count--;
	SM_LOG("heap","free[0x%010llx]   0x%010llx bytes  \t(%.2f MB used)",base,(long long)size,(double)_size/1024/1024);
}

bool heap::alloc_from_block(block_alloc& block, size_t& size, heap_alloc& result)
{
	// A function to search a specific block for an allocation, returning true if 'result' was set
	for (auto slot = block.slots.begin(); slot != block.slots.end(); ++slot) {
		SM_DBASSERT(slot->base + slot->size <= block.base + block.size);
		// If the free slot is big enough, then we're done searching.
		if (size <= slot->size) {
			result = heap_alloc(slot->base,size,(size_t)&block);
			free_slot newslot(slot->base + size, slot->size - size);
			auto hint = block.slots.erase(slot);
			if (newslot.size > 0)
				block.slots.insert(hint,newslot);
			return true;
		}
	}
	return false;
}

heap_status heap::status() const
{
	heap_status status;
	status.host_total = get_system_memory_total();
	status.host_avail = get_system_memory_avail();
	status.host_used  = get_process_memory_used();
	status.device_total = _balloc->get_total_memory();
	status.device_avail = _balloc->get_avail_memory();
	status.device_used  = _size;
	status.device_committed  = _capacity;
	return status;
}

////////////////////////////////////////////////////////////////////

SM_NAMESPACE_END
