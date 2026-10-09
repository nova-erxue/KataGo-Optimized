#ifndef CORE_SIMPLEALLOCATOR_H
#define CORE_SIMPLEALLOCATOR_H

#include "../core/global.h"

template<typename T>
struct SizedBuf;

template<typename T>
class SimpleAllocator {
  std::function<T(size_t)> allocateFunc;
  std::function<void(T)> releaseFunc;

  std::map<size_t,std::vector<T>> buffers;

public:
  SimpleAllocator(std::function<T(size_t)> allocateFunc_, std::function<void(T)> releaseFunc_)
    :allocateFunc(allocateFunc_),releaseFunc(releaseFunc_),buffers()
  {
  }
  ~SimpleAllocator() {
    for(auto& iter: buffers) {
      for(T& buf: iter.second) {
        releaseFunc(buf);
      }
    }
  }

  SimpleAllocator() = delete;
  SimpleAllocator(const SimpleAllocator&) = delete;
  SimpleAllocator& operator=(const SimpleAllocator&) = delete;

  friend struct SizedBuf<T>;
};


template<typename T>
struct SizedBuf {
  size_t size;
  T buf;
  SimpleAllocator<T>* allocator;

  SizedBuf(SimpleAllocator<T>* alloc, size_t s)
    : size(s),buf(),allocator(alloc)
  {
    // Borrow the free list in place. Copying it allocates host memory on every
    // scratch-buffer checkout, even when no new device allocation is needed.
    std::vector<T>& buffers = allocator->buffers[size];
    if(buffers.empty())
      buf = allocator->allocateFunc(size);
    else {
      buf = buffers.back();
      buffers.pop_back();
    }
  }
  ~SizedBuf() {
    allocator->buffers[size].push_back(buf);
  }

  SizedBuf() = delete;
  SizedBuf(const SizedBuf&) = delete;
  SizedBuf& operator=(const SizedBuf&) = delete;

  friend class SimpleAllocator<T>;
};


#endif // CORE_SIMPLEALLOCATOR_H
