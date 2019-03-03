#ifndef HPC_SIMD_ALIGNED_ALLOCATOR_H
#define HPC_SIMD_ALIGNED_ALLOCATOR_H

#include <xmmintrin.h>
#include <cstdint>
#include <stdexcept>

using namespace std;

namespace flux {

template <typename T, std::size_t Alignment>
class SimdAlignedAllocator {
public:
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef ptrdiff_t difference_type;

    T* address(T& r) const { return &r; }

    const T* address(const T& s) const { return &s; }

    std::size_t max_size() const {
        // The following has been carefully written to be independent of
        // the definition of size_t and to avoid signed/unsigned warnings
        return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
    }

    // The following must be the same for all allocators.
    template <typename U>
    struct rebind {
        typedef SimdAlignedAllocator<U, Alignment> other;
    };

    bool operator!=(const SimdAlignedAllocator& other) const { return !(*this == other); }

    void construct(T* const p, const T& t) const {
        void* const pv = static_cast<void*>(p);

        new (pv) T(t);
    }

    void destroy(T* const p) const { p->~T(); }

    // Returns true if and only if storage allocated from *this can be deallocated from other, and vice versa.
    // Always returns true for stateless allocators.
    bool operator==(const SimdAlignedAllocator& other) const { return true; }

    // Default constructor, copy constructor, rebinding constructor, and destructor.
    // Empty for stateless allocators.
    SimdAlignedAllocator() {}

    SimdAlignedAllocator(const SimdAlignedAllocator&) {}

    template <typename U>
    SimdAlignedAllocator(const SimdAlignedAllocator<U, Alignment>&) {}

    ~SimdAlignedAllocator() {}

    // The following will be different for each allocator.
    T* allocate(const std::size_t n) const {
        if (n == 0) {
            return nullptr;
        }

        if (n > max_size()) {
            throw std::length_error("SimdAlignedAllocator<T>::allocate() - Integer overflow.");
        }

        void* const pv = _mm_malloc(n * sizeof(T), Alignment);

        // Allocators should throw std::bad_alloc in the case of memory allocation failure.
        if (pv == nullptr) {
            throw std::bad_alloc();
        }

        return static_cast<T*>(pv);
    }

    void deallocate(T* const p, const std::size_t n) const { _mm_free(p); }

    // The following will be the same for all allocators that ignore hints.
    template <typename U>
    T* allocate(const std::size_t n, const U* /* const hint */) const {
        return allocate(n);
    }

private:
    // Allocators are not required to be assignable
    SimdAlignedAllocator& operator=(const SimdAlignedAllocator&);
};
}  // namespace flux

#endif
