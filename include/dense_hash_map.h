#ifndef DENSE_HASH_MAP_H
#define DENSE_HASH_MAP_H

/**
 * this comes from google dense_hash_map/dense_hash_set
 * exclude io operation which we don't need
 * this implementation use Quadratic collision resolution which have great cache locality
 *
 * usage:
 * 1. must call set_empty_key() immediately after construction
 * 2. must call set_deleted_key() if you want to use erase(), the deleted and empty keys must differ.
 * 3. to force the memory to be freed, call resize(0)
 * 4. setting the minimum load factor to 0.0 guarantees that the hash table will never shrink
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace flux {

template <class T>
class libc_allocator_with_realloc {
public:
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;

    libc_allocator_with_realloc() {}
    libc_allocator_with_realloc(const libc_allocator_with_realloc&) {}
    ~libc_allocator_with_realloc() {}

    pointer address(reference r) const { return &r; }
    const_pointer address(const_reference r) const { return &r; }

    pointer allocate(size_type n, const_pointer = 0) { return static_cast<pointer>(malloc(n * sizeof(value_type))); }
    void deallocate(pointer p, size_type) { free(p); }
    pointer reallocate(pointer p, size_type n) { return static_cast<pointer>(realloc(p, n * sizeof(value_type))); }

    size_type max_size() const { return static_cast<size_type>(-1) / sizeof(value_type); }

    void construct(pointer p, const value_type& val) { new (p) value_type(val); }
    void destroy(pointer p) { p->~value_type(); }

    template <class U>
    libc_allocator_with_realloc(const libc_allocator_with_realloc<U>&) {}

    template <class U>
    struct rebind {
        typedef libc_allocator_with_realloc<U> other;
    };
};

// libc_allocator_with_realloc<void> specialization.
template <>
class libc_allocator_with_realloc<void> {
public:
    typedef void value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef void* pointer;
    typedef const void* const_pointer;

    template <class U>
    struct rebind {
        typedef libc_allocator_with_realloc<U> other;
    };
};

template <class T>
inline bool operator==(const libc_allocator_with_realloc<T>&, const libc_allocator_with_realloc<T>&) {
    return true;
}

template <class T>
inline bool operator!=(const libc_allocator_with_realloc<T>&, const libc_allocator_with_realloc<T>&) {
    return false;
}

/**
 * Settings contains parameters for growing and shrinking the table.
 *
 * the default hash of pointers is the identity hash, so probably all the low bits are 0.
 * We identify when we think we're hashing a pointer, and chop off the low bits.
 */
template <typename Key, typename HashFunc, typename SizeType, int HT_MIN_BUCKETS>
class sh_hashtable_settings : public HashFunc {
public:
    typedef Key key_type;
    typedef HashFunc hasher;
    typedef SizeType size_type;

public:
    sh_hashtable_settings(const hasher& hf, const float ht_occupancy_flt, const float ht_empty_flt)
        : hasher(hf),
          enlarge_threshold_(0),
          shrink_threshold_(0),
          consider_shrink_(false),
          use_empty_(false),
          use_deleted_(false),
          num_ht_copies_(0) {
        set_enlarge_factor(ht_occupancy_flt);
        set_shrink_factor(ht_empty_flt);
    }

    size_type hash(const key_type& v) const {
        // We munge the hash value when we don't trust hasher::operator().
        return hash_munger<Key>::MungedHash(hasher::operator()(v));
    }

    float enlarge_factor() const { return enlarge_factor_; }
    void set_enlarge_factor(float f) { enlarge_factor_ = f; }
    float shrink_factor() const { return shrink_factor_; }
    void set_shrink_factor(float f) { shrink_factor_ = f; }

    size_type enlarge_threshold() const { return enlarge_threshold_; }
    void set_enlarge_threshold(size_type t) { enlarge_threshold_ = t; }
    size_type shrink_threshold() const { return shrink_threshold_; }
    void set_shrink_threshold(size_type t) { shrink_threshold_ = t; }

    size_type enlarge_size(size_type x) const { return static_cast<size_type>(x * enlarge_factor_); }
    size_type shrink_size(size_type x) const { return static_cast<size_type>(x * shrink_factor_); }

    bool consider_shrink() const { return consider_shrink_; }
    void set_consider_shrink(bool t) { consider_shrink_ = t; }

    bool use_empty() const { return use_empty_; }
    void set_use_empty(bool t) { use_empty_ = t; }

    bool use_deleted() const { return use_deleted_; }
    void set_use_deleted(bool t) { use_deleted_ = t; }

    size_type num_ht_copies() const { return static_cast<size_type>(num_ht_copies_); }
    void inc_num_ht_copies() { ++num_ht_copies_; }

    // Reset the enlarge and shrink thresholds
    void reset_thresholds(size_type num_buckets) {
        set_enlarge_threshold(enlarge_size(num_buckets));
        set_shrink_threshold(shrink_size(num_buckets));
        // whatever caused us to reset already considered
        set_consider_shrink(false);
    }

    // caller is responsible for calling reset_threshold right after set_resizing_parameters.
    void set_resizing_parameters(float shrink, float grow) {
        assert(shrink >= 0.0);
        assert(grow <= 1.0);
        if (shrink > grow / 2.0f) shrink = grow / 2.0f;  // otherwise we thrash hash table size
        set_shrink_factor(shrink);
        set_enlarge_factor(grow);
    }

    // This is the smallest size a hash table can be without being too crowded
    // If you like, you can give a min #buckets as well as a min #elts
    size_type min_buckets(size_type num_elts, size_type min_buckets_wanted) {
        float enlarge = enlarge_factor();
        size_type sz = HT_MIN_BUCKETS;  // min buckets allowed
        while (sz < min_buckets_wanted || num_elts >= static_cast<size_type>(sz * enlarge)) {
            // this just prevents overflowing size_type, since sz can exceed max_size() here.
            if (static_cast<size_type>(sz * 2) < sz) {
                throw std::length_error("resize overflow");  // protect against overflow
            }
            sz *= 2;
        }
        return sz;
    }

private:
    template <class HashKey>
    class hash_munger {
    public:
        static size_t MungedHash(size_t hash) { return hash; }
    };
    // This matches when the hash table key is a pointer.
    template <class HashKey>
    class hash_munger<HashKey*> {
    public:
        static size_t MungedHash(size_t hash) {
            // this matters if we ever change sparse/dense_hash_* to compare hashes before comparing actual values.
            return hash / sizeof(void*);  // get rid of known-0 bits
        }
    };

    size_type enlarge_threshold_;  // table.size() * enlarge_factor
    size_type shrink_threshold_;   // table.size() * shrink_factor
    float enlarge_factor_;         // how full before resize
    float shrink_factor_;          // how empty before resize
    // consider_shrink=true if we should try to shrink before next insert
    bool consider_shrink_;
    bool use_empty_;    // used only by densehashtable, not sparsehashtable
    bool use_deleted_;  // false until del_key has been set
    // num_ht_copies is a counter incremented every Copy/Move
    unsigned int num_ht_copies_;
};

// The probing method
// Linear probing
// #define JUMP_(key, num_probes)    ( 1 )
// Quadratic probing
#define JUMP_(key, num_probes) (num_probes)

// Value: what is stored in the table (each bucket is a Value).
// Key: something in a 1-to-1 correspondence to a Value, that can be used
//      to search for a Value in the table (find() takes a Key).
// HashFcn: Takes a Key and returns an integer, the more unique the better.
// ExtractKey: given a Value, returns the unique Key associated with it.
//             Must inherit from unary_function, or at least have a
//             result_type enum indicating the return type of operator().
// SetKey: given a Value* and a Key, modifies the value such that
//         ExtractKey(value) == key.  We guarantee this is only called
//         with key == deleted_key or key == empty_key.
// EqualKey: Given two Keys, says whether they are the same (that is,
//           if they are both associated with the same Value).
// Alloc: STL allocator to use to allocate memory.

template <class Value, class Key, class HashFcn, class ExtractKey, class SetKey, class EqualKey, class Alloc>
class dense_hashtable;

template <class V, class K, class HF, class ExK, class SetK, class EqK, class A>
struct dense_hashtable_iterator;

template <class V, class K, class HF, class ExK, class SetK, class EqK, class A>
struct dense_hashtable_const_iterator;

// We're just an array, but we need to skip over empty and deleted elements
template <class V, class K, class HF, class ExK, class SetK, class EqK, class A>
struct dense_hashtable_iterator {
private:
    using value_alloc_type = typename std::allocator_traits<A>::template rebind_alloc<V>;

public:
    typedef dense_hashtable_iterator<V, K, HF, ExK, SetK, EqK, A> iterator;
    typedef dense_hashtable_const_iterator<V, K, HF, ExK, SetK, EqK, A> const_iterator;

    typedef std::forward_iterator_tag iterator_category;  // very little defined!
    typedef V value_type;
    typedef typename value_alloc_type::difference_type difference_type;
    typedef typename value_alloc_type::size_type size_type;
    typedef typename value_alloc_type::reference reference;
    typedef typename value_alloc_type::pointer pointer;

    // "Real" constructor and default constructor
    dense_hashtable_iterator(const dense_hashtable<V, K, HF, ExK, SetK, EqK, A>* h, pointer it, pointer it_end,
                             bool advance)
        : ht(h), pos(it), end(it_end) {
        if (advance) advance_past_empty_and_deleted();
    }
    dense_hashtable_iterator() {}
    // The default destructor is fine; we don't define one
    // The default operator= is fine; we don't define one

    // Happy dereferencer
    reference operator*() const { return *pos; }
    pointer operator->() const { return &(operator*()); }

    // Arithmetic.  The only hard part is making sure that
    // we're not on an empty or marked-deleted array element
    void advance_past_empty_and_deleted() {
        while (pos != end && (ht->test_empty(*this) || ht->test_deleted(*this))) ++pos;
    }
    iterator& operator++() {
        assert(pos != end);
        ++pos;
        advance_past_empty_and_deleted();
        return *this;
    }
    iterator operator++(int) {
        iterator tmp(*this);
        ++*this;
        return tmp;
    }

    // Comparison.
    bool operator==(const iterator& it) const { return pos == it.pos; }
    bool operator!=(const iterator& it) const { return pos != it.pos; }

    // The actual data
    const dense_hashtable<V, K, HF, ExK, SetK, EqK, A>* ht;
    pointer pos, end;
};

// Now do it all again, but with const-ness!
template <class V, class K, class HF, class ExK, class SetK, class EqK, class A>
struct dense_hashtable_const_iterator {
private:
    using value_alloc_type = typename std::allocator_traits<A>::template rebind_alloc<V>;

public:
    typedef dense_hashtable_iterator<V, K, HF, ExK, SetK, EqK, A> iterator;
    typedef dense_hashtable_const_iterator<V, K, HF, ExK, SetK, EqK, A> const_iterator;

    typedef std::forward_iterator_tag iterator_category;  // very little defined!
    typedef V value_type;
    typedef typename value_alloc_type::difference_type difference_type;
    typedef typename value_alloc_type::size_type size_type;
    typedef typename value_alloc_type::const_reference reference;
    typedef typename value_alloc_type::const_pointer pointer;

    // "Real" constructor and default constructor
    dense_hashtable_const_iterator(const dense_hashtable<V, K, HF, ExK, SetK, EqK, A>* h, pointer it, pointer it_end,
                                   bool advance)
        : ht(h), pos(it), end(it_end) {
        if (advance) advance_past_empty_and_deleted();
    }
    dense_hashtable_const_iterator() : ht(NULL), pos(pointer()), end(pointer()) {}
    // This lets us convert regular iterators to const iterators
    dense_hashtable_const_iterator(const iterator& it) : ht(it.ht), pos(it.pos), end(it.end) {}
    // The default destructor is fine; we don't define one
    // The default operator= is fine; we don't define one

    // Happy dereferencer
    reference operator*() const { return *pos; }
    pointer operator->() const { return &(operator*()); }

    // Arithmetic.  The only hard part is making sure that
    // we're not on an empty or marked-deleted array element
    void advance_past_empty_and_deleted() {
        while (pos != end && (ht->test_empty(*this) || ht->test_deleted(*this))) ++pos;
    }
    const_iterator& operator++() {
        assert(pos != end);
        ++pos;
        advance_past_empty_and_deleted();
        return *this;
    }
    const_iterator operator++(int) {
        const_iterator tmp(*this);
        ++*this;
        return tmp;
    }

    // Comparison.
    bool operator==(const const_iterator& it) const { return pos == it.pos; }
    bool operator!=(const const_iterator& it) const { return pos != it.pos; }

    // The actual data
    const dense_hashtable<V, K, HF, ExK, SetK, EqK, A>* ht;
    pointer pos, end;
};

template <class Value, class Key, class HashFcn, class ExtractKey, class SetKey, class EqualKey, class Alloc>
class dense_hashtable {
private:
    using value_alloc_type = typename std::allocator_traits<Alloc>::template rebind_alloc<Value>;

public:
    typedef Key key_type;
    typedef Value value_type;
    typedef HashFcn hasher;
    typedef EqualKey key_equal;
    typedef Alloc allocator_type;

    typedef typename value_alloc_type::size_type size_type;
    typedef typename value_alloc_type::difference_type difference_type;
    typedef typename value_alloc_type::reference reference;
    typedef typename value_alloc_type::const_reference const_reference;
    typedef typename value_alloc_type::pointer pointer;
    typedef typename value_alloc_type::const_pointer const_pointer;
    typedef dense_hashtable_iterator<Value, Key, HashFcn, ExtractKey, SetKey, EqualKey, Alloc> iterator;

    typedef dense_hashtable_const_iterator<Value, Key, HashFcn, ExtractKey, SetKey, EqualKey, Alloc> const_iterator;

    // These come from tr1.  For us they're the same as regular iterators.
    typedef iterator local_iterator;
    typedef const_iterator const_local_iterator;

    // How full we let the table get before we resize, by default.
    // Knuth says .8 is good -- higher causes us to probe too much,
    // though it saves memory.
    static const int HT_OCCUPANCY_PCT;  // defined at the bottom of this file

    // How empty we let the table get before we resize lower, by default.
    // (0.0 means never resize lower.)
    // It should be less than OCCUPANCY_PCT / 2 or we thrash resizing
    static const int HT_EMPTY_PCT;  // defined at the bottom of this file

    // Minimum size we're willing to let hashtables be.
    // Must be a power of two, and at least 4.
    // Note, however, that for a given hashtable, the initial size is a
    // function of the first constructor arg, and may be >HT_MIN_BUCKETS.
    static const size_type HT_MIN_BUCKETS = 4;

    // By default, if you don't specify a hashtable size at
    // construction-time, we use this size.  Must be a power of two, and
    // at least HT_MIN_BUCKETS.
    static const size_type HT_DEFAULT_STARTING_BUCKETS = 32;

    // ITERATOR FUNCTIONS
    iterator begin() { return iterator(this, table, table + num_buckets, true); }
    iterator end() { return iterator(this, table + num_buckets, table + num_buckets, true); }
    const_iterator begin() const { return const_iterator(this, table, table + num_buckets, true); }
    const_iterator end() const { return const_iterator(this, table + num_buckets, table + num_buckets, true); }

    // These come from tr1 unordered_map.  They iterate over 'bucket' n.
    // We'll just consider bucket n to be the n-th element of the table.
    local_iterator begin(size_type i) { return local_iterator(this, table + i, table + i + 1, false); }
    local_iterator end(size_type i) {
        local_iterator it = begin(i);
        if (!test_empty(i) && !test_deleted(i)) ++it;
        return it;
    }
    const_local_iterator begin(size_type i) const {
        return const_local_iterator(this, table + i, table + i + 1, false);
    }
    const_local_iterator end(size_type i) const {
        const_local_iterator it = begin(i);
        if (!test_empty(i) && !test_deleted(i)) ++it;
        return it;
    }

    // ACCESSOR FUNCTIONS for the things we templatize on, basically
    hasher hash_funct() const { return settings; }
    key_equal key_eq() const { return key_info; }
    allocator_type get_allocator() const { return allocator_type(val_info); }

    // Accessor function for statistics gathering.
    int num_table_copies() const { return settings.num_ht_copies(); }

private:
    // Annoyingly, we can't copy values around, because they might have
    // const components (they're probably pair<const X, Y>).  We use
    // explicit destructor invocation and placement new to get around
    // this.  Arg.
    template <typename... Args>
    void set_value(pointer dst, Args&&... args) {
        dst->~value_type();  // delete the old value, if any
        new (dst) value_type(std::forward<Args>(args)...);
    }

    void destroy_buckets(size_type first, size_type last) {
        for (; first != last; ++first) table[first].~value_type();
    }

    // DELETE HELPER FUNCTIONS
    // This lets the user describe a key that will indicate deleted
    // table entries.  This key should be an "impossible" entry --
    // if you try to insert it for real, you won't be able to retrieve it!
    // (NB: while you pass in an entire value, only the key part is looked
    // at.  This is just because I don't know how to assign just a key.)
private:
    void squash_deleted() {  // gets rid of any deleted entries we have
        if (num_deleted) {   // get rid of deleted before writing
            size_type resize_to = settings.min_buckets(num_elements, bucket_count());
            dense_hashtable tmp(std::move(*this), resize_to);  // copying will get rid of deleted
            swap(tmp);                                         // now we are tmp
        }
        assert(num_deleted == 0);
    }

    // Test if the given key is the deleted indicator.  Requires
    // num_deleted > 0, for correctness of read(), and because that
    // guarantees that key_info.del_key is valid.
    bool test_deleted_key(const key_type& key) const {
        assert(num_deleted > 0);
        return equals(key_info.del_key, key);
    }

public:
    void set_deleted_key(const key_type& key) {
        // the empty indicator (if specified) and the deleted indicator
        // must be different
        assert((!settings.use_empty() || !equals(key, key_info.empty_key)) &&
               "Passed the empty-key to set_deleted_key");
        // It's only safe to change what "deleted" means if we purge deleted guys
        squash_deleted();
        settings.set_use_deleted(true);
        key_info.del_key = key;
    }
    void clear_deleted_key() {
        squash_deleted();
        settings.set_use_deleted(false);
    }
    key_type deleted_key() const {
        assert(settings.use_deleted() && "Must set deleted key before calling deleted_key");
        return key_info.del_key;
    }

    // These are public so the iterators can use them
    // True if the item at position bucknum is "deleted" marker
    bool test_deleted(size_type bucknum) const {
        // Invariant: !use_deleted() implies num_deleted is 0.
        assert(settings.use_deleted() || num_deleted == 0);
        return num_deleted > 0 && test_deleted_key(get_key(table[bucknum]));
    }
    bool test_deleted(const iterator& it) const {
        // Invariant: !use_deleted() implies num_deleted is 0.
        assert(settings.use_deleted() || num_deleted == 0);
        return num_deleted > 0 && test_deleted_key(get_key(*it));
    }
    bool test_deleted(const const_iterator& it) const {
        // Invariant: !use_deleted() implies num_deleted is 0.
        assert(settings.use_deleted() || num_deleted == 0);
        return num_deleted > 0 && test_deleted_key(get_key(*it));
    }

private:
    void check_use_deleted(const char* caller) {
        (void)caller;  // could log it if the assert failed
        assert(settings.use_deleted());
    }

    // Set it so test_deleted is true.  true if object didn't used to be deleted.
    bool set_deleted(iterator& it) {
        check_use_deleted("set_deleted()");
        bool retval = !test_deleted(it);
        // &* converts from iterator to value-type.
        set_key(&(*it), key_info.del_key);
        return retval;
    }
    // Set it so test_deleted is false.  true if object used to be deleted.
    bool clear_deleted(iterator& it) {
        check_use_deleted("clear_deleted()");
        // Happens automatically when we assign something else in its place.
        return test_deleted(it);
    }

    // We also allow to set/clear the deleted bit on a const iterator.
    // We allow a const_iterator for the same reason you can delete a
    // const pointer: it's convenient, and semantically you can't use
    // 'it' after it's been deleted anyway, so its const-ness doesn't
    // really matter.
    bool set_deleted(const_iterator& it) {
        check_use_deleted("set_deleted()");
        bool retval = !test_deleted(it);
        set_key(const_cast<pointer>(&(*it)), key_info.del_key);
        return retval;
    }
    // Set it so test_deleted is false.  true if object used to be deleted.
    bool clear_deleted(const_iterator& it) {
        check_use_deleted("clear_deleted()");
        return test_deleted(it);
    }

    // EMPTY HELPER FUNCTIONS
    // This lets the user describe a key that will indicate empty (unused)
    // table entries.  This key should be an "impossible" entry --
    // if you try to insert it for real, you won't be able to retrieve it!
    // (NB: while you pass in an entire value, only the key part is looked
    // at.  This is just because I don't know how to assign just a key.)
public:
    // These are public so the iterators can use them
    // True if the item at position bucknum is "empty" marker
    bool test_empty(size_type bucknum) const {
        assert(settings.use_empty());  // we always need to know what's empty!
        return equals(key_info.empty_key, get_key(table[bucknum]));
    }
    bool test_empty(const iterator& it) const {
        assert(settings.use_empty());  // we always need to know what's empty!
        return equals(key_info.empty_key, get_key(*it));
    }
    bool test_empty(const const_iterator& it) const {
        assert(settings.use_empty());  // we always need to know what's empty!
        return equals(key_info.empty_key, get_key(*it));
    }

private:
    void fill_range_with_empty(pointer table_start, size_type count) {
        for (size_type i = 0; i < count; ++i) {
            construct_key(&table_start[i], key_info.empty_key);
        }
    }

public:
    void set_empty_key(const key_type& key) {
        // Once you set the empty key, you can't change it
        assert(!settings.use_empty() && "Calling set_empty_key multiple times");
        // The deleted indicator (if specified) and the empty indicator
        // must be different.
        assert((!settings.use_deleted() || !equals(key, key_info.del_key)) &&
               "Setting the empty key the same as the deleted key");
        settings.set_use_empty(true);
        key_info.empty_key = key;

        assert(!table);  // must set before first use
        // num_buckets was set in constructor even though table was NULL
        table = val_info.allocate(num_buckets);
        assert(table);
        fill_range_with_empty(table, num_buckets);
    }
    key_type empty_key() const {
        assert(settings.use_empty());
        return key_info.empty_key;
    }

    // FUNCTIONS CONCERNING SIZE
public:
    size_type size() const { return num_elements - num_deleted; }
    size_type max_size() const { return val_info.max_size(); }
    bool empty() const { return size() == 0; }
    size_type bucket_count() const { return num_buckets; }
    size_type max_bucket_count() const { return max_size(); }
    size_type nonempty_bucket_count() const { return num_elements; }
    // These are tr1 methods.  Their idea of 'bucket' doesn't map well to
    // what we do.  We just say every bucket has 0 or 1 items in it.
    size_type bucket_size(size_type i) const { return begin(i) == end(i) ? 0 : 1; }

private:
    // Because of the above, size_type(-1) is never legal; use it for errors
    static const size_type ILLEGAL_BUCKET = size_type(-1);

    // Used after a string of deletes.  Returns true if we actually shrunk.
    bool maybe_shrink() {
        assert(num_elements >= num_deleted);
        assert((bucket_count() & (bucket_count() - 1)) == 0);  // is a power of two
        assert(bucket_count() >= HT_MIN_BUCKETS);
        bool retval = false;

        // If you construct a hashtable with < HT_DEFAULT_STARTING_BUCKETS,
        // we'll never shrink until you get relatively big, and we'll never
        // shrink below HT_DEFAULT_STARTING_BUCKETS.  Otherwise, something
        // like "dense_hash_set<int> x; x.insert(4); x.erase(4);" will
        // shrink us down to HT_MIN_BUCKETS buckets, which is too small.
        const size_type num_remain = num_elements - num_deleted;
        const size_type shrink_threshold = settings.shrink_threshold();
        if (shrink_threshold > 0 && num_remain < shrink_threshold && bucket_count() > HT_DEFAULT_STARTING_BUCKETS) {
            const float shrink_factor = settings.shrink_factor();
            size_type sz = bucket_count() / 2;  // find how much we should shrink
            while (sz > HT_DEFAULT_STARTING_BUCKETS && num_remain < sz * shrink_factor) {
                sz /= 2;  // stay a power of 2
            }
            dense_hashtable tmp(std::move(*this), sz);  // Do the actual resizing
            swap(tmp);                                  // now we are tmp
            retval = true;
        }
        settings.set_consider_shrink(false);  // because we just considered it
        return retval;
    }

    // We'll let you resize a hashtable -- though this makes us copy all!
    // When you resize, you say, "make it big enough for this many more elements"
    // Returns true if we actually resized, false if size was already ok.
    bool resize_delta(size_type delta) {
        bool did_resize = false;
        if (settings.consider_shrink()) {  // see if lots of deletes happened
            if (maybe_shrink()) did_resize = true;
        }
        if (num_elements >= (std::numeric_limits<size_type>::max)() - delta) {
            throw std::length_error("resize overflow");
        }
        if (bucket_count() >= HT_MIN_BUCKETS && (num_elements + delta) <= settings.enlarge_threshold())
            return did_resize;  // we're ok as we are

        // Sometimes, we need to resize just to get rid of all the
        // "deleted" buckets that are clogging up the hashtable.  So when
        // deciding whether to resize, count the deleted buckets (which
        // are currently taking up room).  But later, when we decide what
        // size to resize to, *don't* count deleted buckets, since they
        // get discarded during the resize.
        size_type needed_size = settings.min_buckets(num_elements + delta, 0);
        if (needed_size <= bucket_count())  // we have enough buckets
            return did_resize;

        size_type resize_to = settings.min_buckets(num_elements - num_deleted + delta, bucket_count());

        // When num_deleted is large, we may still grow but we do not want to
        // over expand.  So we reduce needed_size by a portion of num_deleted
        // (the exact portion does not matter).  This is especially helpful
        // when min_load_factor is zero (no shrink at all) to avoid doubling
        // the bucket count to infinity.  See also test ResizeWithoutShrink.
        needed_size = settings.min_buckets(num_elements - num_deleted / 4 + delta, 0);

        if (resize_to < needed_size &&  // may double resize_to
            resize_to < (std::numeric_limits<size_type>::max)() / 2) {
            // This situation means that we have enough deleted elements,
            // that once we purge them, we won't actually have needed to
            // grow.  But we may want to grow anyway: if we just purge one
            // element, say, we'll have to grow anyway next time we
            // insert.  Might as well grow now, since we're already going
            // through the trouble of copying (in order to purge the
            // deleted elements).
            const size_type target = static_cast<size_type>(settings.shrink_size(resize_to * 2));
            if (num_elements - num_deleted + delta >= target) {
                // Good, we won't be below the shrink threshhold even if we double.
                resize_to *= 2;
            }
        }
        dense_hashtable tmp(std::move(*this), resize_to);
        swap(tmp);  // now we are tmp
        return true;
    }

    // We require table be not-NULL and empty before calling this.
    void resize_table(size_type /*old_size*/, size_type new_size, std::true_type) {
        table = val_info.realloc_or_die(table, new_size);
    }

    void resize_table(size_type old_size, size_type new_size, std::false_type) {
        val_info.deallocate(table, old_size);
        table = val_info.allocate(new_size);
    }

    // Used to actually do the rehashing when we grow/shrink a hashtable
    template <typename Hashtable>
    void copy_or_move_from(Hashtable&& ht, size_type min_buckets_wanted) {
        clear_to_size(settings.min_buckets(ht.size(), min_buckets_wanted));

        // We use a normal iterator to get non-deleted bcks from ht
        // We could use insert() here, but since we know there are
        // no duplicates and no deleted items, we can be more efficient
        assert((bucket_count() & (bucket_count() - 1)) == 0);  // a power of two
        for (auto&& value : ht) {
            size_type num_probes = 0;  // how many times we've probed
            size_type bucknum;
            const size_type bucket_count_minus_one = bucket_count() - 1;
            for (bucknum = hash(get_key(value)) & bucket_count_minus_one; !test_empty(bucknum);  // not empty
                 bucknum = (bucknum + JUMP_(key, num_probes)) & bucket_count_minus_one) {
                ++num_probes;
                assert(num_probes < bucket_count() && "Hashtable is full: an error in key_equal<> or hash<>");
            }

            using will_move = std::is_rvalue_reference<Hashtable&&>;
            using value_t = typename std::conditional<will_move::value, value_type&&, const_reference>::type;

            set_value(&table[bucknum], std::forward<value_t>(value));
            num_elements++;
        }
        settings.inc_num_ht_copies();
    }

    // Required by the spec for hashed associative container
public:
    // Though the docs say this should be num_buckets, I think it's much
    // more useful as num_elements.  As a special feature, calling with
    // req_elements==0 will cause us to shrink if we can, saving space.
    void resize(size_type req_elements) {  // resize to this or larger
        if (settings.consider_shrink() || req_elements == 0) maybe_shrink();
        if (req_elements > num_elements) resize_delta(req_elements - num_elements);
    }

    // Get and change the value of shrink_factor and enlarge_factor.  The
    // description at the beginning of this file explains how to choose
    // the values.  Setting the shrink parameter to 0.0 ensures that the
    // table never shrinks.
    void get_resizing_parameters(float* shrink, float* grow) const {
        *shrink = settings.shrink_factor();
        *grow = settings.enlarge_factor();
    }
    void set_resizing_parameters(float shrink, float grow) {
        settings.set_resizing_parameters(shrink, grow);
        settings.reset_thresholds(bucket_count());
    }

    // CONSTRUCTORS -- as required by the specs, we take a size,
    // but also let you specify a hashfunction, key comparator,
    // and key extractor.  We also define a copy constructor and =.
    // DESTRUCTOR -- needs to free the table
    explicit dense_hashtable(size_type expected_max_items_in_table = 0, const HashFcn& hf = HashFcn(),
                             const EqualKey& eql = EqualKey(), const ExtractKey& ext = ExtractKey(),
                             const SetKey& set = SetKey(), const Alloc& alloc = Alloc())
        : settings(hf),
          key_info(ext, set, eql),
          num_deleted(0),
          num_elements(0),
          num_buckets(expected_max_items_in_table == 0 ? HT_DEFAULT_STARTING_BUCKETS
                                                       : settings.min_buckets(expected_max_items_in_table, 0)),
          val_info(alloc_impl<value_alloc_type>(alloc)),
          table(NULL) {
        // table is NULL until emptyval is set.  However, we set num_buckets
        // here so we know how much space to allocate once emptyval is set
        settings.reset_thresholds(bucket_count());
    }

    // As a convenience for resize(), we allow an optional second argument
    // which lets you make this new hashtable a different size than ht
    dense_hashtable(const dense_hashtable& ht, size_type min_buckets_wanted = HT_DEFAULT_STARTING_BUCKETS)
        : settings(ht.settings),
          key_info(ht.key_info),
          num_deleted(0),
          num_elements(0),
          num_buckets(0),
          val_info(ht.val_info),
          table(NULL) {
        if (!ht.settings.use_empty()) {
            // If use_empty isn't set, copy_from will crash, so we do our own copying.
            assert(ht.empty());
            num_buckets = settings.min_buckets(ht.size(), min_buckets_wanted);
            settings.reset_thresholds(bucket_count());
            return;
        }
        settings.reset_thresholds(bucket_count());
        copy_or_move_from(ht, min_buckets_wanted);  // copy_or_move_from() ignores deleted entries
    }

    dense_hashtable(dense_hashtable&& ht) : dense_hashtable() { swap(ht); }

    dense_hashtable(dense_hashtable&& ht, size_type min_buckets_wanted)
        : settings(ht.settings),
          key_info(ht.key_info),
          num_deleted(0),
          num_elements(0),
          num_buckets(0),
          val_info(std::move(ht.val_info)),
          table(NULL) {
        if (!ht.settings.use_empty()) {
            // If use_empty isn't set, copy_or_move_from will crash, so we do our own copying.
            assert(ht.empty());
            num_buckets = settings.min_buckets(ht.size(), min_buckets_wanted);
            settings.reset_thresholds(bucket_count());
            return;
        }
        settings.reset_thresholds(bucket_count());
        copy_or_move_from(std::move(ht), min_buckets_wanted);  // copy_or_move_from() ignores deleted entries
    }

    dense_hashtable& operator=(const dense_hashtable& ht) {
        if (&ht == this) return *this;  // don't copy onto ourselves
        if (!ht.settings.use_empty()) {
            assert(ht.empty());
            dense_hashtable empty_table(ht);  // empty table with ht's thresholds
            this->swap(empty_table);
            return *this;
        }
        settings = ht.settings;
        key_info = ht.key_info;
        // copy_or_move_from() calls clear and sets num_deleted to 0 too
        copy_or_move_from(ht, HT_MIN_BUCKETS);
        // we purposefully don't copy the allocator, which may not be copyable
        return *this;
    }

    dense_hashtable& operator=(dense_hashtable&& ht) {
        assert(&ht != this);  // this should not happen
        swap(ht);
        return *this;
    }

    ~dense_hashtable() {
        if (table) {
            destroy_buckets(0, num_buckets);
            val_info.deallocate(table, num_buckets);
        }
    }

    // Many STL algorithms use swap instead of copy constructors
    void swap(dense_hashtable& ht) {
        std::swap(settings, ht.settings);
        std::swap(key_info, ht.key_info);
        std::swap(num_deleted, ht.num_deleted);
        std::swap(num_elements, ht.num_elements);
        std::swap(num_buckets, ht.num_buckets);
        std::swap(table, ht.table);
        settings.reset_thresholds(bucket_count());  // also resets consider_shrink
        ht.settings.reset_thresholds(ht.bucket_count());
        // purposefully don't swap the allocator, which may not be swap-able
    }

private:
    void clear_to_size(size_type new_num_buckets) {
        if (!table) {
            table = val_info.allocate(new_num_buckets);
        } else {
            destroy_buckets(0, num_buckets);
            if (new_num_buckets != num_buckets) {  // resize, if necessary
                typedef std::integral_constant<
                    bool, std::is_same<value_alloc_type, libc_allocator_with_realloc<value_type>>::value>
                    realloc_ok;
                resize_table(num_buckets, new_num_buckets, realloc_ok());
            }
        }
        assert(table);
        fill_range_with_empty(table, new_num_buckets);
        num_elements = 0;
        num_deleted = 0;
        num_buckets = new_num_buckets;  // our new size
        settings.reset_thresholds(bucket_count());
    }

public:
    // It's always nice to be able to clear a table without deallocating it
    void clear() {
        // If the table is already empty, and the number of buckets is
        // already as we desire, there's nothing to do.
        const size_type new_num_buckets = settings.min_buckets(0, 0);
        if (num_elements == 0 && new_num_buckets == num_buckets) {
            return;
        }
        clear_to_size(new_num_buckets);
    }

    // Clear the table without resizing it.
    // Mimicks the stl_hashtable's behaviour when clear()-ing in that it
    // does not modify the bucket count
    void clear_no_resize() {
        if (num_elements > 0) {
            assert(table);
            destroy_buckets(0, num_buckets);
            fill_range_with_empty(table, num_buckets);
        }
        // don't consider to shrink before another erase()
        settings.reset_thresholds(bucket_count());
        num_elements = 0;
        num_deleted = 0;
    }

private:
    /**
     * Returns a pair of positions: 1st where the object is, 2nd where it would go if you wanted to insert it
     * 1st is ILLEGAL_BUCKET if object is not found
     * 2nd is ILLEGAL_BUCKET if it is.
     *
     * Note: because of deletions where-to-insert is not trivial:
     * it's the first deleted bucket we see, as long as we don't find the key later
     */
    std::pair<size_type, size_type> find_position(const key_type& key) const {
        size_type num_probes = 0;  // how many times we've probed
        const size_type bucket_count_minus_one = bucket_count() - 1;
        size_type bucknum = hash(key) & bucket_count_minus_one;
        size_type insert_pos = ILLEGAL_BUCKET;     // where we would insert
        while (true) {                             // probe until something happens
            if (test_empty(bucknum)) {             // bucket is empty
                if (insert_pos == ILLEGAL_BUCKET)  // found no prior place to insert
                    return std::pair<size_type, size_type>(ILLEGAL_BUCKET, bucknum);
                else
                    return std::pair<size_type, size_type>(ILLEGAL_BUCKET, insert_pos);

            } else if (test_deleted(bucknum)) {  // keep searching, but mark to insert
                if (insert_pos == ILLEGAL_BUCKET) insert_pos = bucknum;
            } else if (equals(key, get_key(table[bucknum]))) {
                return std::pair<size_type, size_type>(bucknum, ILLEGAL_BUCKET);
            }
            ++num_probes;  // we're doing another probe
            bucknum = (bucknum + JUMP_(key, num_probes)) & bucket_count_minus_one;
            assert(num_probes < bucket_count() && "Hashtable is full: an error in key_equal<> or hash<>");
        }
    }

public:
    iterator find(const key_type& key) {
        if (size() == 0) return end();
        std::pair<size_type, size_type> pos = find_position(key);
        if (pos.first == ILLEGAL_BUCKET)  // alas, not there
            return end();
        else
            return iterator(this, table + pos.first, table + num_buckets, false);
    }

    const_iterator find(const key_type& key) const {
        if (size() == 0) return end();
        std::pair<size_type, size_type> pos = find_position(key);
        if (pos.first == ILLEGAL_BUCKET)  // alas, not there
            return end();
        else
            return const_iterator(this, table + pos.first, table + num_buckets, false);
    }

    // This is a tr1 method: the bucket a given key is in, or what bucket
    // it would be put in, if it were to be inserted.  Shrug.
    size_type bucket(const key_type& key) const {
        std::pair<size_type, size_type> pos = find_position(key);
        return pos.first == ILLEGAL_BUCKET ? pos.second : pos.first;
    }

    // Counts how many elements have key key.  For maps, it's either 0 or 1.
    size_type count(const key_type& key) const {
        std::pair<size_type, size_type> pos = find_position(key);
        return pos.first == ILLEGAL_BUCKET ? 0 : 1;
    }

    // Likewise, equal_range doesn't really make sense for us.  Oh well.
    std::pair<iterator, iterator> equal_range(const key_type& key) {
        iterator pos = find(key);  // either an iterator or end
        if (pos == end()) {
            return std::pair<iterator, iterator>(pos, pos);
        } else {
            const iterator startpos = pos++;
            return std::pair<iterator, iterator>(startpos, pos);
        }
    }
    std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const {
        const_iterator pos = find(key);  // either an iterator or end
        if (pos == end()) {
            return std::pair<const_iterator, const_iterator>(pos, pos);
        } else {
            const const_iterator startpos = pos++;
            return std::pair<const_iterator, const_iterator>(startpos, pos);
        }
    }

    // INSERTION ROUTINES
private:
    // Private method used by insert_noresize and find_or_insert.
    template <typename... Args>
    iterator insert_at(size_type pos, Args&&... args) {
        if (size() >= max_size()) {
            throw std::length_error("insert overflow");
        }
        if (test_deleted(pos)) {  // just replace if it's been del.
            // shrug: shouldn't need to be const.
            const_iterator delpos(this, table + pos, table + num_buckets, false);
            clear_deleted(delpos);
            assert(num_deleted > 0);
            --num_deleted;  // used to be, now it isn't
        } else {
            ++num_elements;  // replacing an empty bucket
        }
        set_value(&table[pos], std::forward<Args>(args)...);
        return iterator(this, table + pos, table + num_buckets, false);
    }

    // If you know *this is big enough to hold obj, use this routine
    template <typename K, typename... Args>
    std::pair<iterator, bool> insert_noresize(K&& key, Args&&... args) {
        // First, double-check we're not inserting del_key or emptyval
        assert(settings.use_empty() && "Inserting without empty key");
        assert(!equals(std::forward<K>(key), key_info.empty_key) && "Inserting the empty key");
        assert((!settings.use_deleted() || !equals(key, key_info.del_key)) && "Inserting the deleted key");

        const std::pair<size_type, size_type> pos = find_position(key);
        if (pos.first != ILLEGAL_BUCKET) {  // object was already there, false: we didn't insert
            return std::pair<iterator, bool>(iterator(this, table + pos.first, table + num_buckets, false), false);
        } else {  // pos.second says where to put it
            return std::pair<iterator, bool>(insert_at(pos.second, std::forward<Args>(args)...), true);
        }
    }

    // Specializations of insert(it, it) depending on the power of the iterator:
    // (1) Iterator supports operator-, resize before inserting
    template <class ForwardIterator>
    void insert(ForwardIterator f, ForwardIterator l, std::forward_iterator_tag) {
        size_t dist = std::distance(f, l);
        if (dist >= (std::numeric_limits<size_type>::max)()) {
            throw std::length_error("insert-range overflow");
        }
        resize_delta(static_cast<size_type>(dist));
        for (; dist > 0; --dist, ++f) {
            insert_noresize(get_key(*f), *f);
        }
    }

    // (2) Arbitrary iterator, can't tell how much to resize
    template <class InputIterator>
    void insert(InputIterator f, InputIterator l, std::input_iterator_tag) {
        for (; f != l; ++f) insert(*f);
    }

public:
    // This is the normal insert routine, used by the outside world
    template <typename Arg>
    std::pair<iterator, bool> insert(Arg&& obj) {
        resize_delta(1);  // adding an object, grow if need be
        return insert_noresize(get_key(std::forward<Arg>(obj)), std::forward<Arg>(obj));
    }

    template <typename K, typename... Args>
    std::pair<iterator, bool> emplace(K&& key, Args&&... args) {
        resize_delta(1);
        // here we push key twice as we need it once for the indexing, and the rest of the params are for the emplace
        // itself
        return insert_noresize(std::forward<K>(key), std::forward<K>(key), std::forward<Args>(args)...);
    }

    template <typename K, typename... Args>
    std::pair<iterator, bool> emplace_hint(const_iterator hint, K&& key, Args&&... args) {
        resize_delta(1);

        if (equals(key, hint->first)) {
            return {iterator(this, const_cast<pointer>(hint.pos), const_cast<pointer>(hint.end), false), false};
        }

        // here we push key twice as we need it once for the indexing, and the rest of the params are for the emplace
        // itself
        return insert_noresize(std::forward<K>(key), std::forward<K>(key), std::forward<Args>(args)...);
    }

    // When inserting a lot at a time, we specialize on the type of iterator
    template <class InputIterator>
    void insert(InputIterator f, InputIterator l) {
        // specializes on iterator type
        insert(f, l, typename std::iterator_traits<InputIterator>::iterator_category());
    }

    // DefaultValue is a functor that takes a key and returns a value_type
    // representing the default value to be inserted if none is found.
    template <class T, class K>
    value_type& find_or_insert(K&& key) {
        // First, double-check we're not inserting emptykey or del_key
        assert((!settings.use_empty() || !equals(key, key_info.empty_key)) && "Inserting the empty key");
        assert((!settings.use_deleted() || !equals(key, key_info.del_key)) && "Inserting the deleted key");
        const std::pair<size_type, size_type> pos = find_position(key);
        if (pos.first != ILLEGAL_BUCKET) {  // object was already there
            return table[pos.first];
        } else if (resize_delta(1)) {  // needed to rehash to make room
            // Since we resized, we can't use pos, so recalculate where to insert.
            return *insert_noresize(std::forward<K>(key), std::forward<K>(key), T()).first;
        } else {  // no need to rehash, insert right here
            return *insert_at(pos.second, std::forward<K>(key), T());
        }
    }

    // DELETION ROUTINES
    size_type erase(const key_type& key) {
        // First, double-check we're not trying to erase del_key or emptyval.
        assert((!settings.use_empty() || !equals(key, key_info.empty_key)) && "Erasing the empty key");
        assert((!settings.use_deleted() || !equals(key, key_info.del_key)) && "Erasing the deleted key");
        const_iterator pos = find(key);  // shrug: shouldn't need to be const
        if (pos != end()) {
            assert(!test_deleted(pos));  // or find() shouldn't have returned it
            set_deleted(pos);
            ++num_deleted;
            settings.set_consider_shrink(true);  // will think about shrink after next insert
            return 1;                            // because we deleted one thing
        } else {
            return 0;  // because we deleted nothing
        }
    }

    // We return the iterator past the deleted item.
    iterator erase(const_iterator pos) {
        if (pos == end()) return end();  // sanity check
        if (set_deleted(pos)) {          // true if object has been newly deleted
            ++num_deleted;
            settings.set_consider_shrink(true);  // will think about shrink after next insert
        }
        return iterator(this, const_cast<pointer>(pos.pos), const_cast<pointer>(pos.end), true);
    }

    iterator erase(const_iterator f, const_iterator l) {
        for (; f != l; ++f) {
            if (set_deleted(f))  // should always be true
                ++num_deleted;
        }
        settings.set_consider_shrink(true);  // will think about shrink after next insert
        return iterator(this, const_cast<pointer>(f.pos), const_cast<pointer>(f.end), false);
    }

    // COMPARISON
    bool operator==(const dense_hashtable& ht) const {
        if (size() != ht.size()) {
            return false;
        } else if (this == &ht) {
            return true;
        } else {
            // Iterate through the elements in "this" and see if the
            // corresponding element is in ht
            for (const_iterator it = begin(); it != end(); ++it) {
                const_iterator it2 = ht.find(get_key(*it));
                if ((it2 == ht.end()) || (*it != *it2)) {
                    return false;
                }
            }
            return true;
        }
    }
    bool operator!=(const dense_hashtable& ht) const { return !(*this == ht); }

private:
    template <class A>
    class alloc_impl : public A {
    public:
        typedef typename A::pointer pointer;
        typedef typename A::size_type size_type;

        // Convert a normal allocator to one that has realloc_or_die()
        alloc_impl(const A& a) : A(a) {}

        pointer realloc_or_die(pointer /*ptr*/, size_type /*n*/) {
            fprintf(stderr, "realloc_or_die is only supported for libc_allocator_with_realloc\n");
            exit(1);
            return NULL;
        }
    };

    // A template specialization of alloc_impl for libc_allocator_with_realloc that can handle realloc_or_die.
    template <class A>
    class alloc_impl<libc_allocator_with_realloc<A>> : public libc_allocator_with_realloc<A> {
    public:
        typedef typename libc_allocator_with_realloc<A>::pointer pointer;
        typedef typename libc_allocator_with_realloc<A>::size_type size_type;

        alloc_impl(const libc_allocator_with_realloc<A>& a) : libc_allocator_with_realloc<A>(a) {}

        pointer realloc_or_die(pointer ptr, size_type n) {
            pointer retval = this->reallocate(ptr, n);
            if (retval == NULL) {
                fprintf(stderr, "sparsehash: FATAL ERROR: failed to reallocate %lu elements for ptr %p",
                        static_cast<unsigned long>(n), static_cast<void*>(ptr));
                exit(1);
            }
            return retval;
        }
    };

    // Package allocator with empty val to eliminate memory needed for
    // the zero-size allocator.
    // If new fields are added to this class, we should add them to
    // operator= and swap.
    class ValInfo : public alloc_impl<value_alloc_type> {
    public:
        typedef typename alloc_impl<value_alloc_type>::value_type value_type;

        ValInfo(const alloc_impl<value_alloc_type>& a) : alloc_impl<value_alloc_type>(a) {}
    };

    // Package functors with another class to eliminate memory needed for
    // zero-size functors.  Since ExtractKey and hasher's operator() might
    // have the same function signature, they must be packaged in
    // different classes.
    struct Settings : sh_hashtable_settings<key_type, hasher, size_type, HT_MIN_BUCKETS> {
        explicit Settings(const hasher& hf)
            : sh_hashtable_settings<key_type, hasher, size_type, HT_MIN_BUCKETS>(hf, HT_OCCUPANCY_PCT / 100.0f,
                                                                                 HT_EMPTY_PCT / 100.0f) {}
    };

    // Packages ExtractKey and SetKey functors.
    class KeyInfo : public ExtractKey, public SetKey, public EqualKey {
    public:
        KeyInfo(const ExtractKey& ek, const SetKey& sk, const EqualKey& eq)
            : ExtractKey(ek), SetKey(sk), EqualKey(eq) {}

        // We want to return the exact same type as ExtractKey: Key or const Key&
        template <typename V>
        typename ExtractKey::result_type get_key(V&& v) const {
            return ExtractKey::operator()(std::forward<V>(v));
        }
        void set_key(pointer v, const key_type& k) const { SetKey::operator()(v, k); }
        void construct_key(pointer v, const key_type& k) const { SetKey::operator()(v, k, true); }
        bool equals(const key_type& a, const key_type& b) const { return EqualKey::operator()(a, b); }

        // Which key marks deleted entries.
        // TODO(csilvers): make a pointer, and get rid of use_deleted (benchmark!)
        typename std::remove_const<key_type>::type del_key;
        typename std::remove_const<key_type>::type empty_key;
    };

    // Utility functions to access the templated operators
    size_type hash(const key_type& v) const { return settings.hash(v); }
    bool equals(const key_type& a, const key_type& b) const { return key_info.equals(a, b); }
    template <typename V>
    typename ExtractKey::result_type get_key(V&& v) const {
        return key_info.get_key(std::forward<V>(v));
    }
    void set_key(pointer v, const key_type& k) const { key_info.set_key(v, k); }
    void construct_key(pointer v, const key_type& k) const { key_info.construct_key(v, k); }

private:
    // Actual data
    Settings settings;
    KeyInfo key_info;

    size_type num_deleted;  // how many occupied buckets are marked deleted
    size_type num_elements;
    size_type num_buckets;
    ValInfo val_info;  // holds emptyval, and also the allocator
    pointer table;
};

// We need a global swap as well
template <class V, class K, class HF, class ExK, class SetK, class EqK, class A>
inline void swap(dense_hashtable<V, K, HF, ExK, SetK, EqK, A>& x, dense_hashtable<V, K, HF, ExK, SetK, EqK, A>& y) {
    x.swap(y);
}

#undef JUMP_

template <class V, class K, class HF, class ExK, class SetK, class EqK, class A>
const typename dense_hashtable<V, K, HF, ExK, SetK, EqK, A>::size_type
    dense_hashtable<V, K, HF, ExK, SetK, EqK, A>::ILLEGAL_BUCKET;

// How full we let the table get before we resize.  Knuth says .8 is
// good -- higher causes us to probe too much, though saves memory.
// However, we go with .5, getting better performance at the cost of
// more space (a trade-off densehashtable explicitly chooses to make).
// Feel free to play around with different values, though, via
// max_load_factor() and/or set_resizing_parameters().
template <class V, class K, class HF, class ExK, class SetK, class EqK, class A>
const int dense_hashtable<V, K, HF, ExK, SetK, EqK, A>::HT_OCCUPANCY_PCT = 50;

// How empty we let the table get before we resize lower.
// It should be less than OCCUPANCY_PCT / 2 or we thrash resizing.
template <class V, class K, class HF, class ExK, class SetK, class EqK, class A>
const int dense_hashtable<V, K, HF, ExK, SetK, EqK, A>::HT_EMPTY_PCT =
    static_cast<int>(0.4 * dense_hashtable<V, K, HF, ExK, SetK, EqK, A>::HT_OCCUPANCY_PCT);

template <class Key, class T, class HashFcn = std::hash<Key>, class EqualKey = std::equal_to<Key>,
          class Alloc = libc_allocator_with_realloc<std::pair<const Key, T>>>
class dense_hash_map {
private:
    // Apparently select1st is not stl-standard, so we define our own
    struct SelectKey {
        typedef const Key& result_type;
        template <typename Pair>
        const Key& operator()(Pair&& p) const {
            return p.first;
        }
    };
    struct SetKey {
        void operator()(std::pair<const Key, T>* value, const Key& new_key) const {
            using NCKey = typename std::remove_cv<Key>::type;
            *const_cast<NCKey*>(&value->first) = new_key;

            // It would be nice to clear the rest of value here as well, in
            // case it's taking up a lot of memory.  We do this by clearing
            // the value.  This assumes T has a zero-arg constructor!
            value->second = T();
        }
        void operator()(std::pair<const Key, T>* value, const Key& new_key, bool) const {
            new (value) std::pair<const Key, T>(std::piecewise_construct, std::forward_as_tuple(new_key),
                                                std::forward_as_tuple());
        }
    };
    // The actual data
    typedef dense_hashtable<std::pair<const Key, T>, Key, HashFcn, SelectKey, SetKey, EqualKey, Alloc> ht;
    ht rep;

public:
    typedef typename ht::key_type key_type;
    typedef T data_type;
    typedef T mapped_type;
    typedef typename ht::value_type value_type;
    typedef typename ht::hasher hasher;
    typedef typename ht::key_equal key_equal;
    typedef Alloc allocator_type;

    typedef typename ht::size_type size_type;
    typedef typename ht::difference_type difference_type;
    typedef typename ht::pointer pointer;
    typedef typename ht::const_pointer const_pointer;
    typedef typename ht::reference reference;
    typedef typename ht::const_reference const_reference;

    typedef typename ht::iterator iterator;
    typedef typename ht::const_iterator const_iterator;
    typedef typename ht::local_iterator local_iterator;
    typedef typename ht::const_local_iterator const_local_iterator;

    // Iterator functions
    iterator begin() { return rep.begin(); }
    iterator end() { return rep.end(); }
    const_iterator begin() const { return rep.begin(); }
    const_iterator end() const { return rep.end(); }
    const_iterator cbegin() const { return rep.begin(); }
    const_iterator cend() const { return rep.end(); }

    // These come from tr1's unordered_map. For us, a bucket has 0 or 1 elements.
    local_iterator begin(size_type i) { return rep.begin(i); }
    local_iterator end(size_type i) { return rep.end(i); }
    const_local_iterator begin(size_type i) const { return rep.begin(i); }
    const_local_iterator end(size_type i) const { return rep.end(i); }
    const_local_iterator cbegin(size_type i) const { return rep.begin(i); }
    const_local_iterator cend(size_type i) const { return rep.end(i); }

    // Accessor functions
    allocator_type get_allocator() const { return rep.get_allocator(); }
    hasher hash_funct() const { return rep.hash_funct(); }
    hasher hash_function() const { return hash_funct(); }
    key_equal key_eq() const { return rep.key_eq(); }

    // Constructors
    explicit dense_hash_map(size_type expected_max_items_in_table = 0, const hasher& hf = hasher(),
                            const key_equal& eql = key_equal(), const allocator_type& alloc = allocator_type())
        : rep(expected_max_items_in_table, hf, eql, SelectKey(), SetKey(), alloc) {}

    template <class InputIterator>
    dense_hash_map(InputIterator f, InputIterator l, const key_type& empty_key_val,
                   size_type expected_max_items_in_table = 0, const hasher& hf = hasher(),
                   const key_equal& eql = key_equal(), const allocator_type& alloc = allocator_type())
        : rep(expected_max_items_in_table, hf, eql, SelectKey(), SetKey(), alloc) {
        set_empty_key(empty_key_val);
        rep.insert(f, l);
    }
    // We use the default copy constructor
    // We use the default operator=()
    // We use the default destructor

    void clear() { rep.clear(); }
    // This clears the hash map without resizing it down to the minimum
    // bucket count, but rather keeps the number of buckets constant
    void clear_no_resize() { rep.clear_no_resize(); }
    void swap(dense_hash_map& hs) { rep.swap(hs.rep); }

    // Functions concerning size
    size_type size() const { return rep.size(); }
    size_type max_size() const { return rep.max_size(); }
    bool empty() const { return rep.empty(); }
    size_type bucket_count() const { return rep.bucket_count(); }
    size_type max_bucket_count() const { return rep.max_bucket_count(); }

    // These are tr1 methods.  bucket() is the bucket the key is or would be in.
    size_type bucket_size(size_type i) const { return rep.bucket_size(i); }
    size_type bucket(const key_type& key) const { return rep.bucket(key); }
    float load_factor() const { return size() * 1.0f / bucket_count(); }
    float max_load_factor() const {
        float shrink, grow;
        rep.get_resizing_parameters(&shrink, &grow);
        return grow;
    }
    void max_load_factor(float new_grow) {
        float shrink, grow;
        rep.get_resizing_parameters(&shrink, &grow);
        rep.set_resizing_parameters(shrink, new_grow);
    }
    // These aren't tr1 methods but perhaps ought to be.
    float min_load_factor() const {
        float shrink, grow;
        rep.get_resizing_parameters(&shrink, &grow);
        return shrink;
    }
    void min_load_factor(float new_shrink) {
        float shrink, grow;
        rep.get_resizing_parameters(&shrink, &grow);
        rep.set_resizing_parameters(new_shrink, grow);
    }
    // Deprecated; use min_load_factor() or max_load_factor() instead.
    void set_resizing_parameters(float shrink, float grow) { rep.set_resizing_parameters(shrink, grow); }

    void resize(size_type hint) { rep.resize(hint); }
    void rehash(size_type hint) { resize(hint); }  // the tr1 name

    // Lookup routines
    iterator find(const key_type& key) { return rep.find(key); }
    const_iterator find(const key_type& key) const { return rep.find(key); }

    data_type& operator[](const key_type& key) {  // This is our value-add!
        // If key is in the hashtable, returns find(key)->second,
        // otherwise returns insert(value_type(key, T()).first->second.
        // Note it does not create an empty T unless the find fails.
        return rep.template find_or_insert<data_type>(key).second;
    }

    data_type& operator[](key_type&& key) { return rep.template find_or_insert<data_type>(std::move(key)).second; }

    size_type count(const key_type& key) const { return rep.count(key); }

    std::pair<iterator, iterator> equal_range(const key_type& key) { return rep.equal_range(key); }
    std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const { return rep.equal_range(key); }

    // Insertion routines
    std::pair<iterator, bool> insert(const value_type& obj) { return rep.insert(obj); }

    template <typename Pair, typename = typename std::enable_if<std::is_constructible<value_type, Pair&&>::value>::type>
    std::pair<iterator, bool> insert(Pair&& obj) {
        return rep.insert(std::forward<Pair>(obj));
    }

    // overload to allow {} syntax: .insert( { {key}, {args} } )
    std::pair<iterator, bool> insert(value_type&& obj) { return rep.insert(std::move(obj)); }

    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return rep.emplace(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace_hint(const_iterator hint, Args&&... args) {
        return rep.emplace_hint(hint, std::forward<Args>(args)...);
    }

    template <class InputIterator>
    void insert(InputIterator f, InputIterator l) {
        rep.insert(f, l);
    }
    void insert(const_iterator f, const_iterator l) { rep.insert(f, l); }
    void insert(std::initializer_list<value_type> ilist) { rep.insert(ilist.begin(), ilist.end()); }
    // Required for std::insert_iterator; the passed-in iterator is ignored.
    iterator insert(const_iterator, const value_type& obj) { return insert(obj).first; }
    iterator insert(const_iterator, value_type&& obj) { return insert(std::move(obj)).first; }
    template <class P, class = typename std::enable_if<std::is_constructible<value_type, P&&>::value &&
                                                       !std::is_same<value_type, P>::value>::type>
    iterator insert(const_iterator, P&& obj) {
        return insert(std::forward<P>(obj)).first;
    }

    // Deletion and empty routines
    // THESE ARE NON-STANDARD!  I make you specify an "impossible" key
    // value to identify deleted and empty buckets.  You can change the
    // deleted key as time goes on, or get rid of it entirely to be insert-only.
    // YOU MUST CALL THIS!
    void set_empty_key(const key_type& key) { rep.set_empty_key(key); }
    key_type empty_key() const { return rep.empty_key(); }

    void set_deleted_key(const key_type& key) { rep.set_deleted_key(key); }
    void clear_deleted_key() { rep.clear_deleted_key(); }
    key_type deleted_key() const { return rep.deleted_key(); }

    // These are standard
    size_type erase(const key_type& key) { return rep.erase(key); }
    iterator erase(const_iterator it) { return rep.erase(it); }
    iterator erase(const_iterator f, const_iterator l) { return rep.erase(f, l); }

    // Comparison
    bool operator==(const dense_hash_map& hs) const { return rep == hs.rep; }
    bool operator!=(const dense_hash_map& hs) const { return rep != hs.rep; }
};

// We need a global swap as well
template <class Key, class T, class HashFcn, class EqualKey, class Alloc>
inline void swap(dense_hash_map<Key, T, HashFcn, EqualKey, Alloc>& hm1,
                 dense_hash_map<Key, T, HashFcn, EqualKey, Alloc>& hm2) {
    hm1.swap(hm2);
}

template <class Value, class HashFcn = std::hash<Value>, class EqualKey = std::equal_to<Value>,
          class Alloc = libc_allocator_with_realloc<Value>>
class dense_hash_set {
private:
    // Apparently identity is not stl-standard, so we define our own
    struct Identity {
        typedef const Value& result_type;
        template <typename V>
        const Value& operator()(V&& v) const {
            return v;
        }
    };
    struct SetKey {
        void operator()(Value* value, const Value& new_key) const { *value = new_key; }
        void operator()(Value* value, const Value& new_key, bool) const { new (value) Value(new_key); }
    };

    // The actual data
    typedef dense_hashtable<Value, Value, HashFcn, Identity, SetKey, EqualKey, Alloc> ht;
    ht rep;

public:
    typedef typename ht::key_type key_type;
    typedef typename ht::value_type value_type;
    typedef typename ht::hasher hasher;
    typedef typename ht::key_equal key_equal;
    typedef Alloc allocator_type;

    typedef typename ht::size_type size_type;
    typedef typename ht::difference_type difference_type;
    typedef typename ht::const_pointer pointer;
    typedef typename ht::const_pointer const_pointer;
    typedef typename ht::const_reference reference;
    typedef typename ht::const_reference const_reference;

    typedef typename ht::const_iterator iterator;
    typedef typename ht::const_iterator const_iterator;
    typedef typename ht::const_local_iterator local_iterator;
    typedef typename ht::const_local_iterator const_local_iterator;

    // Iterator functions -- recall all iterators are const
    iterator begin() const { return rep.begin(); }
    iterator end() const { return rep.end(); }
    const_iterator cbegin() const { return rep.begin(); }
    const_iterator cend() const { return rep.end(); }

    // These come from tr1's unordered_set. For us, a bucket has 0 or 1 elements.
    local_iterator begin(size_type i) const { return rep.begin(i); }
    local_iterator end(size_type i) const { return rep.end(i); }
    local_iterator cbegin(size_type i) const { return rep.begin(i); }
    local_iterator cend(size_type i) const { return rep.end(i); }

    // Accessor functions
    allocator_type get_allocator() const { return rep.get_allocator(); }
    hasher hash_funct() const { return rep.hash_funct(); }
    hasher hash_function() const { return hash_funct(); }  // tr1 name
    key_equal key_eq() const { return rep.key_eq(); }

    // Constructors
    explicit dense_hash_set(size_type expected_max_items_in_table = 0, const hasher& hf = hasher(),
                            const key_equal& eql = key_equal(), const allocator_type& alloc = allocator_type())
        : rep(expected_max_items_in_table, hf, eql, Identity(), SetKey(), alloc) {}

    template <class InputIterator>
    dense_hash_set(InputIterator f, InputIterator l, const key_type& empty_key_val,
                   size_type expected_max_items_in_table = 0, const hasher& hf = hasher(),
                   const key_equal& eql = key_equal(), const allocator_type& alloc = allocator_type())
        : rep(expected_max_items_in_table, hf, eql, Identity(), SetKey(), alloc) {
        set_empty_key(empty_key_val);
        rep.insert(f, l);
    }
    // We use the default copy constructor
    // We use the default operator=()
    // We use the default destructor

    void clear() { rep.clear(); }
    // This clears the hash set without resizing it down to the minimum
    // bucket count, but rather keeps the number of buckets constant
    void clear_no_resize() { rep.clear_no_resize(); }
    void swap(dense_hash_set& hs) { rep.swap(hs.rep); }

    // Functions concerning size
    size_type size() const { return rep.size(); }
    size_type max_size() const { return rep.max_size(); }
    bool empty() const { return rep.empty(); }
    size_type bucket_count() const { return rep.bucket_count(); }
    size_type max_bucket_count() const { return rep.max_bucket_count(); }

    // These are tr1 methods.  bucket() is the bucket the key is or would be in.
    size_type bucket_size(size_type i) const { return rep.bucket_size(i); }
    size_type bucket(const key_type& key) const { return rep.bucket(key); }
    float load_factor() const { return size() * 1.0f / bucket_count(); }
    float max_load_factor() const {
        float shrink, grow;
        rep.get_resizing_parameters(&shrink, &grow);
        return grow;
    }
    void max_load_factor(float new_grow) {
        float shrink, grow;
        rep.get_resizing_parameters(&shrink, &grow);
        rep.set_resizing_parameters(shrink, new_grow);
    }
    // These aren't tr1 methods but perhaps ought to be.
    float min_load_factor() const {
        float shrink, grow;
        rep.get_resizing_parameters(&shrink, &grow);
        return shrink;
    }
    void min_load_factor(float new_shrink) {
        float shrink, grow;
        rep.get_resizing_parameters(&shrink, &grow);
        rep.set_resizing_parameters(new_shrink, grow);
    }
    // Deprecated; use min_load_factor() or max_load_factor() instead.
    void set_resizing_parameters(float shrink, float grow) { rep.set_resizing_parameters(shrink, grow); }

    void resize(size_type hint) { rep.resize(hint); }
    void rehash(size_type hint) { resize(hint); }  // the tr1 name

    // Lookup routines
    iterator find(const key_type& key) const { return rep.find(key); }

    size_type count(const key_type& key) const { return rep.count(key); }

    std::pair<iterator, iterator> equal_range(const key_type& key) const { return rep.equal_range(key); }

    // Insertion routines
    std::pair<iterator, bool> insert(const value_type& obj) {
        std::pair<typename ht::iterator, bool> p = rep.insert(obj);
        return std::pair<iterator, bool>(p.first, p.second);  // const to non-const
    }

    std::pair<iterator, bool> insert(value_type&& obj) {
        std::pair<typename ht::iterator, bool> p = rep.insert(std::move(obj));
        return std::pair<iterator, bool>(p.first, p.second);  // const to non-const
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return rep.emplace(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace_hint(const_iterator hint, Args&&... args) {
        return rep.emplace_hint(hint, std::forward<Args>(args)...);
    }

    template <class InputIterator>
    void insert(InputIterator f, InputIterator l) {
        rep.insert(f, l);
    }
    void insert(const_iterator f, const_iterator l) { rep.insert(f, l); }
    void insert(std::initializer_list<value_type> ilist) { rep.insert(ilist.begin(), ilist.end()); }
    // Required for std::insert_iterator; the passed-in iterator is ignored.
    iterator insert(const_iterator, const value_type& obj) { return insert(obj).first; }
    iterator insert(const_iterator, value_type&& obj) { return insert(std::move(obj)).first; }

    // Deletion and empty routines
    // THESE ARE NON-STANDARD!  I make you specify an "impossible" key
    // value to identify deleted and empty buckets.  You can change the
    // deleted key as time goes on, or get rid of it entirely to be insert-only.
    void set_empty_key(const key_type& key) { rep.set_empty_key(key); }
    key_type empty_key() const { return rep.empty_key(); }

    void set_deleted_key(const key_type& key) { rep.set_deleted_key(key); }
    void clear_deleted_key() { rep.clear_deleted_key(); }
    key_type deleted_key() const { return rep.deleted_key(); }

    // These are standard
    size_type erase(const key_type& key) { return rep.erase(key); }
    iterator erase(const_iterator it) { return rep.erase(it); }
    iterator erase(const_iterator f, const_iterator l) { return rep.erase(f, l); }

    // Comparison
    bool operator==(const dense_hash_set& hs) const { return rep == hs.rep; }
};

template <class Val, class HashFcn, class EqualKey, class Alloc>
inline void swap(dense_hash_set<Val, HashFcn, EqualKey, Alloc>& hs1,
                 dense_hash_set<Val, HashFcn, EqualKey, Alloc>& hs2) {
    hs1.swap(hs2);
}

}  // namespace flux

#endif
