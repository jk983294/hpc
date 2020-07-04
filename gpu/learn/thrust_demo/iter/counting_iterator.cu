#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

/**
 * If a sequence of increasing values is required, then counting_iterator is the appropriate choice
 *
 * While constant_iterator and counting_iterator act as arrays, they don’t actually require any memory storage.
 * Whenever we dereference one of these iterators it generates the appropriate value on-the-fly and
 * returns it to the calling function
 */

int main()
{
    // create iterators
    thrust::counting_iterator<int> first(10);
    thrust::counting_iterator<int> last = first + 3;

    std::cout << first[0] << std::endl;   // returns 10
    std::cout << first[1] << std::endl;   // returns 11
    std::cout << first[100] << std::endl; // returns 110

    int res = thrust::reduce(first, last);   // returns 33 (i.e. 10 + 11 + 12)
    std::cout << "result=" << res << std::endl;
    return 0;
}