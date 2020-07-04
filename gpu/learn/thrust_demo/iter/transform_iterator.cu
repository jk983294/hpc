#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

/**
 * kernel fusion, i.e. combining separate algorithms like transform and reduce into a single transform_reduce operation
 */

int main()
{
    thrust::device_vector<int> vec(10);
    thrust::sequence(vec.begin(), vec.end());

    auto first = thrust::make_transform_iterator(vec.begin(), thrust::negate<int>());
    auto last  = thrust::make_transform_iterator(vec.end(),   thrust::negate<int>());

    std::cout << first[0] << std::endl;   // returns 0
    std::cout << first[1] << std::endl;   // returns -1
    std::cout << first[2] << std::endl; // returns -2

    int res = thrust::reduce(first, last);   // returns -45 (i.e. 0 -1 -2 ...)
    std::cout << "result=" << res << std::endl;

    // avoid cumbersome to specify the full type of the iterator
    res = thrust::reduce(thrust::make_transform_iterator(vec.begin(), thrust::negate<int>()),
                   thrust::make_transform_iterator(vec.end(),   thrust::negate<int>()));
    std::cout << "result=" << res << std::endl;
    return 0;
}