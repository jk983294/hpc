#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

/**
 * constant_iterator is simply an iterator that returns the same value whenever we access it
 * Whenever an input sequence of constant values is needed, constant_iterator is a convenient and efficient solution
 */

int main()
{
    // create iterators
    thrust::constant_iterator<int> first(10);
    thrust::constant_iterator<int> last = first + 3;

    std::cout << first[0] << std::endl;   // returns 10
    std::cout << first[1] << std::endl;   // returns 10
    std::cout << first[100] << std::endl; // returns 10

    int res = thrust::reduce(first, last);   // returns 30 (i.e. 3 * 10)
    std::cout << "result=" << res << std::endl;
    return 0;
}