#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

/**
 * permutation_iterator allows us to fuse gather and scatter operations
 *
 * When a permutation_iterator is used as an output sequence of a function it is equivalent to
 * fusing a scatter operation to the algorithm.
 * In general permutation_iterator allows you to operate on a specific set of values in a sequence
 * instead of the entire sequence.
 */

int main()
{
    // array to gather from
    thrust::device_vector<int> vec(10);
    thrust::sequence(vec.begin(), vec.end());

    // gather locations
    thrust::device_vector<int> map(4);
    map[0] = 3;
    map[1] = 1;
    map[2] = 0;
    map[3] = 5;

    // fuse gather with reduction:
    // sum = source[map[0]] + source[map[1]] + ...
    int sum = thrust::reduce(thrust::make_permutation_iterator(vec.begin(), map.begin()),
                         thrust::make_permutation_iterator(vec.begin(), map.end()));
    std::cout << "result=" << sum << std::endl;
    return 0;
}