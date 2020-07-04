#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

/**
 * it takes multiple input sequences and yields a sequence of tuples
 *
 * The zip_iterator allows us to combine many independent sequences into a single sequence of tuples,
 * which can be processed by a broad set of algorithms.
 */

int main()
{
    // initialize vectors
    thrust::device_vector<int>  A(3);
    thrust::device_vector<char> B(3);
    A[0] = 10;  A[1] = 20;  A[2] = 30;
    B[0] = 'x'; B[1] = 'y'; B[2] = 'z';

    // create iterator (type omitted)
    auto first = thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin()));
    auto last  = thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end()));

    std::cout << first[0].get<0>() << "," << first[0].get<1>() << std::endl;   // returns tuple(10, 'x')
    std::cout << first[1].get<0>() << "," << first[1].get<1>() << std::endl;   // returns tuple(20, 'y')
    std::cout << first[2].get<0>() << "," << first[2].get<1>() << std::endl;   // returns tuple(30, 'z')

    // maximum of [first, last)
    thrust::maximum< thrust::tuple<int,char> > binary_op;
    thrust::tuple<int, char> init = first[0];
    auto ret = thrust::reduce(first, last, init, binary_op); // returns tuple(30, 'z')
    std::cout << ret.get<0>() << "," << ret.get<1>() << std::endl;
    return 0;
}