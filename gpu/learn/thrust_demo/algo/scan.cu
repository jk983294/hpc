#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

/**
 * Parallel prefix-sums, or scan operations
 */

int main()
{
    thrust::device_vector<int> X(2 << 12);
    thrust::device_vector<int> Y(2 << 12);

    thrust::sequence(X.begin(), X.end());
    thrust::fill(Y.begin(), Y.end(), 0);

    thrust::inclusive_scan(X.begin(), X.end(), Y.begin());
    std::cout << "data=" << X[0]  << ","<< X[1] << ","<< X[2] << ","<< X[3] << std::endl;
    std::cout << "inclusive_scan result=" << Y[0]  << ","<< Y[1] << ","<< Y[2] << ","<< Y[3] << std::endl;

    thrust::exclusive_scan(X.begin(), X.end(), Y.begin());
    std::cout << "exclusive_scan result=" << Y[0]  << ","<< Y[1] << ","<< Y[2] << ","<< Y[3] << std::endl;
    return 0;
}