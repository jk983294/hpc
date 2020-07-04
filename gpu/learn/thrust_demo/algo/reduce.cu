#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

struct my_less_than
{
    __host__ __device__
    bool operator()(int x)
    {
        return x < 1000000;
    }
};

int main()
{
    thrust::device_vector<int> X(2 << 16);

    std::generate(X.begin(), X.end(), rand);

    int sum = thrust::reduce(X.begin(), X.end(), (int) 0, thrust::plus<int>());
    int sum1 = thrust::reduce(X.begin(), X.end(), (int) 0);
    int sum2 = thrust::reduce(X.begin(), X.end());
    std::cout << "after thrust::reduce, sum=" << sum  << ","<< sum1 << ","<< sum2 << std::endl;

    int result = thrust::count(X.begin(), X.end(), 42);
    std::cout << "after thrust::reduce, count=" << result << std::endl;

    result = thrust::count_if(X.begin(), X.end(), my_less_than());
    std::cout << "after thrust::count_if value < value, count=" << result << std::endl;

    /**
     * need --expt-extended-lambda to compile
     */
    result = thrust::count_if(X.begin(), X.end(),  [] __device__ (int v) { return v < 100000; });
    std::cout << "after thrust::count_if value < value, count=" << result << std::endl;
    return 0;
}