#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square {
    __host__ __device__
    T operator()(const T& x) const {
        return x * x;
    }
};

void norm2_demo() {
    /**
     * With thrust::transform_reduce we can apply kernel fusion to reduction kernels
     */
    thrust::device_vector<float> d_x(2 << 16);
    thrust::sequence(d_x.begin(), d_x.end());

    // setup arguments
    square<float>        unary_op;
    thrust::plus<float> binary_op;
    float init = 0;

    // compute norm
    float norm = std::sqrt( thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op) );

    std::cout << "norm=" << norm << std::endl;
}

int main()
{
    norm2_demo();
    return 0;
}