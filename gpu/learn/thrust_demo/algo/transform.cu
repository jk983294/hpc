#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

struct saxpy_functor {
    const float a;

    explicit saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
    float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};

void saxpy_fast(float a, thrust::device_vector<float>& X, thrust::device_vector<float>& Y, thrust::device_vector<float>& Z)
{
    // Z <- a * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), saxpy_functor(a));
}

void saxpy_slow(float a, thrust::device_vector<float>& X, thrust::device_vector<float>& Y, thrust::device_vector<float>& Z)
{
    thrust::device_vector<float> temp(X.size());

    // temp <- a
    thrust::fill(temp.begin(), temp.end(), a);

    // temp <- a * X
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

    // Z <- a * X + Y
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Z.begin(), thrust::plus<float>());
}

void saxpy_demo() {
    /**
     * fast_saxpy: performs 2N reads and N writes
     * slow_saxpy: performs 4N reads and 3N writes
     *
     * n memory bound algorithms like SAXPY it is generally worthwhile to apply kernel fusion
     * (combining multiple operations into a single kernel) to minimize the number of memory transactions
     */
    thrust::device_vector<float> X(2 << 16);
    thrust::device_vector<float> Y(2 << 16);
    thrust::device_vector<float> Z(2 << 16);
    std::generate(X.begin(), X.end(), rand);
    std::generate(Y.begin(), Y.end(), rand);

    saxpy_fast(5, X, Y, Z);
    std::cout << "after saxpy_fast, value=" << Z[0] << "," << Z[1] << "," << Z[2] << std::endl;
    saxpy_slow(5, X, Y, Z);
    std::cout << "after saxpy_slow, value=" << Z[0] << "," << Z[1] << "," << Z[2] << std::endl;
}

int main()
{
    // allocate three device_vectors with 10 elements
    thrust::device_vector<int> X(10);
    thrust::device_vector<int> Y(10);
    thrust::device_vector<int> Z(10);

    // initialize X to 0,1,2,3, ....
    thrust::sequence(X.begin(), X.end());
    std::cout << "after thrust::sequence, value=" << X[0] << "," << X[1] << "," << X[2] << std::endl;

    // compute Y = -X
    thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());
    std::cout << "after thrust::transform negate, value=" << Y[0] << "," << Y[1] << "," << Y[2] << std::endl;

    // fill Z with twos
    thrust::fill(Z.begin(), Z.end(), 2);
    std::cout << "after thrust::fill negate, value=" << Z[0] << "," << Z[1] << "," << Z[2] << std::endl;

    // compute Y = X mod 2
    thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>());
    std::cout << "after thrust::transform Y = X % Z, value=" << Y[0] << "," << Y[1] << "," << Y[2] << std::endl;

    // replace all the ones in Y with tens
    thrust::replace(Y.begin(), Y.end(), 1, 10);
    std::cout << "after thrust::replace Y = 10 if value == 1, value=";

    // print Y
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, ","));
    std::cout << std::endl;

    saxpy_demo();
    return 0;
}