#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)
#include <thrust/zip_function.h>
#endif // >= C++11
#include <algorithm>
#include <cstdlib>

/**
 * it takes multiple input sequences and yields a sequence of tuples
 *
 */

struct arbitrary_functor1
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<3>(t) = thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
    }
};

#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)
struct arbitrary_functor2
{
    __host__ __device__
    void operator()(const float& a, const float& b, const float& c, float& d)
    {
        // D[i] = A[i] + B[i] * C[i];
        d = a + b * c;
    }
};
#endif


int main()
{
    // initialize vectors
    thrust::device_vector<float> A(5);
    thrust::device_vector<float> B(5);
    thrust::device_vector<float> C(5);
    thrust::device_vector<float> D1(5);

    // initialize input vectors
    A[0] = 3;  B[0] = 6;  C[0] = 2;
    A[1] = 4;  B[1] = 7;  C[1] = 5;
    A[2] = 0;  B[2] = 2;  C[2] = 7;
    A[3] = 8;  B[3] = 1;  C[3] = 4;
    A[4] = 2; B[4] = 8; C[4] = 3;

    // apply the transformation
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D1.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end(),   C.end(),   D1.end())),
                     arbitrary_functor1());

    // print the output
    std::cout << "Tuple functor" << std::endl;
    for(int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D1[i] << std::endl;

#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)
    thrust::device_vector<float> D2(5);
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D2.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end(),   C.end(),   D2.end())),
                     thrust::make_zip_function(arbitrary_functor2()));

    // print the output
    std::cout << "N-ary functor" << std::endl;
    for(int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D2[i] << std::endl;
#endif
    return 0;
}