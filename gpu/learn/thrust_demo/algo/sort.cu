#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

void basic() {
    // generate 32M random numbers serially
    thrust::host_vector<int> h_vec(32 << 20);
    std::generate(h_vec.begin(), h_vec.end(), rand);
    std::cout << "before data=" << h_vec[0]  << ","<< h_vec[1] << ","<< h_vec[2] << ","<< h_vec[3] << std::endl;

    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;

    // sort data on the device (846M keys per second on GeForce GTX 480)
    thrust::sort(d_vec.begin(), d_vec.end());

    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    std::cout << "after data=" << h_vec[0]  << ","<< h_vec[1] << ","<< h_vec[2] << ","<< h_vec[3] << std::endl;
}

void reverse_sort() {
    thrust::host_vector<int> h_vec(32 << 20);
    std::generate(h_vec.begin(), h_vec.end(), rand);
    std::cout << "before data=" << h_vec[0]  << ","<< h_vec[1] << ","<< h_vec[2] << ","<< h_vec[3] << std::endl;

    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;

    thrust::sort(d_vec.begin(), d_vec.end(), thrust::greater<int>());

    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    std::cout << "after reverse_sort data=" << h_vec[0]  << ","<< h_vec[1] << ","<< h_vec[2] << ","<< h_vec[3] << std::endl;
}

void key_sort() {
    thrust::device_vector<int> d_keys(2 << 16);
    thrust::device_vector<int> d_values(2 << 16);
    std::generate(d_keys.begin(), d_keys.end(), rand);
    std::generate(d_values.begin(), d_values.end(), rand);
    std::cout << "before key=" << d_keys[0]  << ","<< d_keys[1] << ","<< d_keys[2] << ","<< d_keys[3] << std::endl;
    std::cout << "before data=" << d_values[0]  << ","<< d_values[1] << ","<< d_values[2] << ","<< d_values[3] << std::endl;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    std::cout << "after key_sort key=" << d_keys[0]  << ","<< d_keys[1] << ","<< d_keys[2] << ","<< d_keys[3] << std::endl;
    std::cout << "after key_sort data=" << d_values[0]  << ","<< d_values[1] << ","<< d_values[2] << ","<< d_values[3] << std::endl;
}

int main()
{

    basic();
    reverse_sort();
    key_sort();
    return 0;
}