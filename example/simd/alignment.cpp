#include <iomanip>
#include <iostream>

using namespace std;

int main() {
    int N = 1024 * 1024 * 64;

    float* a;
    /**
     * posix_memalign allocates aligned data on the heap
     * gcc attribute allocates aligned data on the stack
     */
    posix_memalign((void**)&a, 32, N * sizeof(float));
    uint64_t addr = reinterpret_cast<uint64_t>(a);
    cout << addr << " align: " << (addr % 32) << endl;

    float* b = (float*)aligned_alloc(32, N * sizeof(float));
    addr = reinterpret_cast<uint64_t>(b);
    cout << addr << " align: " << (addr % 32) << endl;

    typename std::aligned_storage<sizeof(float) * 8, 32>::type data[1024];
    addr = reinterpret_cast<uint64_t>(data);
    cout << addr << " align: " << (addr % 32) << endl;
    return 0;
}
