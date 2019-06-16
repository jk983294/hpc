#include <omp.h>
#include <iostream>

using namespace std;

int main() {
    omp_set_num_threads(4);

    #pragma omp parallel  // implicitly get threads set in omp_set_num_threads which is 4
    {
        int id = omp_get_thread_num();
        cout << "hello " << id << endl;
    }

    #pragma omp parallel num_threads(8)  // explicitly ask 8 threads
    {
        int id = omp_get_thread_num();
        cout << "world " << id << endl;
    }

    cout << "done " << endl;
    return 0;
}
