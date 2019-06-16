#include <omp.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

double calc_a(int i) { return 1.1 * i; }
double calc_b(int i, const vector<double>& a) { return std::accumulate(a.begin(), a.end(), 0.0) * i; }

int main() {
    omp_set_num_threads(4);
    vector<double> a(4, 0), b(4, 0);

    #pragma omp parallel num_threads(4)
    {
        int id = omp_get_thread_num();
        a[id] = calc_a(id);

        /**
         * only one thread is master(its id is 0), it will execute below code, and no sync with other threads
         */
        #pragma omp master
        { cout << "i am the master " << id << endl; }

        /**
         * only one thread will execute below code, it doesn't need to be master
         * the fast thread reach here will execute
         * other threads will wait util that guy finish the job
         * nowait will remove the sync, other threads won't wait
         */
        #pragma omp single
        { cout << "i am the single " << id << endl; }

        #pragma omp barrier

        b[id] = calc_b(id, a);
    }

    for (auto i : a) std::cout << i << ' ';
    cout << endl;
    for (auto i : b) std::cout << i << ' ';
    cout << endl;
    cout << "done " << endl;
    return 0;
}
