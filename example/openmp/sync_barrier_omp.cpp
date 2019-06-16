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
         * barrier means all threads stop here, and continue when all ready
         */
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
